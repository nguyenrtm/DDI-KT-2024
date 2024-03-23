import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, SAGPooling

class GNN(torch.nn.Module):
    def __init__(self, 
                 atom_embedding_dim: int = 16,
                 bond_embedding_dim: int = 8,
                 bool_embedding_dim: int = 2,
                 num_node_features: int = 14,
                 num_edge_features: int = 5,
                 atom_normalize_dim: int = 30,
                 bond_normalize_dim: int = 2,
                 hidden_channels: int = 128, 
                 dropout_rate: float = 0.2,
                 num_layers_gnn: int = 3,
                 gnn_option: str = 'GATV2CONV',
                 readout_option: str = 'global_max_pool',
                 activation_function: str = 'relu',
                 device: str = 'cpu'):
        
        super(GNN, self).__init__()
        self.device = device
        self.dropout = dropout_rate
        self.hidden_channels = hidden_channels

        self.atom_encoder = nn.Embedding(num_embeddings=119, embedding_dim=atom_embedding_dim, padding_idx=0)
        self.bond_encoder = nn.Embedding(num_embeddings=22, embedding_dim=bond_embedding_dim, padding_idx=0)
        self.boolean_encoder = nn.Embedding(num_embeddings=3, embedding_dim=bool_embedding_dim, padding_idx=2)
        self.gnn_option = gnn_option
        self.num_layers_gnn = num_layers_gnn
        self.readout_option = readout_option
        self.node_dim = atom_normalize_dim+atom_embedding_dim+bool_embedding_dim*3
        self.edge_dim = bond_normalize_dim+bond_embedding_dim+bool_embedding_dim*3

        if gnn_option == 'GATV2CONV':
            self.gnn1 = GATv2Conv(self.node_dim, 
                                  hidden_channels,
                                  edge_dim=self.edge_dim)
            self.gnn = GATv2Conv(hidden_channels, 
                                 hidden_channels,
                                 edge_dim=self.edge_dim)
        elif gnn_option == 'GCNCONV':
            self.gnn1 = GCNConv(self.node_dim, 
                                hidden_channels)
            self.gnn = GCNConv(hidden_channels, 
                               hidden_channels)
        elif gnn_option == 'ATTENTIVEFP':
            self.gnn = AttentiveFP(in_channels=self.node_dim,
                                   hidden_channels=hidden_channels,
                                   out_channels=hidden_channels,
                                   edge_dim=self.edge_dim,
                                   num_layers=3,
                                   num_timesteps=2,
                                   dropout=self.dropout)
        self.normalize_node = nn.Linear(num_node_features-4, atom_normalize_dim)
        self.normalize_edge = nn.Linear(num_edge_features-4, bond_normalize_dim)
            
        if activation_function == 'relu':
            self.act = F.relu
        if activation_function == 'leaky_relu':
            self.act = F.leaky_relu
            
        if readout_option == 'global_max_pool':
            self.readout_layer = global_max_pool
        elif readout_option == 'global_mean_pool':
            self.readout_layer = global_mean_pool
        elif readout_option == 'global_add_pool':
            self.readout_layer = global_add_pool
        elif readout_option == 'sag_pooling':
            self.readout_layer = SAGPooling(hidden_channels, 
                                            ratio=0.8,
                                            GNN=GCNConv)

    def forward(self, mol):
        if mol.mol == None:
            return torch.zeros([1, self.hidden_channels]).to(self.device)
        
        # Encoding categorical features
        x, edge_index, edge_attr, batch = mol.x, mol.edge_index, mol.edge_attr, mol.batch

        atomic_num0 = self.atom_encoder(x[:, 0].int()) # encode atom type
        atom_is_aromatic0 = self.boolean_encoder(x[:, -3].int()) # encode aromaticity
        atom_has_owning_mol = self.boolean_encoder(x[:, -2].int()) # encode owning mol
        atom_is_in_ring = self.boolean_encoder(x[:, -1].int()) # encode ring information

        bond_type0 = self.bond_encoder(edge_attr[:, 0].int()) # encode bond type
        bond_is_conjugated0 = self.boolean_encoder(edge_attr[:, -3].int()) # encode conjugation
        bond_is_aromatic0 = self.boolean_encoder(edge_attr[:, -2].int()) # encode aromaticity
        bond_is_in_ring = self.boolean_encoder(edge_attr[:, -1].int())

        x = torch.cat([atomic_num0, 
                       self.normalize_node(x[:, 1:11]), 
                       atom_is_aromatic0, 
                       atom_has_owning_mol,
                       atom_is_in_ring], dim=1)

        edge_attr = torch.cat([bond_type0, 
                               self.normalize_edge(edge_attr[:, 1:2]), 
                               bond_is_conjugated0, 
                               bond_is_aromatic0, 
                               bond_is_in_ring], dim=1)

        # GNN pass
        if self.gnn_option == 'ATTENTIVEFP':
            x = self.gnn(x, edge_index, edge_attr, batch)
        elif self.gnn_option == 'GATV2CONV':
            x = self.act(self.gnn1(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)

            for i in range(self.num_layers_gnn - 2):
                x = self.act(self.gnn(x, edge_index, edge_attr))
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.act(self.gnn(x, edge_index, edge_attr))
            x = self.readout_layer(x, batch)
        elif self.gnn_option == 'GCNCONV':
            x = self.act(self.gnn1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

            for i in range(self.num_layers_gnn - 2):
                x = self.act(self.gnn(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.act(self.gnn(x, edge_index))
            
            if self.readout_option == 'sag_pooling':
                x, edge_index, _, batch, _, _ = self.readout_layer(x, edge_index, edge_attr, batch)
                x = global_add_pool(x, batch)
            else:
                x = self.readout_layer(x, batch)
        
        return x
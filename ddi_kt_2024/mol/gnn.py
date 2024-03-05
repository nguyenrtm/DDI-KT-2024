import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, 
                 atom_embedding_dim: int = 64,
                 bond_embedding_dim: int = 16,
                 bool_embedding_dim: int = 2,
                 num_node_features: int = 10,
                 num_edge_features: int = 4,
                 hidden_channels: int = 512, 
                 dropout_rate: float = 0.2,
                 device: str = 'cpu'):
        
        super(GNN, self).__init__()
        self.device = device
        self.dropout = dropout_rate
        self.hidden_channels = hidden_channels

        self.atom_encoder = nn.Embedding(num_embeddings=119, embedding_dim=atom_embedding_dim, padding_idx=0)
        self.bond_encoder = nn.Embedding(num_embeddings=22, embedding_dim=bond_embedding_dim, padding_idx=0)
        self.boolean_encoder = nn.Embedding(num_embeddings=3, embedding_dim=bool_embedding_dim, padding_idx=2)

        self.conv1 = GATv2Conv(num_node_features-2+atom_embedding_dim+bool_embedding_dim, 
                               hidden_channels*4,
                               edge_dim=num_edge_features-3+bond_embedding_dim+bool_embedding_dim*2)
        
        self.conv2 = GATv2Conv(hidden_channels*4, 
                               hidden_channels*2,
                               edge_dim=num_edge_features-3+bond_embedding_dim+bool_embedding_dim*2)
        
        self.conv3 = GATv2Conv(hidden_channels*2, 
                               hidden_channels,
                               edge_dim=num_edge_features-3+bond_embedding_dim+bool_embedding_dim*2)
        
        self.conv4 = GATv2Conv(hidden_channels, 
                               hidden_channels,
                               edge_dim=num_edge_features-3+bond_embedding_dim+bool_embedding_dim*2)

    def forward(self, mol):
        if mol.mol == None:
            return torch.zeros([1, self.hidden_channels]).to(self.device)
        
        x, edge_index, edge_attr, batch = mol.x, mol.edge_index, mol.edge_attr, mol.batch

        atomic_num0 = self.atom_encoder(x[:, 0].int()) # encode atom type
        atom_is_aromatic0 = self.boolean_encoder(x[:, -1].int()) # encode aromaticity

        bond_type0 = self.bond_encoder(edge_attr[:, 0].int()) # encode bond type
        bond_is_conjugated0 = self.boolean_encoder(edge_attr[:, -2].int()) # encode conjugation
        bond_is_aromatic0 = self.boolean_encoder(edge_attr[:, -1].int()) # encode aromaticity

        x = torch.cat([atomic_num0, x[:, 1:9], atom_is_aromatic0], dim=1)
        edge_attr = torch.cat([bond_type0, edge_attr[:, 1:2], bond_is_conjugated0, bond_is_aromatic0], dim=1)

        # GNN pass
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        
        return x
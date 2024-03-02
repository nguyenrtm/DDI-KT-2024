import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from src.seed import MANUAL_SEED

class GCN(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int = 4, 
                 hidden_channels: int = 16, 
                 dropout_rate: float = 0.2,
                 device: str = 'cpu'):
        super(GCN, self).__init__()
        torch.manual_seed(MANUAL_SEED)
        self.device = device
        self.dropout = dropout_rate
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, mol):
        if mol == None:
            return torch.zeros([1, self.hidden_channels]).to(self.device)
        
        x, edge_index, batch = mol.x, mol.edge_index, mol.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        
        return x
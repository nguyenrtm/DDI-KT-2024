import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

MANUAL_SEED = 41


class GCN(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int = 4, 
                 hidden_channels: int = 16, 
                 num_classes: int = 5,
                 device: str = 'cpu'):
        super(GCN, self).__init__()
        torch.manual_seed(MANUAL_SEED)
        self.device = device
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, mol1, mol2):
        x_1, edge_index_1 = mol1.x, mol1.edge_index
        x_2, edge_index_2 = mol2.x, mol2.edge_index

        x_1 = self.conv1(x_1, edge_index_1)
        x_1 = F.relu(x_1)
        x_1 = F.dropout(x_1, p=0.2, training=self.training)
        x_1 = self.conv2(x_1, edge_index_1)
        x_1 = x_1.relu()
        x_1 = F.dropout(x_1, p=0.2, training=self.training)
        x_1 = self.conv3(x_1, edge_index_1)

        x_2 = self.conv1(x_2, edge_index_2)
        x_2 = F.relu(x_2)
        x_2 = F.dropout(x_2, p=0.2, training=self.training)
        x_2 = self.conv2(x_2, edge_index_2)
        x_2 = x_2.relu()
        x_2 = F.dropout(x_2, p=0.2, training=self.training)
        x_2 = self.conv3(x_2, edge_index_2)

        x = torch.cat([x_1, x_2], dim=0)
        
        # Readout layer
        batch = torch.zeros(x_1.size(0) + x_2.size(0), dtype=torch.long).to(self.device)
        x = global_mean_pool(x, batch)

        # Final classifier
        x = self.lin(x)
        
        return x
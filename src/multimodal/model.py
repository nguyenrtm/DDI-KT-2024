import torch
import torch.nn

from src.text.model.text_model import TextModel
from src.mol.gcn import GCN

class MultimodalModel(torch.nn.Module):
    def __init__(self, 
                 we,
                 dropout_rate: float = 0.5,
                 word_embedding_size: int = 200,
                 tag_number: int = 51,
                 tag_embedding_size: int = 50,
                 position_number: int = 4,
                 position_embedding_size: int = 50,
                 direction_number: int = 3,
                 direction_embedding_size: int = 50,
                 edge_number: int = 46,
                 edge_embedding_size: int = 200,
                 token_embedding_size: int = 500,
                 dep_embedding_size: int = 500,
                 conv1_out_channels: int = 256,
                 conv2_out_channels: int = 256,
                 conv3_out_channels: int = 256,
                 conv1_length: int = 1,
                 conv2_length: int = 2,
                 conv3_length: int = 3,
                 target_class: int = 5,
                 num_node_features: int = 4, 
                 hidden_channels: int = 256,
                 device: str = 'cpu'):
        super(MultimodalModel, self).__init__()
        self.device = device

        self.text_model = TextModel(we=we,
                                    dropout_rate=dropout_rate,
                                    word_embedding_size=word_embedding_size,
                                    tag_number=tag_number,
                                    tag_embedding_size=tag_embedding_size,
                                    position_number=position_number,
                                    position_embedding_size=position_embedding_size,
                                    direction_number=direction_number,
                                    direction_embedding_size=direction_embedding_size,
                                    edge_number=edge_number,
                                    edge_embedding_size=edge_embedding_size,
                                    token_embedding_size=token_embedding_size,
                                    dep_embedding_size=dep_embedding_size,
                                    conv1_out_channels=conv1_out_channels,
                                    conv2_out_channels=conv2_out_channels,
                                    conv3_out_channels=conv3_out_channels,
                                    conv1_length=conv1_length,
                                    conv2_length=conv2_length,
                                    conv3_length=conv3_length,
                                    target_class=target_class,
                                    classifier=False).to(device)

        self.gcn1 = GCN(num_node_features=num_node_features,
                       hidden_channels=hidden_channels,
                       dropout_rate=dropout_rate, 
                       device=device).to(device)
        
        self.gcn2 = GCN(num_node_features=num_node_features,
                        hidden_channels=hidden_channels,
                        dropout_rate=dropout_rate, 
                        device=device).to(device)

        self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+2*hidden_channels, 
                                            out_features=target_class,
                                            bias=False)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, text_x, mol_x1, mol_x2):
        text_x = self.text_model(text_x)
        mol_x1 = self.gcn1(mol_x1)
        mol_x2 = self.gcn2(mol_x2)

        x = torch.cat((text_x, mol_x1, mol_x2), dim=1)

        # Classifier
        x = self.dense_to_tag(x)
        x = self.softmax(x)

        return x
import torch
import torch.nn

from ddi_kt_2024.text.model.text_model import TextModel, BertModel
from ddi_kt_2024.mol.gnn import GNN

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
                 text_model: str = 'bert',
                 modal: str = 'multimodal',
                 device: str = 'cpu'):
        super(MultimodalModel, self).__init__()
        self.device = device

        if text_model == 'bert':
            self.text_model = BertModel(dropout_rate=dropout_rate,
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
        elif text_model == 'fasttext':
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

        self.gnn1 = GNN(num_node_features=num_node_features,
                       hidden_channels=hidden_channels,
                       dropout_rate=dropout_rate, 
                       device=device).to(device)
        
        self.gnn2 = GNN(num_node_features=num_node_features,
                        hidden_channels=hidden_channels,
                        dropout_rate=dropout_rate, 
                        device=device).to(device)

        self.modal = modal
        
        if self.modal == '0':
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels, 
                                                out_features=target_class,
                                                bias=False)
        elif self.modal == '1':
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+2*hidden_channels, 
                                                out_features=target_class,
                                                bias=False)
        elif self.modal == '2':
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+2*self.smiles_embedding.embedding_size, 
                                                out_features=target_class,
                                                bias=False)
        elif self.modal == '3':
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+2*hidden_channels+2*self.smiles_embedding.embedding_size, 
                                                out_features=target_class,
                                                bias=False)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, 
                text_x, 
                mol_x1 = None, 
                mol_x2 = None, 
                mol_x1_smiles = None, 
                mol_x2_smiles = None):
        if self.modal == '0':
            x = self.text_model(text_x)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '1':
            text_x = self.text_model(text_x)
            mol_x1 = self.gnn1(mol_x1)
            mol_x2 = self.gnn2(mol_x2)

            x = torch.cat((text_x, mol_x1, mol_x2), dim=1)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '2':
            text_x = self.text_model(text_x)
            mol_x1_smiles = self.smiles_embedding(mol_x1)
            mol_x2_smiles = self.smiles_embedding(mol_x2)

            x = torch.cat((text_x, mol_x1_smiles, mol_x2_smiles), dim=1)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '3':
            text_x = self.text_model(text_x)
            mol_x1 = self.gnn1(mol_x1)
            mol_x2 = self.gnn2(mol_x2)
            mol_x1_smiles = self.smiles_embedding(mol_x1)
            mol_x2_smiles = self.smiles_embedding(mol_x2)

            x = torch.cat((text_x, mol_x1, mol_x2, mol_x1_smiles, mol_x2_smiles), dim=1)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
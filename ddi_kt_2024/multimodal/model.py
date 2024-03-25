import torch
import torch.nn

from ddi_kt_2024.text.model.text_model import TextModel, BertModel
from ddi_kt_2024.mol.char_lstm import CharLSTM
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
                 gnn_option: str = 'GATV2CONV',
                 num_layers_gnn: str = 3,
                 readout_option: str = 'global_max_pool',
                 text_model_option: str = 'cnn',
                 activation_function: str = 'relu',
                 device: str = 'cpu',
                 position_embedding_type: str = 'linear',
                 **kwargs):
        super(MultimodalModel, self).__init__()
        self.device = device
        self.text_modal_size = kwargs['text_modal_size']
        self.graph_modal_size = kwargs['graph_modal_size']

        if text_model == 'bert':
            if modal == '1_early_fusion':
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
                                            model_option=text_model_option,
                                            classifier=False,
                                            with_fusion=True,
                                            position_embedding_type=position_embedding_type,
                                            device=device,
                                            **kwargs).to(device)
            else:
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
                                            model_option=text_model_option,
                                            classifier=False,
                                            with_fusion=False,
                                            position_embedding_type=position_embedding_type,
                                            device=device,
                                            **kwargs).to(device)

        self.gnn1 = GNN(num_node_features=num_node_features,
                        hidden_channels=hidden_channels,
                        dropout_rate=dropout_rate, 
                        gnn_option=gnn_option,
                        num_layers_gnn=num_layers_gnn,
                        readout_option=readout_option,
                        activation_function=activation_function,
                        device=device).to(device)
        
        self.gnn2 = GNN(num_node_features=num_node_features,
                        hidden_channels=hidden_channels,
                        dropout_rate=dropout_rate, 
                        gnn_option=gnn_option,
                        num_layers_gnn=num_layers_gnn,
                        readout_option=readout_option,
                        activation_function=activation_function,
                        device=device).to(device)

        self.modal = modal
        
        if self.modal == '0' or self.modal == '1_early_fusion':
            if text_model_option == 'bilstm':
                self.dense_to_tag = torch.nn.Linear(in_features=kwargs['lstm_hidden_size']*2, 
                                                    out_features=target_class,
                                                    bias=False)
            elif text_model_option == 'lstm':
                self.dense_to_tag = torch.nn.Linear(in_features=kwargs['lstm_hidden_size'], 
                                                    out_features=target_class,
                                                    bias=False)       
            else:
                self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels, 
                                                    out_features=target_class,
                                                    bias=False)
            
            if kwargs['norm'] == 'batch_norm':
                self.norm_text = torch.nn.BatchNorm1d(num_features=conv1_out_channels+conv2_out_channels+conv3_out_channels)
            elif kwargs['norm'] == 'layer_norm':
                self.norm_text = torch.nn.LayerNorm(normalized_shape=conv1_out_channels+conv2_out_channels+conv3_out_channels)
        elif self.modal[0] == '1':
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+2*hidden_channels, 
                                                out_features=target_class,
                                                bias=False)
            
            if kwargs['norm'] == 'batch_norm':
                self.norm_text = torch.nn.BatchNorm1d(num_features=conv1_out_channels+conv2_out_channels+conv3_out_channels)
                self.norm_g1= torch.nn.BatchNorm1d(num_features=hidden_channels)
                self.norm_g2= torch.nn.BatchNorm1d(num_features=hidden_channels)
            elif kwargs['norm'] == 'layer_norm':
                self.norm_text = torch.nn.LayerNorm(normalized_shape=conv1_out_channels+conv2_out_channels+conv3_out_channels)
                self.norm_g1= torch.nn.LayerNorm(normalized_shape=hidden_channels)
                self.norm_g2= torch.nn.LayerNorm(normalized_shape=hidden_channels)
        elif self.modal == '2':
            self.char_lstm = CharLSTM(hidden_dim=32,
                                      output_dim=16,
                                      device=device).to(device)
            
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+2*self.char_lstm.output_dim, 
                                                out_features=target_class,
                                                bias=False)
            if kwargs['norm'] == 'batch_norm':
                self.norm_text = torch.nn.BatchNorm1d(num_features=conv1_out_channels+conv2_out_channels+conv3_out_channels)
                self.norm_f1= torch.nn.BatchNorm1d(num_features=self.char_lstm.output_dim)
                self.norm_f2= torch.nn.BatchNorm1d(num_features=self.char_lstm.output_dim)
            elif kwargs['norm'] == 'layer_norm':
                self.norm_text = torch.nn.LayerNorm(normalized_shape=conv1_out_channels+conv2_out_channels+conv3_out_channels)
                self.norm_f1= torch.nn.LayerNorm(normalized_shape=self.char_lstm.output_dim)
                self.norm_f2= torch.nn.LayerNorm(normalized_shape=self.char_lstm.output_dim)

        elif self.modal == '3':
            self.char_lstm = CharLSTM(hidden_dim=32,
                                      output_dim=16,
                                      device=device).to(device)
            
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+2*hidden_channels+2*self.char_lstm.output_dim, 
                                                out_features=target_class,
                                                bias=False)
            
            if kwargs['norm'] == 'batch_norm':
                self.norm_text = torch.nn.BatchNorm1d(num_features=conv1_out_channels+conv2_out_channels+conv3_out_channels)
                self.norm_g1 = torch.nn.BatchNorm1d(num_features=hidden_channels)
                self.norm_g2 = torch.nn.BatchNorm1d(num_features=hidden_channels)
                self.norm_f1= torch.nn.BatchNorm1d(num_features=self.char_lstm.output_dim)
                self.norm_f2= torch.nn.BatchNorm1d(num_features=self.char_lstm.output_dim)
            elif kwargs['norm'] == 'layer_norm':
                self.norm_text = torch.nn.LayerNorm(normalized_shape=conv1_out_channels+conv2_out_channels+conv3_out_channels)
                self.norm_g1 = torch.nn.LayerNorm(normalized_shape=hidden_channels)
                self.norm_g2 = torch.nn.LayerNorm(normalized_shape=hidden_channels)
                self.norm_f1= torch.nn.LayerNorm(normalized_shape=self.char_lstm.output_dim)
                self.norm_f2= torch.nn.LayerNorm(normalized_shape=self.char_lstm.output_dim)
                
        elif self.modal == 'gnn_only' or self.modal == 'gnn_only.2':
            self.dense_to_tag = torch.nn.Linear(in_features=2*hidden_channels, 
                                                out_features=target_class,
                                                bias=False)
            if kwargs['norm'] == 'batch_norm':
                self.norm_g1= torch.nn.BatchNorm1d(num_features=hidden_channels)
                self.norm_g2= torch.nn.BatchNorm1d(num_features=hidden_channels)
            elif kwargs['norm'] == 'layer_norm':
                self.norm_g1= torch.nn.LayerNorm(normalized_shape=hidden_channels)
                self.norm_g2= torch.nn.LayerNorm(normalized_shape=hidden_channels)

        self.softmax = torch.nn.Softmax(dim=1)
        
    def custom_concat_fusion(self, text_batch_vector, graph1_batch_vector, graph2_batch_vector):
        shape = text_batch_vector.shape
        zeros = torch.zeros((1, graph1_batch_vector.shape[1])).to(self.device)
        full_graph_batch_vector = list()
        for i in range(shape[0]):
            graph_batch_vector = list()
            for row in text_batch_vector[i]:
                if len(row.shape) >= 1:
                    if row[4] == torch.tensor(1) and row[13] == torch.tensor(1):
                        graph_batch_vector.append(torch.cat((graph1_batch_vector[i].unsqueeze(dim=0), graph2_batch_vector[i].unsqueeze(dim=0)), dim=1))
                    elif row[4] == torch.tensor(1):
                        graph_batch_vector.append(torch.cat((graph1_batch_vector[i].unsqueeze(dim=0), zeros), dim=1))
                    elif row[13] == torch.tensor(1):
                        graph_batch_vector.append(torch.cat((zeros, graph2_batch_vector[i].unsqueeze(dim=0)), dim=1))
                    else:
                        graph_batch_vector.append(torch.cat((zeros, zeros), dim=1))
            full_graph_batch_vector.append(torch.cat(graph_batch_vector).unsqueeze(dim=0))
        graph_batch_vector = torch.cat(full_graph_batch_vector)
        full_batch_vector = torch.cat((text_batch_vector, graph_batch_vector), dim=2)
        return full_batch_vector

    def forward(self, text_x, **kwargs):
        if self.modal == '0':
            x = self.text_model(text_x)
            x = self.norm_text(x)

            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '1':
            mol_x1 = kwargs['mol_x1']
            mol_x2 = kwargs['mol_x2']

            text_x = self.text_model(text_x)
            mol_x1 = self.gnn1(mol_x1)
            mol_x2 = self.gnn2(mol_x2)

            # text_x = self.norm_text(text_x)
            # mol_x1 = self.norm_g1(mol_x1)
            # mol_x2 = self.norm_g2(mol_x2)

            x = torch.cat((text_x, mol_x1, mol_x2), dim=1)

            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '2':
            mol_x1_formula = kwargs['mol_x1_formula']
            mol_x2_formula = kwargs['mol_x2_formula']
        
            text_x = self.text_model(text_x)
            mol_x1_formula = self.char_lstm(mol_x1_formula)
            mol_x2_formula = self.char_lstm(mol_x2_formula)

            text_x = self.norm_text(text_x)
            mol_x1_formula = self.norm_f1(mol_x1_formula)
            mol_x2_formula = self.norm_f2(mol_x2_formula)

            x = torch.cat((text_x, mol_x1_formula, mol_x2_formula), dim=1)

            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '3':
            mol_x1 = kwargs['mol_x1']
            mol_x2 = kwargs['mol_x2']
            mol_x1_formula = kwargs['mol_x1_formula']
            mol_x2_formula = kwargs['mol_x2_formula']
            
            text_x = self.text_model(text_x)
            mol_x1 = self.gnn1(mol_x1)
            mol_x2 = self.gnn2(mol_x2)
            mol_x1_formula = self.char_lstm(mol_x1_formula)
            mol_x2_formula = self.char_lstm(mol_x2_formula)

            text_x = self.norm_text(text_x)
            mol_x1 = self.norm_g1(mol_x1)
            mol_x2 = self.norm_g2(mol_x2)
            mol_x1_formula = self.norm_f1(mol_x1_formula)
            mol_x2_formula = self.norm_f2(mol_x2_formula)
            
            x = torch.cat((text_x, mol_x1, mol_x2, mol_x1_formula, mol_x2_formula), dim=1)

            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == 'gnn_only':
            mol_x1 = self.gnn1(mol_x1)
            mol_x2 = self.gnn2(mol_x2)

            mol_x1 = self.norm_g1(mol_x1)
            mol_x2 = self.norm_g2(mol_x2)

            x = torch.cat((mol_x1, mol_x2), dim=1)

            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == 'gnn_only.2':
            mol_x1 = self.gnn1(mol_x1)
            mol_x2 = self.gnn1(mol_x2)

            mol_x1 = self.norm_g1(mol_x1)
            mol_x2 = self.norm_g1(mol_x2)

            x = torch.cat((mol_x1, mol_x2), dim=1)

            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '1_early_fusion':
            mol_x1 = self.gnn1(mol_x1)
            mol_x2 = self.gnn2(mol_x2)
            
            text_x = self.custom_concat_fusion(text_x, mol_x1, mol_x2)
            x = self.text_model(text_x)

            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
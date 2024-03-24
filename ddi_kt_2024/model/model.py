import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from ddi_kt_2024.embed.other_embed import sinusoidal_positional_embedding

class Model(nn.Module):
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
                 target_class: int = 5
                 ):

        super(Model, self).__init__()

        self.w2v = nn.Embedding.from_pretrained(torch.tensor(we.vectors))
        self.tag_embedding = nn.Embedding(tag_number, tag_embedding_size, padding_idx=0)
        self.direction_embedding = nn.Embedding(direction_number, direction_embedding_size, padding_idx=0)
        self.edge_embedding = nn.Embedding(edge_number, edge_embedding_size, padding_idx=0)

        self.normalize_position = nn.Linear(in_features=position_number,
                                            out_features=position_embedding_size,
                                            bias=False)
        
        self.normalize_tokens = nn.Linear(in_features=word_embedding_size+tag_embedding_size+position_embedding_size,
                                          out_features=token_embedding_size,
                                          bias=False)
        
        self.normalize_dep = nn.Linear(in_features=direction_embedding_size+edge_embedding_size,
                                       out_features=dep_embedding_size,
                                       bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv1_out_channels,
                      kernel_size=(conv1_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv2_out_channels,
                      kernel_size=(conv2_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv3_out_channels,
                      kernel_size=(conv3_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.relu = nn.ReLU()
        self.dense_to_tag = nn.Linear(in_features=conv1_out_channels + conv2_out_channels + conv3_out_channels,
                                      out_features=target_class,
                                      bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        word_embedding_ent1 = self.w2v(x[:, :, 0])
        tag_embedding_ent1 = self.tag_embedding(x[:, :, 1])
        position_embedding_ent1 = self.normalize_position(x[:, :, 2:6].float())
        position_embedding_ent1 = position_embedding_ent1

        direction_embedding = self.direction_embedding(x[:, :, 6])
        edge_embedding = self.edge_embedding(x[:, :, 7])

        word_embedding_ent2 = self.w2v(x[:, :, 8])
        tag_embedding_ent2 = self.tag_embedding(x[:, :, 9])
        position_embedding_ent2 = self.normalize_position(x[:, :, 10:14].float())
        position_embedding_ent2 = self.relu(position_embedding_ent2)

        tokens_ent1 = torch.cat((word_embedding_ent1, tag_embedding_ent1, position_embedding_ent1), dim=2).float()
        tokens_ent2 = torch.cat((word_embedding_ent2, tag_embedding_ent2, position_embedding_ent2), dim=2).float()
        dep = torch.cat((direction_embedding, edge_embedding), dim=2).float()

        tokens_ent1 = self.relu(self.normalize_tokens(tokens_ent1))
        tokens_ent2 = self.relu(self.normalize_tokens(tokens_ent2))
        dep = self.relu(self.normalize_dep(dep))

        x = torch.cat((tokens_ent1, dep, tokens_ent2), dim=2)

        x = x.unsqueeze(1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x1 = torch.max(x1.squeeze(dim=3), dim=2)[0]
        x2 = torch.max(x2.squeeze(dim=3), dim=2)[0]
        x3 = torch.max(x3.squeeze(dim=3), dim=2)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dense_to_tag(x)
        x = self.softmax(x)

        return x

class BertModel(nn.Module):
    def __init__(self,
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
                 target_class: int = 5
                 ):

        super(BertModel, self).__init__()

        self.word_embedding_size = word_embedding_size
        self.tag_embedding = nn.Embedding(tag_number, tag_embedding_size, padding_idx=0)
        self.direction_embedding = nn.Embedding(direction_number, direction_embedding_size, padding_idx=0)
        self.edge_embedding = nn.Embedding(edge_number, edge_embedding_size, padding_idx=0)
        
        self.normalize_position = nn.Linear(in_features=position_number,
                                            out_features=position_embedding_size,
                                            bias=False)
        
        self.normalize_tokens = nn.Linear(in_features=word_embedding_size+tag_embedding_size+position_embedding_size,
                                          out_features=token_embedding_size,
                                          bias=False)
        
        self.normalize_dep = nn.Linear(in_features=direction_embedding_size+edge_embedding_size,
                                       out_features=dep_embedding_size,
                                       bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv1_out_channels,
                      kernel_size=(conv1_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv2_out_channels,
                      kernel_size=(conv2_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv3_out_channels,
                      kernel_size=(conv3_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.relu = nn.ReLU()
        self.dense_to_tag = nn.Linear(in_features=conv1_out_channels + conv2_out_channels + conv3_out_channels,
                                      out_features=target_class,
                                      bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        word_embedding_ent1 = x[:, :, 14:14+self.word_embedding_size]
        tag_embedding_ent1 = self.tag_embedding(x[:, :, 1].long())
        position_embedding_ent1 = self.normalize_position(x[:, :, 2:6].float())
        position_embedding_ent1 = position_embedding_ent1

        direction_embedding = self.direction_embedding(x[:, :, 6].long())
        edge_embedding = self.edge_embedding(x[:, :, 7].long())

        word_embedding_ent2 = x[:, :, 14+self.word_embedding_size:]
        tag_embedding_ent2 = self.tag_embedding(x[:, :, 9].long())
        position_embedding_ent2 = self.normalize_position(x[:, :, 10:14].float())
        position_embedding_ent2 = self.relu(position_embedding_ent2)

        tokens_ent1 = torch.cat((word_embedding_ent1, tag_embedding_ent1, position_embedding_ent1), dim=2).float()
        tokens_ent2 = torch.cat((word_embedding_ent2, tag_embedding_ent2, position_embedding_ent2), dim=2).float()
        dep = torch.cat((direction_embedding, edge_embedding), dim=2).float()

        tokens_ent1 = self.relu(self.normalize_tokens(tokens_ent1))
        tokens_ent2 = self.relu(self.normalize_tokens(tokens_ent2))
        dep = self.relu(self.normalize_dep(dep))

        x = torch.cat((tokens_ent1, dep, tokens_ent2), dim=2)

        x = x.unsqueeze(1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        # breakpoint()
        x1 = torch.max(x1.squeeze(dim=3), dim=2)[0]
        x2 = torch.max(x2.squeeze(dim=3), dim=2)[0]
        x3 = torch.max(x3.squeeze(dim=3), dim=2)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dense_to_tag(x)
        x = self.softmax(x)

        return x

class EmbeddedRecurrentModel(nn.Module):
    """
    Explain: This model can work with pre-embedded Dataset.
    Support type:
    - LSTM
    - Bi-LSTM
    - GRU (gate recurrent unit)
    Support activation function:
    - ReLU
    - LeakyReLU
    - PReLU
    - GELU
    """
    def __init__(self,
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
                target_class: int = 5
                ):
        pass

class BertWithPostionOnlyModel(nn.Module):
    """
    Only with bert + position encoding
    The stucture: [bert_embedding, pos_ent, zero_ent, pos_tag]
    """
    def __init__(self,
                dropout_rate: float = 0.5,
                word_embedding_size: int = 768,
                position_number: int = 512,
                position_embedding_size: int = 128,
                position_embedding_type: str = "normal",
                tag_number: int = 51,
                tag_embedding_size: int = 64,
                token_embedding_size : int = 256,
                conv1_out_channels: int = 256,
                conv2_out_channels: int = 256,
                conv3_out_channels: int = 256,
                conv1_length: int = 1,
                conv2_length: int = 2,
                conv3_length: int = 3,
                target_class: int = 5,
                dataset_name = "DDI"
                ):
        super(BertWithPostionOnlyModel, self).__init__()
        self.word_embedding_size = word_embedding_size
        self.position_embedding_size = position_embedding_size
        self.device ="cuda"
        self.tag_embedding = nn.Embedding(tag_number, tag_embedding_size, padding_idx=0)
        self.position_embedding_type = position_embedding_type
        if position_embedding_type == "normal":
            self.pos_embedding = nn.Linear(position_number, position_embedding_size, bias=False)
        elif position_embedding_type == "sinusoidal":
            self.pos_embedding = self.sinusoidal_positional_encoding
        elif position_embedding_type == "rotary":
            self.pos_embedding = self.rotary_positional_embedding
        else:
            raise ValueError("Wrong type pos embed")

        self.dropout = nn.Dropout(dropout_rate)

        self.normalize_tokens = nn.Linear(in_features = word_embedding_size+tag_embedding_size+position_embedding_size,
            out_features=token_embedding_size,
            bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv1_out_channels,
                      kernel_size=(conv1_length, token_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv2_out_channels,
                      kernel_size=(conv2_length, token_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv3_out_channels,
                      kernel_size=(conv3_length, token_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )
        self.dense_to_tag = nn.Linear(in_features = conv1_out_channels + conv2_out_channels + conv3_out_channels,out_features=target_class,
                        bias=False)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def sinusoidal_positional_encoding(self, position):
        d_model = int((self.position_embedding_size - 1) / 2)
        # position = position.unsqueeze(dim=2)
        angle_rads = torch.arange(d_model) // 2 * torch.pi / torch.pow(10000, 2 * (torch.arange(d_model) // 2) / d_model)
        angle_rads = angle_rads.to(self.device)
        angle_rads = angle_rads.unsqueeze(dim=0).unsqueeze(dim=0).expand((position.shape[0], 1, angle_rads.shape[0]))
        angle_rads = torch.bmm(position, angle_rads)
        pos_encoding = torch.zeros((angle_rads.shape[0], angle_rads.shape[1], angle_rads.shape[2])).to(self.device)
        pos_encoding[:, :, 0::2] = torch.sin(angle_rads[:, :, 0::2])
        pos_encoding[:, :, 1::2] = torch.cos(angle_rads[:, :, 1::2])
        return pos_encoding

    # def rotary_positional_embedding(self, position):
    #     d_model = int((self.position_embedding_size - 1) / 2)
    #     position = position.unsqueeze(dim=2)
    #     freqs = torch.exp(torch.linspace(0., -1., int(d_model // 2)) * torch.log(torch.tensor(10000.))).to(self.device)
    #     freqs = freqs.unsqueeze(dim=0).unsqueeze(dim=0).expand((position.shape[0], 1, freqs.shape[0]))
    #     angles = position * freqs
    #     rotary_matrix = torch.stack([torch.sin(angles), torch.cos(angles)], axis=-1).to(self.device)
    #     return rotary_matrix.reshape((position.shape[0], position.shape[1], d_model))

    def forward(self, x):
        x = x.float()

        if self.position_embedding_type == "normal": # Linear
            pos_embedding = self.pos_embedding(x[:,:,self.word_embedding_size: self.word_embedding_size+4])
        elif self.position_embedding_type == "sinusoidal":
            position_embedding_ent = x[:, :, self.word_embedding_size: self.word_embedding_size+4].float()
            pos3 = self.sinusoidal_positional_encoding(position_embedding_ent[:, :, 0])
            pos4 = self.sinusoidal_positional_encoding(position_embedding_ent[:, :, 1])
            pos_embedding = torch.cat((pos3, pos4, position_embedding_ent[:, :, 2:]), dim=2) 
        else: # rotary
            position_embedding_ent = x[:, :, self.word_embedding_size: self.word_embedding_size+4].float()
            pos3 = self.rotary_positional_embedding(position_embedding_ent[:, :, 0])
            pos4 = self.rotary_positional_embedding(position_embedding_ent[:, :, 1])
            pos_embedding = torch.cat((pos3, pos4, position_embedding_ent[:, :, 2:]), dim=2) 

        tag_embedding = self.tag_embedding(x[:,:,-1].long())
        x = self.normalize_tokens(torch.cat((x[:,:,:self.word_embedding_size], pos_embedding, tag_embedding), dim =2))
        
        x = x.unsqueeze(1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x1 = torch.max(x1.squeeze(dim=3), dim=2)[0]
        x2 = torch.max(x2.squeeze(dim=3), dim=2)[0]
        x3 = torch.max(x3.squeeze(dim=3), dim=2)[0]
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dense_to_tag(x)
        x = self.softmax(x)
        return x


class BertForSequenceClassification(nn.Module):
    # def __init__(self, self, config, gnn_config):
    def __init__(self,
                num_labels,
                dropout_prob,
                hidden_size,
                conv_window_size: list,
                max_seq_length,
                pos_emb_dim,
                middle_layer_size,
                model_name_or_path
                ):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(self.dropout_prob)
    
        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':GELU()}
        self.activation = activations[self.activation]

        if self.use_cnn:
            self.conv_list = nn.ModuleList([nn.Conv1d(hidden_size+2*self.pos_emb_dim, hidden_size, w, padding=(w-1)//2) for w in self.conv_window_size])
            self.pos_emb = nn.Embedding(2*self.max_seq_length, self.pos_emb_dim, padding_idx=0)

        # if self.use_desc and self.use_mol:
        #     self.desc_conv = nn.Conv1d(config.hidden_size, self.desc_conv_output_size, self.desc_conv_window_size, padding=(self.desc_conv_window_size-1)//2)
        #     self.classifier = nn.Linear(config.hidden_size+2*self.desc_conv_output_size+2*gnn_config.dim, config.num_labels)
        #     self.middle_classifier = nn.Linear(config.hidden_size+2*self.desc_conv_output_size+2*gnn_config.dim, self.middle_layer_size)
        # elif self.use_desc:
        #     self.desc_conv = nn.Conv1d(config.hidden_size, self.desc_conv_output_size, self.desc_conv_window_size, padding=(self.desc_conv_window_size-1)//2)
        #     if self.desc_layer_hidden != 0: self.W_desc = nn.Linear(2*self.desc_conv_output_size, 2*self.desc_conv_output_size)
        #     if self.middle_layer_size == 0:
        #         self.classifier = nn.Linear(config.hidden_size+2*self.desc_conv_output_size, config.num_labels)
        #     else:
        #         self.middle_classifier = nn.Linear(config.hidden_size+2*self.desc_conv_output_size, self.middle_layer_size)
        #         self.classifier = nn.Linear(self.middle_layer_size, config.num_labels)
        # elif self.use_mol:
        #     if self.middle_layer_size == 0:
        #         self.classifier = nn.Linear(config.hidden_size+2*gnn_config.dim, config.num_labels)
        #     else:
        #         self.middle_classifier = nn.Linear(config.hidden_size+2*gnn_config.dim, self.middle_layer_size)
        #         self.classifier = nn.Linear(self.middle_layer_size, config.num_labels)
        # else:
        if self.middle_layer_size == 0:
            self.classifier = nn.Linear(len(self.conv_window_size)*hidden_size, num_labels)
        else:
            self.middle_classifier = nn.Linear(len(self.conv_window_size)*config.hidden_size, self.middle_layer_size)
            self.classifier = nn.Linear(self.middle_layer_size, num_labels)
        self.init_weights()
        
        if self.use_cnn:
            self.pos_emb.weight.data.uniform_(-1e-3, 1e-3)

        self.bert = BertModel.from_pretrained(self.model_name_or_path)
        # if self.use_desc: self.desc_bert = BertModel.from_pretrained(self.model_name_or_path)
        # if self.use_mol: self.gnn = MolecularGraphNeuralNetwork(gnn_config.N_fingerprints, gnn_config.dim, gnn_config.layer_hidden, gnn_config.layer_output, gnn_config.mode, gnn_config.activation)

        self.use_cnn = self.use_cnn
        # self.use_desc = self.use_desc
        # self.desc_layer_hidden = self.desc_layer_hidden
        # self.gnn_layer_output = self.gnn_layer_output
        # self.use_mol = self.use_mol
        self.middle_layer_size = self.middle_layer_size

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                relative_dist1=None, relative_dist2=None,
                desc1_ii=None, desc1_am=None, desc1_tti=None,
                desc2_ii=None, desc2_am=None, desc2_tti=None,
                fingerprint=None,
                labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]
        #pooled_output = self.dropout(pooled_output)

        if self.use_cnn:
            relative_dist1 *= attention_mask
            relative_dist2 *= attention_mask
            pos_embs1 = self.pos_emb(relative_dist1)
            pos_embs2 = self.pos_emb(relative_dist2)
            conv_input = torch.cat((outputs[0], pos_embs1, pos_embs2), 2)
            conv_outputs = []
            for c in self.conv_list:
                conv_output = self.activation(c(conv_input.transpose(1,2)))
                conv_output, _ = torch.max(conv_output, -1)
                conv_outputs.append(conv_output)
            pooled_output = torch.cat(conv_outputs, 1)

        if self.use_desc:
            desc1_outputs = self.desc_bert(desc1_ii, attention_mask=desc1_am, token_type_ids=desc1_tti)
            desc2_outputs = self.desc_bert(desc2_ii, attention_mask=desc2_am, token_type_ids=desc2_tti)
            desc1_conv_input = desc1_outputs[0]
            desc2_conv_input = desc2_outputs[0]
            desc1_conv_output = self.activation(self.desc_conv(desc1_conv_input.transpose(1,2)))
            desc2_conv_output = self.activation(self.desc_conv(desc2_conv_input.transpose(1,2)))
            pooled_desc1_output, _ = torch.max(desc1_conv_output, -1)
            pooled_desc2_output, _ = torch.max(desc2_conv_output, -1)
            if self.desc_layer_hidden != 0:
                pooled_desc_output = self.activation(self.W_desc(torch.cat((pooled_desc1_output, pooled_desc2_output), 1)))
                pooled_output = torch.cat((pooled_output, pooled_desc_output), 1)
            else:
                pooled_output = torch.cat((pooled_output, pooled_desc1_output, pooled_desc2_output), 1)

        if self.use_mol:
            if fingerprint.ndim == 3: # In case of mini-batchsize = 1
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            else:
                fingerprint = np.expand_dims(fingerprint, 0)
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            gnn_output1 = self.gnn.gnn(fingerprint1)
            gnn_output2 = self.gnn.gnn(fingerprint2)
            gnn_output = torch.cat((gnn_output1, gnn_output2), 1)
            pooled_output = torch.cat((pooled_output, gnn_output), 1)
        
        pooled_output = self.dropout(pooled_output)
        if self.middle_layer_size == 0:
            logits = self.classifier(pooled_output)
        else:
            middle_output = self.activation(self.middle_classifier(pooled_output))
            logits = self.classifier(middle_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def zero_init_params(self):
        self.update_cnt = 0
        for x in self.parameters():
            x.data *= 0

    def accumulate_params(self, model):
        self.update_cnt += 1
        for x, y in zip(self.parameters(), model.parameters()):
            x.data += y.data

    def average_params(self):
        for x in self.parameters():
            x.data /= self.update_cnt

    def restore_params(self):
        for x in self.parameters():
            x.data *= self.update_cnt
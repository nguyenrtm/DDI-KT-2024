import torch
import torch.nn as nn

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
                target_class: int = 5
                ):
        super(BertWithPostionOnlyModel, self).__init__()
        self.word_embedding_size = word_embedding_size
        self.position_embedding_size = position_embedding_size
        self.device ="cuda"
        self.tag_embedding = nn.Embedding(tag_number, tag_embedding_size, padding_idx=0)

        if position_embedding_type == "normal":
            self.pos_embedding = nn.Linear(position_number, position_embedding_size, bias=False)
        elif position_embedding_type == "sinusoidal":
            self.pos_embedding = self.sinusoidal_positional_encoding
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

    def forward(self, x):
        x = x.float()
        pos_embedding = self.pos_embedding(x[:,:,self.word_embedding_size: self.word_embedding_size+4])
#         print(x[:,:,-1].long().shape)
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

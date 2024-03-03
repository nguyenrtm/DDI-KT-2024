import torch
import torch.nn as nn

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
        tag_embedding_ent1 = self.tag_embedding(x[:, :, 1])
        position_embedding_ent1 = self.normalize_position(x[:, :, 2:6].float())
        position_embedding_ent1 = position_embedding_ent1

        direction_embedding = self.direction_embedding(x[:, :, 6])
        edge_embedding = self.edge_embedding(x[:, :, 7])

        word_embedding_ent2 = x[:, :, 14+self.word_embedding_size:]
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
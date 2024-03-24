import torch
import torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self,
                 num_chars,
                 hidden_dim,
                 output_dim,
                 dropout_rate,
                 device):
        super(CharLSTM, self).__init__()
        self.device = device
        self.num_chars = num_chars
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.character_encoder = nn.Embedding(num_embeddings=num_chars, embedding_dim=hidden_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout_rate)
        
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.character_encoder(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)

        return x
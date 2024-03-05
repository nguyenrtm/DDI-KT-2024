import torch
import torch.nn as nn
from ddi_kt_2024.utils import load_pkl

class SMILESEmbedding(torch.nn.Module):
    def __init__(self,
                 embedding_size: int = 600,
                 embedding_dict_path: str = 'cache/chembert/chembert.ddi.dict.pkl',
                 device: str = 'cpu'):
        super(SMILESEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_dict = load_pkl(embedding_dict_path)
        self.device = device

    def forward(self, mol):
        if mol.mol == None:
            return torch.zeros([1, self.embedding_size]).to(self.device)

        smiles = mol.smiles

        if smiles in self.embedding_dict:
            return torch.tensor(self.embedding_dict[smiles]).unsqueeze(dim=0).to(self.device)
        else:
            return torch.zeros([1, self.embedding_size]).to(self.device)
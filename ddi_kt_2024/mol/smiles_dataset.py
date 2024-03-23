from torch.utils.data import Dataset
from tqdm import tqdm
from ddi_kt_2024.mol.features import smi_to_bert

from ddi_kt_2024.utils import load_pkl

class SmilesDataset(Dataset):
  def __init__(self, x, 
               embedding_dict_path: str = 'cache/chembert/chembert.ddi.dict.pkl', 
               embedding_size: int = 600,
               element: int = 1):
    mols = list()
    embedding_dict = load_pkl(embedding_dict_path)
    if element == 1:
      for m in tqdm(x, desc="Converting SMILES to BERT"):
        x1 = smi_to_bert(m[0], embedding_dict, embedding_size)
        mols.append(x1)
    elif element == 2:
      for m in tqdm(x, desc="Converting SMILES to BERT"):
        x2 = smi_to_bert(m[1], embedding_dict, embedding_size)
        mols.append(x2)
    self.x = [m for m in mols]

  def negative_instance_filtering(self, path):
      with open(path, 'r') as f:
          lines = f.read().split('\n')[:-1]
          lst = [int(x.strip()) for x in lines]

      new_x = list()

      for idx in lst:
          new_x.append(self.x[idx])

      self.x = new_x

  def __getitem__(self, idx):
    x = self.x[idx]
    return x

  def __len__(self):
    return len(self.x)
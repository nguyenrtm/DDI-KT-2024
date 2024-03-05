from torch.utils.data import Dataset
from tqdm import tqdm
from ddi_kt_2024.mol.features import smi_to_bert

class SmilesDataset(Dataset):
  def __init__(self, x, element=1):
    mols = list()
    if element == 1:
      for m in tqdm(x, desc="Converting SMILES to BERT"):
        x1 = smi_to_bert(m[0])
        mols.append(x1)
    elif element == 2:
      for m in tqdm(x, desc="Converting SMILES to BERT"):
        x2 = smi_to_bert(m[1])
        mols.append(x2)
    self.x = [m for m in mols]

  def __getitem__(self, idx):
    x = self.x[idx]
    return x

  def __len__(self):
    return len(self.x)
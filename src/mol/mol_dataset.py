from torch.utils.data import Dataset
from tqdm import tqdm
from src.mol.features import smi_to_pyg

class MolDataset(Dataset):
  def __init__(self, x, y):
    mols = list()
    for m in tqdm(x, desc="Converting SMILES to PyG"):
      x1 = smi_to_pyg(m[0])
      x2 = smi_to_pyg(m[1])
      mols.append([x1, x2])
    self.x = [m for m in mols]
    self.y = y

  def __getitem__(self, idx):
    x = self.x[idx]
    y = self.y[idx]
    return x, y

  def __len__(self):
    return len(self.x)
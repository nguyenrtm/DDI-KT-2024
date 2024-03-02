from torch.utils.data import Dataset
from tqdm import tqdm
from src.mol.features import smi_to_pyg
from torch_geometric.loader import DataLoader

class MolDataset(Dataset):
  def __init__(self, x, element=1):
    mols = list()
    if element == 1:
      for m in tqdm(x, desc="Converting SMILES to PyG"):
        x1 = smi_to_pyg(m[0])
        mols.append(x1)
    elif element == 2:
      for m in tqdm(x, desc="Converting SMILES to PyG"):
        x2 = smi_to_pyg(m[1])
        mols.append(x2)
    self.x = [m for m in mols]

  def __getitem__(self, idx):
    x = self.x[idx]
    return x

  def __len__(self):
    return len(self.x)
from torch.utils.data import Dataset
from tqdm import tqdm
from ddi_kt_2024.mol.features import smi_to_pyg
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

  def rm_no_smiles(self):
    new_list = list()
    add_index = list()
    for i in range(len(self.x)):
      if self.x[i][0].mol != 'None' and self.x[i][1].mol != 'None':
        new_list.append(self.x[i])
        add_index.append(i)
    
    self.x = new_list

    return add_index


  def __getitem__(self, idx):
    x = self.x[idx]
    return x

  def __len__(self):
    return len(self.x)
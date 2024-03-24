import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class FormulaDataset(Dataset):
  def __init__(self, 
               x,
               lookup_character,
               element: int = 1):
    self.lookup_character = lookup_character
    mols = list()
    if element == 1:
      for m in x:
        x1 = str(m[0])
        lst = list()
        for c in x1:
            if c in self.lookup_character.keys():
               lst.append(self.lookup_character[c])
            else:
                lst.append(0)
        lst = torch.tensor(lst)
        mols.append(lst)
    elif element == 2:
      for m in x:
        x2 = str(m[1])
        lst = list()
        for c in x2:
            if c in self.lookup_character.keys():
               lst.append(self.lookup_character[c])
            else:
                lst.append(0)
        lst = torch.tensor(lst)
        mols.append(lst)
        
    self.x = mols

  def batch_padding(self, batch_size):
    current = 0
    to_return = []
    while current + batch_size < len(self.x):
      batch = self.x[current:current+batch_size]
      max_len_in_batch = max([x.shape[0] for x in batch])

      for i in range(len(batch)):
        tmp = F.pad(batch[i], (0, max_len_in_batch - batch[i].shape[0]), "constant", 0)
        to_return.append(tmp)

      current += batch_size
    
    batch = self.x[current:]
    max_len_in_batch = max([x.shape[0] for x in batch])

    for i in range(len(batch)):
        tmp = F.pad(batch[i], (0, max_len_in_batch - batch[i].shape[0]), "constant", 0)
        to_return.append(tmp)

    self.x = to_return
     
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
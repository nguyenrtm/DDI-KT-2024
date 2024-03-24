import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

class FormulaDataset(Dataset):
  def __init__(self, 
               x,
               lookup_character,
               element: int = 1):
    mols = list()
    if element == 1:
      for m in x:
        x1 = m[0]
        lst = list()
        for c in x1:
           lst.append(self.lookup_character[c])
        lst = torch.tensor(lst).unsqueeze(dim=0)
    elif element == 2:
      for m in tqdm(x):
        x2 = m[1]
        lst = list()
        for c in x2:
           lst.append(self.lookup_character[c])
        lst = torch.tensor(lst).unsqueeze(dim=0)
        
    self.x = lst

  def batch_padding(self, batch_size):
    current = 0
    to_return = []
    while current + batch_size < len(self.data):
      batch = self.data[current:current+batch_size]
      max_len_in_batch = max([x.shape[1] for x in batch])

      for i in range(len(batch)):
        tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[1], 0, 0), "constant", 0)
        to_return.append(tmp)

      current += batch_size
     

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
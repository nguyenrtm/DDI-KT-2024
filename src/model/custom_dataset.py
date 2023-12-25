import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def fix_exception(self):
        i = 0
        while i < len(self.data):
            if self.data[i].shape[1] == 0:
                print(f"WARNING: Exception at data {i}")
                self.data[i] = torch.zeros((1, 1, 10))
            else:
                i += 1
                
    def squeeze(self):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].squeeze()

    def batch_padding(self, batch_size, min_batch_size=3):
        current = 0
        to_return = []

        while current + batch_size < len(self.data):
            batch = self.data[current:current+batch_size]
            max_len_in_batch = max(max([x.shape[1] for x in batch]), min_batch_size)

            for i in range(len(batch)):
                tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[1], 0, 0), "constant", 0)
                to_return.append(tmp)

            current += batch_size

        batch = self.data[current:]
        max_len_in_batch = max(max([x.shape[0] for x in batch]), min_batch_size)

        for i in range(len(batch)):
            tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[1], 0, 0), "constant", 0)
            to_return.append(tmp)

        self.data = to_return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label
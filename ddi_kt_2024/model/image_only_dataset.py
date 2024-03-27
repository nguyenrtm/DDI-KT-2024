import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Draw

from ddi_kt_2024.utils import get_labels, load_pkl

def mapped_property_reader(file_path):
    df = pd.read_csv(file_path)
    return df

def get_property_dict(df, property_name):
    query_dict = dict()
    for i in range(len(df)):
        query_dict[df.iloc[i]['name'].lower()] = df.iloc[i][property_name]
        if type(query_dict[df.iloc[i]['name'].lower()]) != str:
            query_dict[df.iloc[i]['name'].lower()] = 'None'
    return query_dict

def find_drug_property(drug_name, query_dict):
    return query_dict[drug_name.lower()]

class ImageOnlyDataset(Dataset):
    def __init__(self, data=None, labels=None):
        self.data = data
        self.labels = labels
        
    def fix_exception(self):
        i = 0
        while i < len(self.data):
            if self.data[i].shape[1] == 0:
                print(f"WARNING: Exception at data {i}")
                self.data[i] = torch.zeros((1, 1, 14), dtype=int)
            else:
                i += 1
                
    def squeeze(self):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile_1 = self.data[idx]['e1']['@text']
        smile_2 = self.data[idx]['e2']['@text']
        label = self.labels[idx]
        return self.all_images[smile_1.lower()], self.all_images[smile_2.lower()], label
    
    def get_df(self, df_path):
        self.df = mapped_property_reader(df_path)

    def get_filtered_idx(self, txt_path):
        with open(txt_path, "r") as f:
            lines = f.read().split('\n')[:-1]
            self.filtered_idx = [int(x.strip()) for x in lines]

    def prepare_images(self, df_path):
        self.all_images={}
        self.df = mapped_property_reader(df_path)
        dct = get_property_dict(self.df, 'smiles')
        transform = transforms.Compose([
                    transforms.Resize((300, 300)),
                    transforms.ToTensor()
                ])
        for smile in dct.keys():
            smile = str(smile)
            if dct[smile]=='None':
                self.all_images[smile.lower()]=torch.zeros((3, 300, 300), dtype=float)
            else:
                try:
                    this_smile = find_drug_property(smile, dct)
                    m = Chem.MolFromSmiles(this_smile)
                    img = Draw.MolToImage(m)
                    img_tensor = transform(img)
                except:
                    self.all_images[smile.lower()]=torch.zeros((3, 300, 300), dtype=float)
                    continue
                self.all_images[smile.lower()] = img_tensor

        
    def negative_instance_filtering(self, candidates):
        """do get_filtered_idx first"""
        self.all_candidates = candidates
        self.all_labels = get_labels(self.all_candidates)
        new_x = list()
        new_y = list()

        for idx in self.filtered_idx:
            new_x.append(self.all_candidates[idx])
            new_y.append(self.all_labels[idx])

        self.data = new_x
        self.labels = new_y

if __name__=="__main__":
    dataset = ImageOnlyDataset()
    # dataset.get_df("/workspaces/DDI-KT-2024/cache/mapped_drugs/DDI/full.csv")
    dataset.get_filtered_idx("/workspaces/DDI-KT-2024/cache/filtered_ddi/train_filtered_index.txt")
    train_candidates = load_pkl("/workspaces/DDI-KT-2024/cache/pkl/v2/notprocessed.candidates.train.pkl")
    dataset.prepare_images("/workspaces/DDI-KT-2024/cache/mapped_drugs/DDI/full.csv")
    dataset.negative_instance_filtering(train_candidates)
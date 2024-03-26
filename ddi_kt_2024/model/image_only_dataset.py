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
        sample_1 = self.data[idx][0]
        sample_2 = self.data[idx][1]
        label = self.labels[idx]
        return sample_1, sample_2, label
    
    def get_df(self, df_path):
        self.df = mapped_property_reader(df_path)

    def get_filtered_idx(self, txt_path):
        with open(txt_path, "r") as f:
            lines = f.read().split('\n')[:-1]
            self.filtered_idx = [int(x.strip()) for x in lines]

    def prepare(self, candidates):
        """
        Image only AND with filtered idx
        """
        self.data=[]
        self.full_labels=get_labels(candidates)
        self.labels=[]
        self.candidates=[] # For testing and viewing

        for c_idx, candidate in enumerate(candidates):
            if c_idx not in self.filtered_idx:
                continue
            # Get drug name
            drug_1 = candidate['e1']['@text']
            drug_2 = candidate['e2']['@text']

            # Check if can get smile
            dct = get_property_dict(self.df, 'smiles')
            if find_drug_property(drug_1, dct) is None or find_drug_property(drug_2, dct) is None or \
                find_drug_property(drug_1, dct) == 'None' or find_drug_property(drug_2, dct) == 'None':
                self.data.append(tuple(((torch.zeros((3,300,300), dtype=torch.float)),(torch.zeros((3,300,300), dtype=torch.float)))))
                self.labels.append(self.full_labels[c_idx])
                continue
            # Get smile
            smile_1 = find_drug_property(drug_1, dct)
            smile_2 = find_drug_property(drug_2, dct)
            try:
                m = Chem.MolFromSmiles(smile_1)
                img = Draw.MolToImage(m)
                transform = transforms.Compose([
                    transforms.Resize((300, 300)),
                    transforms.ToTensor()
                ])
                img_tensor_1 = transform(img)
            except ValueError:
                print(f"ValueError at idx {c_idx}")
                img_tensor_1 = torch.zeros((3,300,300), dtype=torch.float)

            try:
                m = Chem.MolFromSmiles(smile_2)
                img = Draw.MolToImage(m)
                transform = transforms.Compose([
                    transforms.Resize((300, 300)),
                    transforms.ToTensor()
                ])
                img_tensor_2 = transform(img)
            except:
                print(f"ValueError at idx {c_idx}")
                breakpoint()
                img_tensor_2 = torch.zeros((3,300,300), dtype=torch.float)

            # add
            self.data.append(tuple((img_tensor_1, img_tensor_2)))
            self.labels.append(self.full_labels[c_idx])
            
            if c_idx % 1000 == 0:
                print(f"Preparing {c_idx} images")

if __name__=="__main__":
    dataset = ImageOnlyDataset()
    dataset.get_df("/workspaces/DDI-KT-2024/cache/mapped_drugs/DDI/full.csv")
    dataset.get_filtered_idx("/workspaces/DDI-KT-2024/cache/filtered_ddi/train_filtered_index.txt")
    train_candidates = load_pkl("/workspaces/DDI-KT-2024/cache/pkl/v2/notprocessed.candidates.train.pkl")
    dataset.prepare(train_candidates)
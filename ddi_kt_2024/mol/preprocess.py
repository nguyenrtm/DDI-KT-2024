import pandas as pd
from tqdm import tqdm

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

def candidate_property(all_candidates, query_dict):
    x = list()
    y = list()
    for c in all_candidates:
        e1 = c['e1']['@text']
        e2 = c['e2']['@text']
        smiles1 = find_drug_property(e1, query_dict)
        smiles2 = find_drug_property(e2, query_dict)
        label = c['label']
        if label == 'false':
            label = 0
        elif label == 'advise':
            label = 1
        elif label == 'effect':
            label = 2
        elif label == 'mechanism':
            label = 3
        elif label == 'int':
            label = 4
        x.append([smiles1, smiles2])
        y.append(label)
    
    return x, y
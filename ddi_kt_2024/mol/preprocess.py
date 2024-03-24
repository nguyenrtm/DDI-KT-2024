import pandas as pd

def mapped_smiles_reader(file_path):
    df = pd.read_csv(file_path)
    return df

def find_drug_smiles(df, drug_name):
    drug_name = drug_name.lower()
    for i in range(len(df)):
        if df.iloc[i]['name'].lower() == drug_name:
            return df.iloc[i]['smiles']
    return 'None'

def find_drug_formula(df, drug_name):
    drug_name = drug_name.lower()
    for i in range(len(df)):
        if df.iloc[i]['name'].lower() == drug_name:
            return df.iloc[i]['formula']
    return 'None'

def candidate_smiles(all_candidates, mapped_smiles):
    x = list()
    y = list()
    for c in all_candidates:
        e1 = c['e1']['@text']
        e2 = c['e2']['@text']
        smiles1 = find_drug_smiles(mapped_smiles, e1)
        smiles2 = find_drug_smiles(mapped_smiles, e2)
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

def candidate_formula(all_candidates, mapped_smiles):
    x = list()
    y = list()
    for c in all_candidates:
        e1 = c['e1']['@text']
        e2 = c['e2']['@text']
        formula1 = find_drug_formula(mapped_smiles, e1)
        formula2 = find_drug_formula(mapped_smiles, e2)
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
        x.append([formula1, formula2])
        y.append(label)
    
    return x, y
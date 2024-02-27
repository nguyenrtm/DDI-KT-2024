def mapped_smiles_reader(file_path):
    f = open(file_path, 'r')
    all_drugs = f.read().split('\n')[:-1]
    all_drugs = [x.split('|') for x in all_drugs]
    return all_drugs

def find_drug_smiles(mapped_smiles, drug_name):
    drug_name = drug_name.lower()
    for smiles in mapped_smiles:
        if drug_name == smiles[1]:
            return smiles[2]
    return 'None'

def candidate_smiles(all_candidates, mapped_smiles):
    x = list()
    y = list()
    for c in all_candidates:
        e1 = c['e1']['@text']
        e2 = c['e2']['@text']
        smiles1 = find_drug_smiles(mapped_smiles, e1)
        smiles2 = find_drug_smiles(mapped_smiles, e2)
        if smiles1 != 'None' and smiles2 != 'None':
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
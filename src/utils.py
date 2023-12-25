import pickle as pkl
import torch

def id_find(lst, id):
    for element in lst:
        if element['@id'] == id:
            return element
        
def convert_all_list(ddi_dictionary):
    '''
    Convert every dict to list in ddi_dictionary.
    '''
    for doc in ddi_dictionary:
        doc = doc['document']
        if isinstance(doc['sentence'], dict):
            doc['sentence'] = [doc['sentence']]
        for sentence in doc['sentence']:
            if 'entity' in sentence.keys() and isinstance(sentence['entity'], dict):
                sentence['entity'] = [sentence['entity']]
            if 'pair' in sentence.keys() and isinstance(sentence['pair'], dict):
                sentence['pair'] = [sentence['pair']]
    
    return ddi_dictionary

def get_candidates(ddi_dictionary):
    '''
    Get relation pairs from ddi_dictionary.
    '''
    all_candidates = list()
    for document in ddi_dictionary:
        document = document['document']
        for s in document['sentence']:
            if isinstance(s, dict) and 'pair' in s.keys():
                if isinstance(s['pair'], dict):
                    candidate = dict()
                    _id = s['pair']['@id']
                    _e1 = s['pair']['@e1']
                    _e2 = s['pair']['@e2']
                    _label = s['pair']['@ddi']
                    candidate['id'] = _id
                    candidate['text'] = s['@text']
                    candidate['e1'] = id_find(s['entity'], _e1)
                    candidate['e2'] = id_find(s['entity'], _e2)
                    candidate['label'] = _label
                    all_candidates.append(candidate)
                else: 
                    for pair in s['pair']:
                        candidate = dict()
                        _id = pair['@id']
                        _e1 = pair['@e1']
                        _e2 = pair['@e2']
                        _label = pair['@ddi']
                        candidate['id'] = _id
                        candidate['text'] = s['@text']
                        candidate['e1'] = id_find(s['entity'], _e1)
                        candidate['e2'] = id_find(s['entity'], _e2)
                        candidate['label'] = _label
                        all_candidates.append(candidate)
    return all_candidates

def offset_to_idx(text, offset, nlp):
    '''
    Given offset of token in text, return its index in text.
    '''
    doc = nlp(text)
    offset = offset.split(';')[0]
    start = int(offset.split('-')[0])
    end = int(offset.split('-')[1])
    start_idx = -1
    end_idx = -1
    for i in range(len(doc) - 1):
        if doc[i].idx <= start and doc[i+1].idx > start:
            start_idx = doc[i].i
        if doc[i].idx < end and doc[i+1].idx > end:
            end_idx = doc[i].i
    if start_idx == -1:
        start_idx = len(doc) - 1
        end_idx = len(doc) - 1
    assert start_idx != -1, end_idx != -1
    return start_idx, end_idx

def get_labels(all_candidates):
    label_list = list()
    for candidate in all_candidates:
        assert candidate['label'] in ['true', 'false']
        if candidate['label'] == 'true':
            label_list.append(torch.tensor([1]))
        elif candidate['label'] == 'false':
            label_list.append(torch.tensor([0]))
    return label_list
        
def get_lookup(path):
    '''
    Get lookup table from file
    '''
    with open(path, 'r') as file:
        f = file.read().split('\n')
    return {f[i]: i + 1 for i in range(len(f))}

def lookup(element, dct):
    '''
    Get index of element in lookup table
    '''
    try: 
        idx = dct[element]
    except:
        idx = 0
    return idx

def load_pkl(path):
    with open(path, 'rb') as file:
        return pkl.load(file)
    
def dump_pkl(obj, path):
    with open(path, 'wb') as file:
        pkl.dump(obj, file)
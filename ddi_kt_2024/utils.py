import pickle as pkl
import torch
import numpy as np

def get_trimmed_w2v_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data['embeddings']
    
def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok
    return d


def id_find(lst, id):
    for element in lst:
        if element['@id'] == id:
            return element

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
        if candidate['label'] == 'false':
            label_list.append(torch.tensor([0]))
        elif candidate['label'] == 'advise':
            label_list.append(torch.tensor([1]))
        elif candidate['label'] == 'effect':
            label_list.append(torch.tensor([2]))
        elif candidate['label'] == 'mechanism':
            label_list.append(torch.tensor([3]))
        elif candidate['label'] == 'int':
            label_list.append(torch.tensor([4]))
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
def id_find(lst, id):
    '''
    Find an element in a list of dictionaries by its id.
    '''
    for element in lst:
        if element['@id'] == id:
            return element

def get_candidates(ddi_dictionary):
    '''
    Create pairs of relation candidates from a list of processed XML files.
    '''
    all_candidates = list()
    for document in ddi_dictionary:
        document = document['document']
        for s in document['sentence']:
            if 'pair' in s.keys():
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
    offset = offset.split('-')[0]
    for tok in doc:
        if str(tok.idx) == offset:
            return tok.i
        
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
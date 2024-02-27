import torch
from tqdm import tqdm

from src.utils import offset_to_idx

class PathProcesser:
    def __init__(self, spacy_nlp, lookup_word, lookup_dep, lookup_tag, lookup_direction):
        self.spacy_nlp = spacy_nlp
        self.lookup_word = lookup_word
        self.lookup_dep = lookup_dep
        self.lookup_tag = lookup_tag
        self.lookup_direction = lookup_direction
        
    def get_position_embedding_given_ent(self, 
                                         ent_start: int, 
                                         ent_end: int, 
                                         text_length: int):
        '''
        Given entity index, get position embedding of sentence
        '''
        lst = []
        count_bef = ent_start
        count_in = ent_end - ent_start
        count_aft = text_length - ent_end

        for i in range(count_bef, 0, -1):
            lst.append(-i)
        
        for i in range(count_in + 1):
            lst.append(0)

        for i in range(1, count_aft + 1):
            lst.append(i)
        return lst
    
    def build_position_embedding(self, text, offset):
        doc = self.spacy_nlp.nlp(text)
        text_length = len(doc)
        start_idx, end_idx = offset_to_idx(text, offset, self.spacy_nlp.nlp)
        pos_ent = self.get_position_embedding_given_ent(start_idx, end_idx, text_length)
        zero_ent = [0] * text_length
        for i in range(start_idx, end_idx + 1):
            zero_ent[i] = 1
        return [pos_ent, zero_ent]
    
    def create_mapping(self, candidate, sdp):
        text = candidate['text']
        doc = self.spacy_nlp.nlp(text)
        word_index = list()
        mapped_sdp = list()
        
        for tok in doc:
            pos = tok.i
            tag_key = tok.tag_
            word_key = tok.text
            try:
                word_index.append([self.lookup_word[word_key], 
                                   self.lookup_tag[tag_key]])
            except: 
                if word_key not in self.lookup_word.keys():
                    print(f"Token '{word_key}' is not in vocabulary!")
                    word_index.append([12367, self.lookup_tag[tag_key]])
        
        position_embedding_ent1 = self.build_position_embedding(text, candidate['e1']['@charOffset'])
        position_embedding_ent2 = self.build_position_embedding(text, candidate['e2']['@charOffset'])
        
        for edge in sdp:
            word1_idx = edge[0]
            word2_idx = edge[2]
            word1_keys = word_index[word1_idx] + [position_embedding_ent1[0][word1_idx], position_embedding_ent2[0][word1_idx], position_embedding_ent1[1][word1_idx], position_embedding_ent2[1][word1_idx]]
            word2_keys = word_index[word2_idx] + [position_embedding_ent1[0][word2_idx], position_embedding_ent2[0][word2_idx], position_embedding_ent1[1][word2_idx], position_embedding_ent2[1][word2_idx]]
            edge_keys = [self.lookup_direction[edge[1][0]], self.lookup_dep[edge[1][1]]]
            mapped_sdp.append(word1_keys + edge_keys + word2_keys)
        
        return torch.tensor(mapped_sdp)
    
    def create_mapping_all(self, all_candidates, all_sdp):
        all_mapped_sdp = list()
        for i in tqdm(range(len(all_candidates))):
            mapped_sdp = self.create_mapping(all_candidates[i], all_sdp[i]).unsqueeze(dim=0)
            all_mapped_sdp.append(mapped_sdp)
            
        return all_mapped_sdp
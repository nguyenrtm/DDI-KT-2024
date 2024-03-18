import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel
import numpy as np

from ddi_kt_2024.utils import (
    offset_to_idx, 
    get_lookup, 
    idx_to_offset,
    load_pkl
    )
from ddi_kt_2024.preprocess.spacy_nlp import SpacyNLP
from ddi_kt_2024.embed.get_embed_sentence_level import map_new_tokenize

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
        count_aft = text_length - ent_end - 1

        for i in range(count_bef, 0, -1):
            lst.append(-i)
        
        for i in range(count_in + 1):
            lst.append(0)

        for i in range(1, count_aft + 1):
            lst.append(i)
        return lst
    
    def build_position_embedding(self, text, offset, entity=None):
        doc = self.spacy_nlp.nlp(text)
        text_length = len(doc)
        start_idx, end_idx = offset_to_idx(text, offset, self.spacy_nlp.nlp, True, entity)

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

class TextPosProcessor(PathProcesser):
    """
    The stucture: [bert_embedding, pos_ent, zero_ent, pos_tag]
    """
    def __init__(self, lookup_word, lookup_tag, bert_model):
        """ 
        Lookup_word and lookup_tag from get_lookup()
        Bert_model is just name in huggingface
        """
        self.spacy_nlp = SpacyNLP()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.lookup_word = lookup_word
        self.lookup_tag = lookup_tag

    def return_bert_position(self, left_offset, right_offset, offset_mapping):
        pos_list = list()
        for i in range(len(offset_mapping[0][1:-1])):
            if left_offset < offset_mapping[0][1:-1][i][0]:
                j = i - 1
                while right_offset >= offset_mapping[0][1:-1][j][1]:
                    pos_list.append(j)
                    if j + 1 < len(offset_mapping[0][1:-1]):
                        j += 1
                    else: 
                        return(pos_list[0], pos_list[-1])
                if len(pos_list) ==0 :
                    print("Somehow pos list len = 0. Handle approximately....")
                    if left_offset > offset_mapping[0][1:-1][-1][0]:
                        return (len(offset_mapping[0][1:-1]) -1, len(offset_mapping[0][1:-1]) -1)
                    for i in range(len(offset_mapping[0][1:-1])):
                        if left_offset >= offset_mapping[0][1:-1][i][0]:
                            pos_list.append(i)
                            for j in range(i, len(offset_mapping[0][1:-1])):
                                if right_offset <= offset_mapping[0][1:-1][j][0]:
                                    return (i,j)
                            
                            return (i, len(offset_mapping[0][1:-1]) - 1)
                    print("Wow we have new exception")

                return (pos_list[0], pos_list[-1])
        
        return (len(offset_mapping[0][1:-1]) - 1, len(offset_mapping[0][1:-1]) - 1)

    def get_word_pos_embed_bert_size(self, candidate):
        '''
        ensure 100% but super slow
        Stack word pos together
        Procedure: Get tokenize Get sentence embed 
        '''
        text = candidate['text']
        doc = self.spacy_nlp.nlp(text)

        # Get pos embedding
        [pos_ent_1, zero_ent_1] = self.build_position_embedding(text, candidate['e1']['@charOffset'], candidate['e1']['@text'])
        [pos_ent_2, zero_ent_2] = self.build_position_embedding(text, candidate['e2']['@charOffset'], candidate['e1']['@text'])
        # Get sentence embed
        encoding = self.tokenizer.encode(doc.text, return_tensors="pt")
        sentence_tokenize = self.tokenizer.convert_ids_to_tokens(encoding[0])[1:-1]
        result = self.bert_model(encoding).last_hidden_state.detach()[:,1:-1,:] # Remove [CLS] and [SEP]
        word_index = []
        # word_status = map_new_tokenize([i.text for i in doc], sentence_tokenize)
        
        # Get word indexes
        offset = 0
        for iter, tok in enumerate(doc):
            pos = tok.i
            tag_key = tok.tag_
            word_key = tok.text

            # Get tokenize
            encoding = self.tokenizer.encode(word_key.lower(), return_tensors="pt")
            word_index.append(self.lookup_tag[tag_key])

                # Add more if 1 token spacy = n token bert
            for _ in range(int(encoding.shape[1])-3):
                pos_ent_1.insert(iter+offset, pos_ent_1[iter+offset])
                pos_ent_2.insert(iter+offset, pos_ent_2[iter+offset])
                zero_ent_1.insert(iter+offset, zero_ent_1[iter+offset])
                zero_ent_2.insert(iter+offset, zero_ent_2[iter+offset])
                word_index.append(self.lookup_tag[tag_key])

            offset += int(encoding.shape[1])-3
        
        # Concat
        pos_ent_1 = torch.from_numpy(np.array(pos_ent_1, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        pos_ent_2 = torch.from_numpy(np.array(pos_ent_2, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        zero_ent_1 = torch.from_numpy(np.array(zero_ent_1, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        zero_ent_2 = torch.from_numpy(np.array(zero_ent_2, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        word_index = torch.from_numpy(np.array(word_index, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        
        return torch.cat((result, pos_ent_1, pos_ent_2, zero_ent_1, zero_ent_2, word_index), dim=2)

    def get_word_pos_embed_spacy_size(self, candidate):
        '''
        ensure 100% but slow
        Stack word pos together
        Procedure: Get tokenize Get sentence embed 
        '''
        text = candidate['text']
        doc = self.spacy_nlp.nlp(text)

        # Get pos embedding
        [pos_ent_1, zero_ent_1] = self.build_position_embedding(text, candidate['e1']['@charOffset'], candidate['e1']['@text'])
        [pos_ent_2, zero_ent_2] = self.build_position_embedding(text, candidate['e2']['@charOffset'], candidate['e2']['@text'])
        # Get sentence embed
        encoding = self.tokenizer.encode(doc.text, return_tensors="pt")
        sentence_tokenize = self.tokenizer.convert_ids_to_tokens(encoding[0])[1:-1]
        offset_sentence_tokenize = self.tokenizer([text], return_offsets_mapping=True)

        temp_result = self.bert_model(encoding).last_hidden_state.detach()[:,1:-1,:] # Remove [CLS] and [SEP]
        word_index = []
        result = []
        # word_status = map_new_tokenize([i.text for i in doc], sentence_tokenize)
        # Get word indexes
        # offset = 0
        for iter, tok in enumerate(doc):
            pos = tok.i
            tag_key = tok.tag_
            word_key = tok.text

            # Get tokenize
            # encoding = self.tokenizer.encode(word_key.lower(), return_tensors="pt")
            word_index.append(self.lookup_tag[tag_key])

            # result.append(torch.mean(temp_result[:,iter+offset: iter+offset+int(encoding.shape[1])-2, :], dim=1, keepdim=True))
            word_offset = idx_to_offset(text, iter, self.spacy_nlp.nlp)
        
            word_bert_pos = self.return_bert_position(word_offset[0], word_offset[1], offset_sentence_tokenize['offset_mapping'])

            word_bert_embedding = torch.mean(temp_result[:, word_bert_pos[0]:word_bert_pos[1]+1, :], dim=1, keepdim=True)

            # offset += int(encoding.shape[1])-3
            result.append(word_bert_embedding)

        # Concat
        result = torch.cat(result, dim=1)

        # Remove NaN if any
        nan_mask = torch.isnan(result)
        result[nan_mask]=0.0

        pos_ent_1 = torch.from_numpy(np.array(pos_ent_1, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        pos_ent_2 = torch.from_numpy(np.array(pos_ent_2, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        zero_ent_1 = torch.from_numpy(np.array(zero_ent_1, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        zero_ent_2 = torch.from_numpy(np.array(zero_ent_2, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        word_index = torch.from_numpy(np.array(word_index, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        
        return torch.cat((result, pos_ent_1, pos_ent_2, zero_ent_1, zero_ent_2, word_index), dim=2)

    
    def legacy_get_word_pos_embed(self, candidate):
        '''
        Stack word pos together
        Procedure: Get tokenize Get sentence embed 
        '''
        text = candidate['text']
        doc = self.spacy_nlp.nlp(text)

        # Get tokenize
        encoding = self.tokenizer.encode(doc.text, return_tensors="pt")
        sentence_tokenize = self.tokenizer.convert_ids_to_tokens(encoding[0])[1:-1]
        result = self.bert_model(encoding).last_hidden_state.detach()[:,1:-1,:] # Remove [CLS] and [SEP]

        # Get pos embedding
        [pos_ent_1, zero_ent_1] = self.build_position_embedding(text, candidate['e1']['@charOffset'])
        [pos_ent_2, zero_ent_2] = self.build_position_embedding(text, candidate['e2']['@charOffset'])

        word_index = []
        word_status = map_new_tokenize([i.text for i in doc], sentence_tokenize)

        # Get word indexes
        for tok in doc:
            pos = tok.i
            tag_key = tok.tag_
            word_key = tok.text
            try:
                word_index.append(self.lookup_tag[tag_key])
            except: 
                if word_key not in self.lookup_word.keys():
                    print(f"Token '{word_key}' is not in vocabulary!")
                    word_index.append(self.lookup_tag[tag_key])
        # Let pos_ent, zero_ent and word_index fit with token size
        for status in word_status:
            if status['min_id'] == status['max_id']:
                continue

            values = [
                pos_ent_1[status['min_id']],
                zero_ent_1[status['min_id']],
                pos_ent_2[status['min_id']],
                zero_ent_2[status['min_id']],
                word_index[status['min_id']],
                ]
            for _ in range(status['max_id'] - status['min_id']):
                pos_ent_1.insert(status['min_id'], values[0])
                zero_ent_1.insert(status['min_id'], values[1])
                pos_ent_2.insert(status['min_id'], values[2])
                zero_ent_2.insert(status['min_id'], values[3])
                word_index.insert(status['min_id'], values[4])
            # offset += status['max_id'] - status['min_id']
       
        # Concat
        pos_ent_1 = torch.from_numpy(np.array(pos_ent_1, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        pos_ent_2 = torch.from_numpy(np.array(pos_ent_2, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        zero_ent_1 = torch.from_numpy(np.array(zero_ent_1, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        zero_ent_2 = torch.from_numpy(np.array(zero_ent_2, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)
        word_index = torch.from_numpy(np.array(word_index, dtype=np.float64)).unsqueeze_(dim=1).unsqueeze_(dim=0)

        return torch.cat((result, pos_ent_1, pos_ent_2, zero_ent_1, zero_ent_2, word_index), dim=2)

def test_get_object_to_manually_test(save_path="manually_test.pt"):
    """
    Candidates have to check:
    - All candidates train:
        - 16008-16074
        - 16328-16344
        - 17463-17478
        - 17933-17999
        - 24345-24347
    - All candidates test:
        - 1998-2000
        - 2230-2308
        - 4884-4894
        - 4971-5014
    """
    candidates = [
    {"type": "train", "min": 16008, "max": 16074},
    {"type": "train", "min": 16328, "max": 16344},
    {"type": "train", "min": 17463, "max": 17478},
    {"type": "train", "min": 17933, "max": 17999},
    {"type": "train", "min": 24345, "max": 24347},
    {"type": "test", "min": 1998, "max": 2000},
    {"type": "test", "min": 2230, "max": 2308},
    {"type": "test", "min": 4884, "max": 4894},
    {"type": "test", "min": 4971, "max": 5014}
    ]
    spacy_nlp = SpacyNLP()
    lookup_word = get_lookup("cache/fasttext/nguyennb/all_words.txt")
    lookup_tag = get_lookup("cache/fasttext/nguyennb/all_pos.txt")
    tpp = TextPosProcessor(lookup_word, lookup_tag, 'allenai/scibert_scivocab_uncased')
    all_candidates_train = load_pkl('cache/pkl/v2/notprocessed.candidates.train.pkl')
    all_candidates_test = load_pkl('cache/pkl/v2/notprocessed.candidates.test.pkl')
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    bert_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
    result_list = []
    for candidate in candidates:
        selected_id = random.randint(candidate["min"], candidate["max"])
        if candidate['type'] == "train":
            candidate_content = all_candidates_train[selected_id]
        else:
            candidate_content = all_candidates_test[selected_id]
        
        encoding = tokenizer.encode(spacy_nlp.nlp(candidate_content['text']).text, return_tensors="pt")
        sentence_tokenize = tokenizer.convert_ids_to_tokens(encoding[0])[1:-1]

        result_list.append({
            "type": candidate['type'],
            "content": candidate_content,
            "spacy_tokenize": [i.text for i in spacy_nlp.nlp(candidate_content['text'])],
            "bert_tokenize": sentence_tokenize,
            "bert_output": bert_model(encoding).last_hidden_state.detach()[:,1:-1,:],
            "result": tpp.get_word_pos_embed_spacy_size(candidate_content)
        })

    torch.save(result_list, save_path)

        

if __name__=="__main__":
    # Test
#     spacy_nlp = SpacyNLP()
#     lookup_word = get_lookup("cache/fasttext/nguyennb/all_words.txt")
#     lookup_tag = get_lookup("cache/fasttext/nguyennb/all_pos.txt")
#     tpp = TextPosProcessor(lookup_word, lookup_tag, 'allenai/scibert_scivocab_uncased')
#     candidate = {'label': 'false',
#  'id': 'DDI-DrugBank.d64.s79.p0',
#  'text': 'salicylates;sulfinpyrazone;',
#  'e1': {'@id': 'DDI-DrugBank.d64.s79.e0',
#   '@charOffset': '0-10',
#   '@type': 'group',
#   '@text': 'salicylates'},
#  'e2': {'@id': 'DDI-DrugBank.d64.s79.e1',
#   '@charOffset': '12-25',
#   '@type': 'drug',
#   '@text': 'sulfinpyrazone'}}
#     result = tpp.get_word_pos_embed_spacy_size(candidate)
#     print(f"Result shape: {result.shape}")

    test_get_object_to_manually_test()
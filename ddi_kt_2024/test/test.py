import random

import torch

from ddi_kt_2024.dependency_parsing.path_processer import *
from ddi_kt_2024.model.custom_dataset import *
from ddi_kt_2024.utils import *

def test_word_pos_bert_only(save_path="manually_test.pt"):
    """
    Candidates have to check:
    - All candidates train:
        - 1186-1192
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

def test_truncate_to_extend_from_entities(data_train_path, save_path="manually_test.pt"):
    """
    Load handled train custom dataset and get 10 samples.
    Train data only.
    """
    lookup_word = get_lookup("cache/fasttext/nguyennb/all_words.txt")
    lookup_tag = get_lookup("cache/fasttext/nguyennb/all_pos.txt")
    bert_model='allenai/scibert_scivocab_uncased' 
    data_train = torch.load(data_train_path)
    # Because it's based on handled data object, it should be fast
    data_train.truncate_to_extend_from_entities(lookup_word, lookup_tag, bert_model, 1)
    result_list = []
    for candidate in candidates:
        if candidate['type'] == "test":
            continue
        
        selected_id = random.randint(candidate["min"], candidate["max"])
        candidate_content = data_train.all_candidates[selected_id]
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        sentence_tokenize = tokenizer.convert_ids_to_tokens(encoding[0])[1:-1]
        result_list.append({
            "type": candidate['type'],
            "content": candidate_content,
            "spacy_tokenize": [i.text for i in spacy_nlp.nlp(candidate_content['text'])],
            "bert_tokenize": sentence_tokenize,
            "result": data_train.data[selected_id],
        })
        
    torch.save(result_list, save_path)


if __name__=="__main__":
    candidates = [
    {"type": "train", "min": 1186, "max": 1192},
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
    if option=="word_pos_bert_only":
        test_word_pos_bert_only()
    else:
        test_truncate_to_extend_from_entities("train.pt")
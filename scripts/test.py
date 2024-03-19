import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel
import numpy as np

from ddi_kt_2024.utils import (
    offset_to_idx, 
    get_lookup, 
    idx_to_offset,
    load_pkl,
    get_labels,
    get_processed_labels
    )
from ddi_kt_2024.preprocess.spacy_nlp import SpacyNLP
from ddi_kt_2024.embed.get_embed_sentence_level import map_new_tokenize
from ddi_kt_2024.model.custom_dataset import *

all_candidates_train = load_pkl('cache/pkl/bc5/candidates.train.pkl')
all_candidates_test = load_pkl('cache/pkl/bc5/candidates.test.pkl')
lookup_word = get_lookup("cache/fasttext/bc5/all_words.txt")
lookup_tag = get_lookup("cache/fasttext/bc5/all_pos.txt")

huggingface_model_name = 'allenai/scibert_scivocab_uncased'
y_train = get_processed_labels(all_candidates_train)
y_test = get_processed_labels(all_candidates_test)
data_train = BertPosEmbedOnlyDataset(all_candidates_train, y_train)
data_train.convert_to_tensors(lookup_word, lookup_tag, huggingface_model_name)
data_test = BertPosEmbedOnlyDataset(all_candidates_test, y_test)
data_test.convert_to_tensors(lookup_word, lookup_tag, huggingface_model_name)

torch.save(data_train, f"bc5_{huggingface_model_name.split('/')[-1]}_train.pt")
torch.save(data_test, f"bc5_{huggingface_model_name.split('/')[-1]}_test.pt")

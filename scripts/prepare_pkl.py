"""Script to prepare important pkl files"""
from pathlib import Path
import logging

import click
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
import torch

from ddi_kt_2024.utils import (
    load_pkl,
    get_labels
)
from ddi_kt_2024.reader.yaml_reader import get_yaml_config
from ddi_kt_2024.model.custom_dataset import CustomDataset, BertEmbeddingDataset, BertPosEmbedOnlyDataset
from ddi_kt_2024.model.trainer import Trainer
from ddi_kt_2024.model.word_embedding import WordEmbedding
from ddi_kt_2024 import logging_config
from ddi_kt_2024.utils import standardlize_config, get_lookup

# MAKE CHANGES HERE
prepare_type = "sdp_word_bert_embed_no_pad"
all_candidates_train = load_pkl('cache/pkl/v2/notprocessed.candidates.train.pkl')
all_candidates_test = load_pkl('cache/pkl/v2/notprocessed.candidates.test.pkl')
sdp_train_mapped = load_pkl('cache/pkl/v2/notprocessed.mapped.sdp.train.pkl')
sdp_test_mapped = load_pkl('cache/pkl/v2/notprocessed.mapped.sdp.test.pkl')
we = WordEmbedding(fasttext_path='cache/fasttext/nguyennb/fastText_ddi.npz',
                vocab_path='cache/fasttext/nguyennb/all_words.txt')
lookup_word = get_lookup("cache/fasttext/nguyennb/all_words.txt")
lookup_tag = get_lookup("cache/fasttext/nguyennb/all_pos.txt")

huggingface_model_name = 'allenai/scibert_scivocab_cased'
# END MAKE CHANGES

if prepare_type == "sdp_word_bert_embed_no_pad":
    # Data preparation
    y_train = get_labels(all_candidates_train)
    y_test = get_labels(all_candidates_test)
    # data_train = CustomDataset(sdp_train_mapped, y_train)
    # data_train.fix_exception()
    # data_train.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
    # data_train.squeeze()
    # data_test = CustomDataset(sdp_test_mapped, y_test)
    # data_test.fix_exception()
    # data_test.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
    # data_test.squeeze()
    data_train = BertEmbeddingDataset(all_candidates_train, sdp_train_mapped, y_train)
    data_train.fix_exception()
    # data_train.batch_padding(batch_size=128, min_batch_size=3)
    data_train.add_embed_to_data(
        huggingface_model_name=huggingface_model_name,
        all_words_path='cache/fasttext/nguyennb/all_words.txt',
        embed_size=768,
        mode="mean"
    )
    data_train.squeeze()
    data_train.fix_unsqueeze()
    data_test = BertEmbeddingDataset(all_candidates_test, sdp_test_mapped, y_test)
    data_test.fix_exception()
    # data_test.batch_padding(batch_size=128, min_batch_size=3)
    data_test.add_embed_to_data(
        huggingface_model_name=huggingface_model_name,
        all_words_path='cache/fasttext/nguyennb/all_words.txt',
        embed_size=768,
        mode="mean"
    )
    data_test.squeeze()
    data_test.fix_unsqueeze()


elif prepare_type == "word_pos_bert_embed_only_no_pad":
    y_train = get_labels(all_candidates_train)
    y_test = get_labels(all_candidates_test)
    data_train = BertPosEmbedOnlyDataset(all_candidates_train, y_train)
    data_train.convert_to_tensors(lookup_word, lookup_tag, huggingface_model_name)
    data_test = BertPosEmbedOnlyDataset(all_candidates_test, y_test)
    data_test.convert_to_tensors(lookup_word, lookup_tag, huggingface_model_name)



torch.save(data_train, './train.pt')
torch.save(data_test, './test.pt')
from quick_download import process_api
process("upload", f"ddi_kt_2024_multimodal_research/no_padding_custom_dataset/{huggingface_model_name.split('/')[-1]}/train.pt", "train.pt")
process("upload", f"ddi_kt_2024_multimodal_research/no_padding_custom_dataset/{huggingface_model_name.split('/')[-1]}/test.pt", "test.pt")
"""
Containing script to train
"""
from pathlib import Path
import logging

import click
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader

from ddi_kt_2024.utils import (
    load_pkl,
    get_labels
)
from ddi_kt_2024.reader.yaml_reader import get_yaml_config
from ddi_kt_2024.model.custom_dataset import CustomDataset
from ddi_kt_2024.model.trainer import Trainer
from ddi_kt_2024.model.word_embedding import WordEmbedding
from wandb_setup import wandb_setup
from ddi_kt_2024 import logging_config
from ddi_kt_2024.utils import standardlize_config

@click.command()
@click.option("--yaml_path", required=True, type=str, help="Path to the yaml config")
def run_train(yaml_path):
    # Initialize
    config = get_yaml_config(yaml_path)
    config, wandb_available = wandb_setup(config)
    config = standardlize_config(config)

    # Load pkl files
    all_candidates_train = load_pkl(config.all_candidates_train)
    all_candidates_test = load_pkl(config.all_candidates_test)
    sdp_train_mapped = load_pkl(config.sdp_train_mapped)
    sdp_test_mapped = load_pkl(config.sdp_test_mapped)
    we = WordEmbedding(fasttext_path=config.fasttext_path,
                   vocab_path=config.vocab_path)

    # Data preparation
    y_train = get_labels(all_candidates_train)
    y_test = get_labels(all_candidates_test)
    data_train = CustomDataset(sdp_train_mapped, y_train)
    data_train.fix_exception()
    data_train.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
    data_train.squeeze()
    data_test = CustomDataset(sdp_test_mapped, y_test)
    data_test.fix_exception()
    data_test.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
    data_test.squeeze()
    dataloader_train = DataLoader(data_train, batch_size=config.batch_size)
    dataloader_test = DataLoader(data_test, batch_size=config.batch_size)

    # Model initialization
    model = Trainer(we,
        dropout_rate=config.dropout_rate,
        word_embedding_size=config.word_embedding_size,
        tag_number=config.tag_number,
        tag_embedding_size=config.tag_embedding_size,
        position_number=config.position_number,
        position_embedding_size=config.position_embedding_size,
        direction_number=config.direction_number,
        direction_embedding_size=config.direction_embedding_size,
        edge_number=config.edge_number,
        edge_embedding_size=config.edge_embedding_size,
        token_embedding_size=config.token_embedding_size,
        dep_embedding_size=config.dep_embedding_size,
        conv1_out_channels=config.conv1_out_channels,
        conv2_out_channels=config.conv2_out_channels,
        conv3_out_channels=config.conv3_out_channels,
        conv1_length=config.conv1_length,
        conv2_length=config.conv2_length,
        conv3_length=config.conv3_length,
        w_false=config.w_false,
        w_advice=config.w_advice,
        w_effect=config.w_effect,
        w_mechanism=config.w_mechanism,
        w_int=config.w_int,
        target_class=5,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=config.device,
        wandb_available=wandb_available)

    # Model train
    model.train(dataloader_train, dataloader_test, num_epochs=config.epochs)

if __name__=="__main__":
    run_train()

"""
Containing script to train
"""
from pathlib import Path
import logging
import sys

sys.path.append(".")

import torch
import click
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from ddi_kt_2024.utils import (
    load_pkl,
    get_labels
)
from ddi_kt_2024.reader.yaml_reader import get_yaml_config
from ddi_kt_2024.model.custom_dataset import CustomDataset, BertEmbeddingDataset
from ddi_kt_2024.model.trainer import (
    Trainer, 
    BertTrainer, 
    BertWithPostionOnlyTrainer, 
    BC5_Trainer,
    Asada_Trainer,
    Image_Only_Trainer
)
from ddi_kt_2024.model.image_only_dataset import ImageOnlyDataset
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
    if not wandb_available:
        config = standardlize_config(config)

    # breakpoint()
    # Load pkl files
    if not config.type_embed=="asada_bert_unpad" \
        and not config.type_embed=="image_only":
        all_candidates_train = load_pkl(config.all_candidates_train)
        all_candidates_test = load_pkl(config.all_candidates_test)
        sdp_train_mapped = load_pkl(config.sdp_train_mapped)
        sdp_test_mapped = load_pkl(config.sdp_test_mapped)
        we = WordEmbedding(fasttext_path=config.fasttext_path,
                    vocab_path=config.vocab_path)

        # Data preparation
        y_train = get_labels(all_candidates_train)
        y_test = get_labels(all_candidates_test)
    if config.type_embed == 'fasttext':
        data_train = CustomDataset(sdp_train_mapped, y_train)
        data_train.fix_exception()
        data_train.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_train.squeeze()
        data_test = CustomDataset(sdp_test_mapped, y_test)
        data_test.fix_exception()
        data_test.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_test.squeeze()
    elif config.type_embed == 'bert_sentence':
        data_train = torch.load(config.train_custom_dataset)
        data_test = torch.load(config.test_custom_dataset)
        # breakpoint()
    elif config.type_embed == "bert_sentence_unpad":
        data_train = torch.load(config.train_custom_dataset)
        data_test = torch.load(config.test_custom_dataset)
        data_train.fix_unsqueeze()
        data_test.fix_unsqueeze()
        data_train.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_test.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_train.squeeze()
        data_test.squeeze()
    elif config.type_embed == "bert_pos_unpad" or config.type_embed == "bc5_bert_pos_unpad":
        data_train = torch.load(config.train_custom_dataset)
        data_test = torch.load(config.test_custom_dataset)
        data_train.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_test.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_train.squeeze()
        data_test.squeeze()
        # breakpoint()
    elif config.type_embed == "bert_pos_unpad_pkl":
        data_train = load_pkl(config.train_custom_dataset)
        data_test = load_pkl(config.test_custom_dataset)
        data_train.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_test.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_train.squeeze()
        data_test.squeeze()
    elif config.type_embed == "asada_bert_unpad":
        data_train = torch.load(config.train_custom_dataset)
        data_test = torch.load(config.test_custom_dataset)
        
    elif config.type_embed == "image_only":
        data_train = torch.load(config.train_custom_dataset)
        data_test = torch.load(config.test_custom_dataset)
    else:
        raise ValueError("Value of type_embed isn't supported yet!")
    
    # Convert to dataloader
    if config.type_embed == 'asada_bert_unpad':
        if hasattr(data_train, "random_sampler"):
            train_sampler = RandomSampler(data_train)
            dataloader_train = DataLoader(data_train, sampler=train_sampler, batch_size=config.batch_size)
        else:
            dataloader_train = DataLoader(data_train, batch_size=config.batch_size)
        dataloader_test = DataLoader(data_test, batch_size=config.batch_size)
    else:
        dataloader_train = DataLoader(data_train, batch_size=config.batch_size)
        dataloader_test = DataLoader(data_test, batch_size=config.batch_size)
        
    # Model initialization
    if config.type_embed == 'fasttext':
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
            target_class=config.target_class,
            lr=config.lr,
            weight_decay=config.weight_decay,
            device=config.device,
            wandb_available=wandb_available)
    elif config.type_embed == "bert_sentence" or config.type_embed == "bert_sentence_unpad":
        model = BertTrainer(
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
            target_class=config.target_class,
            lr=config.lr,
            weight_decay=config.weight_decay,
            device=config.device,
            wandb_available=wandb_available)
    elif config.type_embed == "bert_pos_unpad" or config.type_embed =="bert_pos_unpad_pkl":
        model = BertWithPostionOnlyTrainer(
            dropout_rate=config.dropout_rate,
            word_embedding_size=config.word_embedding_size,
            position_number=config.position_number,
            position_embedding_size=config.position_embedding_size,
            position_embedding_type=config.position_embedding_type,
            tag_number=config.tag_number,
            tag_embedding_size=config.tag_embedding_size,
            token_embedding_size=config.token_embedding_size,
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
            target_class=config.target_class,
            lr=config.lr,
            weight_decay=config.weight_decay,
            device=config.device,
            wandb_available=wandb_available
        )
    elif config.type_embed == "bc5_bert_pos_unpad":
        model = BC5_Trainer(
            all_candidates_test,
            dropout_rate=config.dropout_rate,
            word_embedding_size=config.word_embedding_size,
            position_number=config.position_number,
            position_embedding_size=config.position_embedding_size,
            position_embedding_type=config.position_embedding_type,
            tag_number=config.tag_number,
            tag_embedding_size=config.tag_embedding_size,
            token_embedding_size=config.token_embedding_size,
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
            target_class=config.target_class,
            lr=config.lr,
            weight_decay=config.weight_decay,
            device=config.device,
            wandb_available=wandb_available
        )
    elif config.type_embed == "asada_bert_unpad":
        model = Asada_Trainer(wandb_available=wandb_available,
                            freeze_bert=config.freeze_bert,
                            device=config.device
                            )
    elif config.type_embed == "image_only":
        model = Image_Only_Trainer(
            # dropout_rate=config.dropout_rate,
            # device=config.device,
            wandb_available=wandb_available
        )
        
    model.config = config
    
    # Experiment
    if config.parallel_training:
        model.enable_parallel()
        
    # Model train
    model.train(dataloader_train, dataloader_test, num_epochs=config.epochs)
    # model.evaluate(dataloader_test)
if __name__=="__main__":
    run_train()

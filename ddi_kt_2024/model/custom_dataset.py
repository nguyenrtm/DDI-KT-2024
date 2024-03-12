import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging

from ddi_kt_2024.model.huggingface_model import get_model
from ddi_kt_2024.embed.get_embed_sentence_level import map_new_tokenize, concat_to_tensor
from ddi_kt_2024.dependency_parsing.path_processer import TextPosProcessor
from ddi_kt_2024 import logging_config

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def fix_exception(self):
        i = 0
        while i < len(self.data):
            if self.data[i].shape[1] == 0:
                print(f"WARNING: Exception at data {i}")
                self.data[i] = torch.zeros((1, 1, 14), dtype=int)
            else:
                i += 1
                
    def squeeze(self):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].squeeze()

    def batch_padding(self, batch_size, min_batch_size=3):
        current = 0
        to_return = []

        while current + batch_size < len(self.data):
            batch = self.data[current:current+batch_size]
            max_len_in_batch = max(max([x.shape[1] for x in batch]), min_batch_size)

            for i in range(len(batch)):
                tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[1], 0, 0), "constant", 0)
                to_return.append(tmp)

            current += batch_size

        batch = self.data[current:]
        max_len_in_batch = max(max([x.shape[0] for x in batch]), min_batch_size)

        for i in range(len(batch)):
            tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[1], 0, 0), "constant", 0)
            to_return.append(tmp)

        self.data = to_return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

class BertEmbeddingDataset(CustomDataset):
    """ 
    In this class, text will be consider to overlap 
    the old fasttext embedding
    """
    def __init__(self, all_candidates, data, labels):
        self.all_candidates = all_candidates
        self.data = data
        self.labels = labels
        self.get_text()

    def get_text(self):
        self.text = [iter['text'] for iter in self.all_candidates]

    def add_embed_to_data(self, 
    huggingface_model_name = 'dmis-lab/biobert-base-cased-v1.2', 
    all_words_path ='cache/fasttext/nguyennb/all_words.txt', mode = 'mean',
    embed_size=768):
        """ 
        Add embed BERT to data:
        -> Loop through self.data -> append function
        """
        tokenizer, model = get_model(huggingface_model_name)
        with open(all_words_path, "r") as f:
            fasttext_word_list = [i.rstrip() for i in f.readlines()]
        fasttext_word_list = [""] + fasttext_word_list
        for i, sample in enumerate(self.data):
            second_dim_num = int(sample.shape[1])
            data_mapped_0_ids = sample[0,:,0]
            data_mapped_8_ids = sample[0,:,8]

            # Get back all words
            words_0_ids = [fasttext_word_list[int(iter)] for iter in data_mapped_0_ids]
            words_8_ids = [fasttext_word_list[int(iter)] for iter in data_mapped_8_ids]

            # Get tokenize
            encoding = tokenizer.encode(self.text[i], return_tensors="pt")
            sentence_tokenize = tokenizer.convert_ids_to_tokens(encoding[0])
            result = model(encoding).last_hidden_state.detach()

            # Map with new tokenize
            tokenize_map_0_ids = map_new_tokenize(words_0_ids, sentence_tokenize)
            tokenize_map_8_ids = map_new_tokenize(words_8_ids, sentence_tokenize)

            # Declare
            this_sent_embedded_first = torch.Tensor([])
            this_sent_embedded_mean = torch.Tensor([])
            this_sent_embedded_last = torch.Tensor([])

            for tokenize_status in tokenize_map_0_ids:
                this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last = concat_to_tensor(tokenize_status,
                result, this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last, embed_size)

            for tokenize_status in tokenize_map_8_ids:
                this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last = concat_to_tensor(tokenize_status,
                result, this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last, embed_size)

            if mode == 'first':
                self.data[i] = torch.cat(
                    (self.data[i], this_sent_embedded_first.reshape(1,second_dim_num,-1)),
                    dim=2
                )
            elif mode == 'mean':
                self.data[i] = torch.cat(
                    (self.data[i], this_sent_embedded_mean.reshape(1,second_dim_num,-1)),
                    dim=2
                )
            elif mode == 'last':
                self.data[i] = torch.cat(
                    (self.data[i], this_sent_embedded_last.reshape(1,second_dim_num,-1)),
                    dim=2
                )

            if (i+1) % 100 == 0:
                logging.info(f"Handled {i+1} / {len(self.data)}")

    def fix_unsqueeze(self):
        for data_i in self.data:
            if len(list(data_i.shape)) == 1:
                data_i.unsqueeze_(dim=0)
                data_i.unsqueeze_(dim=0) # Not an error
            elif len(list(data_i.shape)) == 2:
                data_i.unsqueeze_(dim=0)

class BertPosEmbedOnlyDataset(BertEmbeddingDataset):
    """Bert + pos customdataset only"""
    def __init__(self, candidates, labels):
        super().__init__(candidates, [], labels)

    def convert_to_tensors(self, lookup_word, lookup_tag, bert_model):
        """ 
        Lookup_word and lookup_tag from get_lookup()
        Bert_model is just name in huggingface
        """
        tpp = TextPosProcessor(lookup_word, lookup_tag, bert_model)
        for iter, candidate in enumerate(self.all_candidates):
            try:
                result = tpp.get_word_pos_embed(candidate)
            except Exception as e:
                breakpoint()
                print(f"Exception when handle at index {iter}")
                del self.labels[iter]
                continue
            self.data.append(result)
            if (iter + 1 )% 100 == 0:
                print(f"Handled {iter+1}/{len(self.all_candidates)}")
        print("Convert to tensor completed!")

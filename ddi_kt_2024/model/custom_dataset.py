import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging

from ddi_kt_2024.model.huggingface_model import get_model
from ddi_kt_2024.embed.get_embed_sentence_level import map_new_tokenize, concat_to_tensor, sdp_map_new_tokenize
from ddi_kt_2024.dependency_parsing.path_processer import TextPosProcessor
from ddi_kt_2024 import logging_config
from ddi_kt_2024.utils import load_pkl, get_labels
from ddi_kt_2024.model.word_embedding import WordEmbedding
from ddi_kt_2024.preprocess.spacy_nlp import SpacyNLP

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
        self.spacy_nlp = SpacyNLP()
        with open(all_words_path, "r") as f:
            fasttext_word_list = [i.rstrip() for i in f.readlines()]
        fasttext_word_list = [""] + fasttext_word_list
        for i, sample in enumerate(self.data[291:326]):
            i+=291
            if torch.all(sample == torch.zeros((1,1,14))):
                print(f"Old handled exception. Skipping...")
                new_shape = list(sample.shape)
                new_shape[-1] = 768*2
                new_tensor = torch.zeros(new_shape)
                self.data[i] = torch.cat((sample, new_tensor),dim =-1)
                continue
            second_dim_num = int(sample.shape[1])
            # data_mapped_0_ids = sample[0,:,0]
            # data_mapped_8_ids = sample[0,:,8]
            text = self.text[i]
            doc = self.spacy_nlp.nlp(text)
            # Get back all words
            # words_0_ids = [fasttext_word_list[int(iter)] for iter in data_mapped_0_ids]
            # words_8_ids = [fasttext_word_list[int(iter)] for iter in data_mapped_8_ids]

            # Get tokenize
            encoding = tokenizer.encode(doc.text, return_tensors="pt")
            sentence_tokenize = tokenizer.convert_ids_to_tokens(encoding[0])
            result = model(encoding).last_hidden_state.detach()

            # Map with new tokenize
            try:
                tokenize_map_0_ids, tokenize_map_8_ids = sdp_map_new_tokenize(doc, encoding, tokenizer, sample[0], fasttext_word_list)
            except Exception as e:
                print(f"Receiving exception at {i}. Process will continue...")
                breakpoint()
                new_shape = list(sample.shape)
                new_shape[-1] = 768*2
                new_tensor = torch.zeros(new_shape)
                self.data[i] = torch.cat((sample, new_tensor),dim =-1)
                continue
            # Declare
            bert_embed_first_1 = torch.Tensor([])
            bert_embed_mean_1 = torch.Tensor([])
            bert_embed_last_1 = torch.Tensor([])
            bert_embed_first_2 = torch.Tensor([])
            bert_embed_mean_2 = torch.Tensor([])
            bert_embed_last_2 = torch.Tensor([])

            for tokenize_status in tokenize_map_0_ids:
                bert_embed_first_1, bert_embed_mean_1, bert_embed_last_1 = concat_to_tensor(tokenize_status,
                result, bert_embed_first_1, bert_embed_mean_1, bert_embed_last_1, embed_size)

            for tokenize_status in tokenize_map_8_ids:
                bert_embed_first_2, bert_embed_mean_2, bert_embed_last_2 = concat_to_tensor(tokenize_status,
                result, bert_embed_first_2, bert_embed_mean_2, bert_embed_last_2, embed_size)
            if mode == 'first':
                self.data[i] = torch.cat(
                    (self.data[i], bert_embed_first_1.reshape(1,second_dim_num,-1), bert_embed_first_2.reshape(1,second_dim_num,-1)),
                    dim=2
                )
            elif mode == 'mean':
                self.data[i] = torch.cat(
                    (self.data[i], bert_embed_mean_1.reshape(1,second_dim_num,-1), bert_embed_mean_2.reshape(1,second_dim_num,-1)),
                    dim=2
                )
            elif mode == 'last':
                self.data[i] = torch.cat(
                    (self.data[i], bert_embed_last_1.reshape(1,second_dim_num,-1), bert_embed_last_2.reshape(1,second_dim_num,-1)),
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

    def convert_to_tensors(self, lookup_word, lookup_tag, bert_model, type="spacy"):
        """ 
        Lookup_word and lookup_tag from get_lookup()
        Bert_model is just name in huggingface
        """
        tpp = TextPosProcessor(lookup_word, lookup_tag, bert_model)
        self.data = []
        self.temp_labels = []
        self.temp_all_candidates = []
        for iter, candidate in enumerate(self.all_candidates):
            try:
                if type=="bert":
                    result = tpp.get_word_pos_embed_bert_size(candidate)
                else:
                    result = tpp.get_word_pos_embed_spacy_size(candidate)
            except Exception as e:
                print(f"Exception when handle at index {iter}")
                continue
            self.data.append(result)
            self.temp_all_candidates.append(self.all_candidates[iter])
            self.temp_labels.append(self.labels[iter])
            if (iter + 1 )% 100 == 0:
                print(f"Handled {iter+1}/{len(self.all_candidates)}")
        self.labels = self.temp_labels
        self.all_candidates = self.temp_all_candidates # For easy debug

        # 
        print("Convert to tensor completed!")

if __name__=="__main__":
    # all_candidates_train = load_pkl('cache/pkl/v2/notprocessed.candidates.train.pkl')
    # sdp_train_mapped = load_pkl('cache/pkl/v2/notprocessed.mapped.sdp.train.pkl')
    # we = WordEmbedding(fasttext_path='cache/fasttext/nguyennb/fastText_ddi.npz',
    #                 vocab_path='cache/fasttext/nguyennb/all_words.txt')

    # huggingface_model_name = 'allenai/scibert_scivocab_uncased'
    # y_train = get_labels(all_candidates_train)
    # data_train = BertEmbeddingDataset(all_candidates_train, sdp_train_mapped, y_train)
    # data_train.fix_exception()
    # data_train.add_embed_to_data(
    #     huggingface_model_name=huggingface_model_name,
    #     all_words_path='cache/fasttext/nguyennb/all_words.txt',
    #     embed_size=768,
    #     mode="mean"
    # )
    all_candidates_test = load_pkl('cache/pkl/v2/notprocessed.candidates.test.pkl')
    sdp_test_mapped = load_pkl('cache/pkl/v2/notprocessed.mapped.sdp.test.pkl')
    we = WordEmbedding(fasttext_path='cache/fasttext/nguyennb/fastText_ddi.npz',
                    vocab_path='cache/fasttext/nguyennb/all_words.txt')

    huggingface_model_name = 'allenai/scibert_scivocab_uncased'
    y_test = get_labels(all_candidates_test)
    data_test = BertEmbeddingDataset(all_candidates_test, sdp_test_mapped, y_test)
    data_test.fix_exception()
    data_test.add_embed_to_data(
        huggingface_model_name=huggingface_model_name,
        all_words_path='cache/fasttext/nguyennb/all_words.txt',
        embed_size=768,
        mode="mean"
    )
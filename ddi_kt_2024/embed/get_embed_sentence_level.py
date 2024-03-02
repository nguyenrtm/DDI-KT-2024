"""
Currently not working due to different size of data
"""

import logging
from pathlib import Path

from transformers import AutoTokenizer, BertModel
import torch

from ddi_kt_2024 import logging_config
# from ddi_kt_2024.model.custom_dataset import CustomDataset, BertEmbeddingDataset
from ddi_kt_2024.utils import dump_pkl, load_pkl, get_labels

def concat_to_tensor(tokenize_status, result, this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last):
    this_sent_embedded_first = torch.cat(
        (this_sent_embedded_first, 
        torch.reshape(
            result[0,tokenize_status['min_id'], :], (1, 1, 768))), dim=1
    )
    if tokenize_status['max_id'] - tokenize_status['min_id'] ==0:
        this_sent_embedded_mean = torch.cat(
            (this_sent_embedded_mean, 
            torch.reshape(
                result[0,tokenize_status['min_id'], :], (1, 1, 768))), dim=1
        )
    else:
        this_sent_embedded_mean = torch.cat(
            (this_sent_embedded_mean, 
            torch.reshape(
                torch.mean(
                    result[0,tokenize_status['min_id']: tokenize_status['max_id']+1, :], axis =0
                , keepdim=True)
                , (1, 1, 768)
                )), dim=1
        )
    this_sent_embedded_last = torch.cat(
        ( this_sent_embedded_last, 
        torch.reshape(
            result[0,tokenize_status['max_id'], :], (1, 1, 768))), dim=1
    )
    return this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last

def map_new_tokenize(words, sentence_tokenize):
    new_tokenize_ids = []
    for word in words:
        word = word.lower()
        status = {
            'min_id': 999,
            'max_id': 0
        }
        for token_id, token in enumerate(sentence_tokenize):
            if token[:2] == "##":
                continue
            if token in word and word[0:len(token)] == token:
                # Get longest word
                min_id = token_id
                max_id = token_id+1
                while True:
                    if sentence_tokenize[max_id][:2] != "##":
                        break
                    else:
                        max_id += 1
                word_determined = "".join("".join(sentence_tokenize[min_id:max_id]).split("##"))            
                if word_determined == word:
                    status['min_id'] = min_id
                    status['max_id'] = max_id-1
                    # breakpoint()
                    break

        if status['min_id'] > status['max_id']:
            # If some how we missed, we take the first id
            status['min_id'] = 0
            status['max_id'] = 0

        new_tokenize_ids.append(status)
    return new_tokenize_ids

def process(model_name, all_candidates, dictionary_path, data_mapped, folder_output):
    """ 
    Get tokenize base on data mapped.
    Data_mapped must after batch padding.

    Output:
    It will output as (sentence length, 32, 768)
    32 because we taking account of both word at index 0 and index 8
    """
    HIDDEN_DIM = 768
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    with open(dictionary_path, "r") as f:
        fasttext_word_list = [i.rstrip() for i in f.readlines()]
        fasttext_word_list = [""] + fasttext_word_list
    # Declaring
    all_sent_embedded_first = torch.Tensor([])
    all_sent_embedded_mean = torch.Tensor([])
    all_sent_embedded_last = torch.Tensor([])
    for i in range(len(all_candidates)):
        # Get id of important words from data_mapped
        data_mapped_ids, _ = data_mapped[i]
        data_mapped_0_ids = data_mapped_ids[:,0]
        data_mapped_8_ids = data_mapped_ids[:,8]
        # Get back all words
        words_0_ids = [fasttext_word_list[int(iter)] for iter in data_mapped_0_ids]
        words_8_ids = [fasttext_word_list[int(iter)] for iter in data_mapped_8_ids]

        # Get tokenize
        encoding = tokenizer.encode(all_candidates[i]['text'], return_tensors="pt")
        try:
            sentence_tokenize = tokenizer.convert_ids_to_tokens(encoding[0])
        except:
            logging.error("Can't tokenized. Return an empty tensor...")
            all_sent_embedded_first = torch.cat((all_sent_embedded_first, torch.zeros(1,32,HIDDEN_DIM)))
            all_sent_embedded_mean = torch.cat((all_sent_embedded_mean, torch.zeros(1,32,HIDDEN_DIM)))
            all_sent_embedded_last = torch.cat((all_sent_embedded_last, torch.zeros(1,32,HIDDEN_DIM)))
            continue
        # Get result
        result = model(encoding).last_hidden_state.detach()

        # Map with new tokenize
        tokenize_map_0_ids = map_new_tokenize(words_0_ids, sentence_tokenize)
        tokenize_map_8_ids = map_new_tokenize(words_8_ids, sentence_tokenize)

        this_sent_embedded_first = torch.Tensor([])
        this_sent_embedded_mean = torch.Tensor([])
        this_sent_embedded_last = torch.Tensor([])

        # Append to tensor based on the result
        for tokenize_status in tokenize_map_0_ids:
            this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last = concat_to_tensor(tokenize_status, result,this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last)

        for tokenize_status in tokenize_map_8_ids:
            this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last = concat_to_tensor(tokenize_status, result, this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last)

        # Append to all_sent
        all_sent_embedded_first = torch.cat((all_sent_embedded_first, this_sent_embedded_first), dim=0)
        all_sent_embedded_mean = torch.cat((all_sent_embedded_mean, this_sent_embedded_mean), dim=0)
        all_sent_embedded_last = torch.cat((all_sent_embedded_last, this_sent_embedded_last), dim=0)
        # Log
        if (i+1) % 100 == 0:
            logging.info(f"Handled {i+1}/{len(all_candidates)} sentences. Shape {all_sent_embedded_first.shape}")
    # Save torch file
    dump_pkl(all_sent_embedded_first, str(Path(folder_output) / "all_sentence_embed_first.pkl"))
    dump_pkl(all_sent_embedded_mean, str(Path(folder_output) / "all_sentence_embed_mean.pkl"))
    dump_pkl(all_sent_embedded_last, str(Path(folder_output) / "all_sentence_embed_last.pkl"))
    logging.info("Everything has been done!")


if __name__=="__main__":
    # sdp_test_mapped = load_pkl('cache/pkl/v2/notprocessed.mapped.sdp.test.pkl')
    # y_test = get_labels(load_pkl('cache/pkl/v2/notprocessed.candidates.test.pkl'))
    # data_test = CustomDataset(sdp_test_mapped, y_test)
    # data_test.fix_exception()
    # data_test.batch_padding(batch_size=128, min_batch_size=3)
    # data_test.squeeze()
    # process('dmis-lab/biobert-base-cased-v1.2', 
    #     load_pkl('cache/pkl/v2/notprocessed.candidates.test.pkl'), 
    #     'cache/fasttext/nguyennb/all_words.txt', 
    #     data_test, 
    #     '.')
    pass
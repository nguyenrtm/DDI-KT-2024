"""
Currently not working due to different size of data
"""

import logging
from pathlib import Path

from transformers import AutoTokenizer, BertModel
import torch

from ddi_kt_2024 import logging_config
from ddi_kt_2024.model.custom_dataset import CustomDataset, BertEmbeddingDataset
from ddi_kt_2024.utils import dump_pkl, load_pkl, get_labels

def concat_to_tensor(tokenize_status, result, this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last, embed_size=768):
    this_sent_embedded_first = torch.cat(
        (this_sent_embedded_first, 
        torch.reshape(
            result[0,tokenize_status['min_id'], :], (1, 1, embed_size))), dim=1
    )
    if tokenize_status['max_id'] - tokenize_status['min_id'] ==0:
        this_sent_embedded_mean = torch.cat(
            (this_sent_embedded_mean, 
            torch.reshape(
                result[0,tokenize_status['min_id'], :], (1, 1, embed_size))), dim=1
        )
    else:
        this_sent_embedded_mean = torch.cat(
            (this_sent_embedded_mean, 
            torch.reshape(
                torch.mean(
                    result[0,tokenize_status['min_id']: tokenize_status['max_id']+1, :], axis =0
                , keepdim=True)
                , (1, 1, embed_size)
                )), dim=1
        )
    this_sent_embedded_last = torch.cat(
        ( this_sent_embedded_last, 
        torch.reshape(
            result[0,tokenize_status['max_id'], :], (1, 1, embed_size))), dim=1
    )
    return this_sent_embedded_first, this_sent_embedded_mean, this_sent_embedded_last

def legacy_map_new_tokenize(words, sentence_tokenize):
    """broken function"""
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

def map_new_tokenize(words, encoding, tokenizer):
    """
    words: list of word
    encoding: sentence embed
    NOT SUPPORT MULTIPLE SENTENCE
    """
    def _check(word, encoding,tokenizer,start_idx_to_check):
        result = None
        flag = False
        if word.strip() == "":
            result = {
                "min_id": 0,
                "max_id": 0
            }
            return result, True, start_idx_to_check
        word_encoding = tokenizer.encode(word, return_tensors="pt")[:,1:-1]
        for index in range(start_idx_to_check, int(encoding.shape[1]) - int(word_encoding.shape[1]) +1):
            if torch.all(word_encoding == encoding[:, index: index+int(word_encoding.shape[1])]):
                result = {
                    "min_id": index,
                    "max_id": index + int(word_encoding.shape[1]) - 1
                }
                flag = True
                start_idx_to_check = index + int(word_encoding.shape[1]) - 1
                break
        return result, flag, start_idx_to_check
    result_list = []
    start_idx_to_check = 0
    duplicate = 0
    sentence_tokenize = tokenizer.convert_ids_to_tokens(encoding[0])
    for word_idx, word in enumerate(words):
        # print(f"idx: {word_idx} {start_idx_to_check}")
        if duplicate >= 1:
            result_list.append(result_list[-1])
            duplicate -=1

            # Check if next token some how separated
            if duplicate == 0:
                if word_idx +1 < len(words) and "##" in sentence_tokenize[result_list[-1]['max_id'] +1]:
                    encoding[0,result_list[-1]['max_id'] +1] = tokenizer.encode(sentence_tokenize[result_list[-1]['max_id'] +1][2:], return_tensors="pt")[0,1:-1]
                    
            continue
        
        word_result, flag, start_idx_to_check = _check(word, encoding, tokenizer, start_idx_to_check)
        if flag:
            result_list.append(word_result)
            start_idx_to_check = start_idx_to_check
            if word_idx +1 < len(words) and "##" in sentence_tokenize[result_list[-1]['max_id'] +1]:
                    encoding[0,result_list[-1]['max_id'] +1] = tokenizer.encode(sentence_tokenize[result_list[-1]['max_id'] +1][2:], return_tensors="pt")[0,1:-1]
        else:
            while True:
                # Happen when spacy smaller than bert (rare but still exist)
                duplicate +=1
                word = "".join([w for w in words[word_idx: word_idx+duplicate]])
                word_result, flag, start_idx_to_check = _check(word, encoding, tokenizer, start_idx_to_check)
                if flag:
                    result_list.append(word_result)
                    start_idx_to_check = start_idx_to_check
                    break
                if word_idx + duplicate > len(words):
                    raise ValueError("WTF")
            
    return result_list

def sdp_map_new_tokenize(text, encoding, tokenizer, data_original, fasttext_word_list):
    """
    We need modify some to make sure that right word right embed
    """
    # Build the map between spacy <-> bert
    spacy_bert_map = map_new_tokenize([i.text for i in text], encoding, tokenizer)

    # Locate 2 entities

    entity_1 = fasttext_word_list[data_original[0,0]]
    entity_2 = fasttext_word_list[data_original[-1,8]]
    entity_1_pos = -1
    entity_2_pos = -1

    # Add some contraint to avoid bullshit position
    max_dis_1 = max(int(torch.max(data_original[:,2])), int(torch.max(data_original[:,-4])))
    min_dis_1 = min(int(torch.min(data_original[:,2])), int(torch.min(data_original[:,-4])))
    max_dis_2 = max(int(torch.max(data_original[:,3])), int(torch.max(data_original[:,-3])))
    min_dis_2 = min(int(torch.min(data_original[:,3])), int(torch.min(data_original[:,-3])))
    # breakpoint()
    doc_len = len([i.text for i in text])
    for idx, element in enumerate(text):
        if element.text.lower() == entity_1.lower() and idx + max_dis_1 < doc_len and idx + min_dis_1 >=0:
            entity_1_pos = idx
            break

    for idx, element in enumerate(text):
        if element.text.lower() == entity_2.lower() and idx + max_dis_2 < doc_len and idx + min_dis_2 >=0:
            entity_2_pos = idx
            break
    
    # Build the 2 new maps
    tokenize_map_0_ids = []
    tokenize_map_8_ids = []
    # print(f"Entity: {entity_1_pos} {entity_2_pos}")

    for i in range(int(data_original.shape[0])):
        # print(f"{entity_1_pos+data_original[i,2]} {entity_2_pos+data_original[i, 3]} {entity_2_pos+data_original[i,-3]} {entity_1_pos+data_original[i,-4]}")
        tokenize_map_0_ids.append(spacy_bert_map[entity_1_pos+data_original[i,2]])
        tokenize_map_8_ids.append(spacy_bert_map[entity_2_pos+data_original[i,-3]])

    return tokenize_map_0_ids, tokenize_map_8_ids

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
        tokenize_map_0_ids = map_new_tokenize(words_0_ids, encoding, tokenizer)
        tokenize_map_8_ids = map_new_tokenize(words_8_ids, encoding, tokenizer)
        
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
    sdp_test_mapped = load_pkl('cache/pkl/v2/notprocessed.mapped.sdp.test.pkl')
    y_test = get_labels(load_pkl('cache/pkl/v2/notprocessed.candidates.test.pkl'))
    data_test = CustomDataset(sdp_test_mapped, y_test)
    data_test.fix_exception()
    # data_test.batch_padding(batch_size=128, min_batch_size=3)
    data_test.squeeze()
    process('dmis-lab/biobert-base-cased-v1.2', 
        load_pkl('cache/pkl/v2/notprocessed.candidates.test.pkl'), 
        'cache/fasttext/nguyennb/all_words.txt', 
        data_test, 
        '.')
    pass
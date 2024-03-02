import logging

from transformers import AutoTokenizer, BertModel
import torch

from ddi_kt_2024 import logging_config
from ddi_kt_2024.model.custom_dataset import CustomDataset, BertEmbeddingDataset
from ddi_kt_2024.utils import dump_pkl, load_pkl, get_labels

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

def process(model_name, all_candidates, dictionary_path, data_mapped, id_word, folder_output):
    """ 
    Get tokenize base on data mapped.
    Data_mapped must after batch padding.
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
        data_mapped_ids = data_mapped_ids[:,id_word]
        # Get back all words
        words = [fasttext_word_list[int(iter)] for iter in data_mapped_ids]
        # Get tokenize
        encoding = tokenizer.encode(all_candidates[i]['text'], return_tensors="pt")
        try:
            sentence_tokenize = tokenizer.convert_ids_to_tokens(encoding[0])
        except:
            logging.error("Can't tokenized. Return an empty tensor...")
            all_sent_embedded_first = torch.cat((all_sent_embedded_first, torch.zeros(16,HIDDEN_DIM)))
            all_sent_embedded_mean = torch.cat((all_sent_embedded_mean, torch.zeros(16,HIDDEN_DIM)))
            all_sent_embedded_last = torch.cat((all_sent_embedded_last, torch.zeros(16,HIDDEN_DIM)))
            continue
        # Get result
        result = model(encoding).last_hidden_state.detach()

        # Map with new tokenize
        tokenize_map = map_new_tokenize(words, sentence_tokenize)

        # Append to tensor based on the result
        for tokenize_status in tokenize_map:
            # breakpoint()
            all_sent_embedded_first = torch.cat(
                (all_sent_embedded_first, 
                torch.reshape(
                    result[0,tokenize_status['min_id'], :], (1, 768)))
            )
            if tokenize_status['max_id'] - tokenize_status['min_id'] ==0:
                all_sent_embedded_mean = torch.cat(
                    (all_sent_embedded_mean, 
                    torch.reshape(
                        result[0,tokenize_status['min_id'], :], (1, 768)))
                )
            else:
                all_sent_embedded_mean = torch.cat(
                    (all_sent_embedded_mean, 
                    torch.reshape(
                        torch.mean(
                            result[0,tokenize_status['min_id']: tokenize_status['max_id']+1, :], axis =0
                        , keepdim=True)
                        , (1, 768)
                        ))
                )
            all_sent_embedded_last = torch.cat(
               ( all_sent_embedded_last, 
                torch.reshape(
                    result[0,tokenize_status['max_id'], :], (1, 768)))
            )
        if (i+1) % 100 == 0:
            logging.info(f"Handled {i+1}/{len(all_candidates)} sentences")
    # Save torch file
    dump_pkl(all_sent_embedded_first, str(Path(folder_output) / "all_sentence_embed_first.pkl"))
    dump_pkl(all_sent_embedded_mean, str(Path(folder_output) / "all_sentence_embed_mean.pkl"))
    dump_pkl(all_sent_embedded_last, str(Path(folder_output) / "all_sentence_embed_last.pkl"))
    logging.info("Everything has been done!")


if __name__=="__main__":
    sdp_train_mapped = load_pkl('cache/pkl/v2/notprocessed.mapped.sdp.train.pkl')
    y_train = get_labels(load_pkl('cache/pkl/v2/notprocessed.candidates.train.pkl'))
    data_train = CustomDataset(sdp_train_mapped, y_train)
    data_train.fix_exception()
    data_train.batch_padding(batch_size=128, min_batch_size=3)
    data_train.squeeze()
    process('dmis-lab/biobert-base-cased-v1.2', 
        load_pkl('cache/pkl/v2/notprocessed.candidates.train.pkl'), 
        'cache/fasttext/nguyennb/all_words.txt', 
        data_train, 
        0, 
        '.')
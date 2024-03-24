"""
Similar method of asada's paper
Details: https://github.com/tticoin/DESC_MOL-DDIE
"""
import torch
from transformers import AutoTokenizer, BertModel
from transformers.data.processors.utils import InputExample, InputFeatures, DataProcessor
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import glue_output_modes as output_modes

def convert_to_examples(candidates, can_type="train", save_path=None):
    """
    Return:
    An InputExample with the following structure:
    {
        "guid": id of candidate,
        "label": label of its candidate,
        "text_a": text,
        "text_b": always null 
    }
     examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    """
    # Change with DRUGOTHER and DRUG1, DRUG2
    current_sentence = ""
    examples = []
    for idx, candidate in enumerate(candidates):
        if (idx + 1) % 100 ==0:
            print(f"H: {idx+1}/{len(candidates)}")
        if candidate['text'] != current_sentence:
            current_sentence = candidate['text']
            all_sentence_entities = set()
            inc = 1
            while True:
                if len(candidates) <= idx + inc:
                    break
                if candidates[idx+inc]['text'] != current_sentence:
                    break
                all_sentence_entities.add(candidates[idx+inc]['e1']['@text'])
                all_sentence_entities.add(candidates[idx+inc]['e2']['@text'])
                inc +=1
            
            all_sentence_entities=list(all_sentence_entities)
        
        offset_1 = candidate['e1']['@charOffset']
        offset_1 = offset_1.split(';')[0]
        offset_1 = (int(offset_1.split('-')[0]), int(offset_1.split('-')[1]))
        
        offset_2 = candidate['e2']['@charOffset']
        offset_2 = offset_2.split(';')[0]
        offset_2 = (int(offset_2.split('-')[0]), int(offset_2.split('-')[1]))

        current_sentence = current_sentence[:offset_1[0]] + "DRUG1" + current_sentence[offset_1[1]:]
        current_sentence = current_sentence[:offset_2[0]] + "DRUG2" + current_sentence[offset_2[1]:]
        
        for entity in all_sentence_entities:
            current_sentence = current_sentence.replace(entity, "DRUGOTHER")

        examples.append(
            InputExample(guid=f"{can_type}_{idx+1}", text_a=current_sentence, text_b=None, label=candidate['label'])
        )

    if save_path is not None:
        torch.save(examples, save_path)
    return examples

def preprocess(examples, model_name, max_seq_length=128, save_path=None):
    if isinstance(examples, str):
        examples = torch.load(examples)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=['negative', 'mechanism', 'effect', 'advise', 'int'],
        max_length=256,
        output_mode=output_modes['mrpc'],
        pad_on_left=0,                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    # Get Position index
    drug_id = tokenizer.vocab['drug']
    one_id = tokenizer.vocab['##1']
    two_id = tokenizer.vocab['##2']

    all_input_ids = [f.input_ids for f in features]
    all_entity1_pos= []
    all_entity2_pos= []
    for input_ids in all_input_ids:
        entity1_pos = max_seq_length-1 
        entity2_pos = max_seq_length-1 
        for i in range(max_seq_length):
            if input_ids[i] == drug_id and input_ids[i+1] == one_id:
                entity1_pos = i
            if input_ids[i] == drug_id and input_ids[i+1] == two_id:
                entity2_pos = i
        all_entity1_pos.append(entity1_pos)
        all_entity2_pos.append(entity2_pos)

    range_list = list(range(max_seq_length, 2*max_seq_length))
    all_relative_dist1 = torch.tensor([[x - e1 for x in range_list] for e1 in all_entity1_pos], dtype=torch.long)
    all_relative_dist2 = torch.tensor([[x - e2 for x in range_list] for e2 in all_entity2_pos], dtype=torch.long)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids,
        all_relative_dist1, all_relative_dist2,
        all_labels)

    if save_path is not None:
        torch.save(dataset, save_path)
    
    return dataset
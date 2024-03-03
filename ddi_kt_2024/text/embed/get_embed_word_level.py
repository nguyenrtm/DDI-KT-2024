from transformers import AutoTokenizer, BertModel
import numpy as np
import torch

def process(model_name, axis):
    """ 
    model_name: name of model
    axis: should be (1,768) for bert, other I don't know
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    first_list = np.empty(axis)
    mean_list = np.empty(axis)
    last_list = np.empty(axis)

    for idx, word in enumerate(content):
        encoding = tokenizer.encode(word, return_tensors="pt")
        result = model(encoding).last_hidden_state.detach()
        if list(encoding.shape)[1] !=0:
            repre_first = result[:,0,:]
            repre_mean = torch.mean(result, dim=1)
            repre_last = result[:,-1,:]
        else:
            repre_first = repre_mean = repre_last = result[:,0,:]
        
        first_list = np.append(first_list, repre_first.numpy(), axis =0)
        mean_list = np.append(mean_list, repre_mean.numpy(), axis =0)    
        last_list = np.append(last_list, repre_last.numpy(), axis =0)    
        if (idx+1) % 200 == 0:
            print(f"{idx+1}/ {len(content)}")
        
    np.savez(f"{model_name.split('/')[-1]}_first_token.npz", embeddings=first_list)
    np.savez(f"{model_name.split('/')[-1]}_mean_token.npz", embeddings=mean_list)
    np.savez(f"{model_name.split('/')[-1]}_last_token.npz", embeddings=last_list)

if __name__=="__main__":
    pass
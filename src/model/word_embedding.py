import torch

from utils import get_trimmed_w2v_vectors, load_vocab

class WordEmbedding:
    def __init__(self, path='./DDI-KT-2024/data/fasttext/fastText_ddi.npz'):
        self.vectors = get_trimmed_w2v_vectors(path)
        self.vocab = load_vocab('./DDI-KT-2024/data/fasttext/all_words.txt')

    def get_word_vector(self, word):
        if word not in self.vocab.keys():
            return torch.tensor(self.vectors[-1]) # $UNK$ vector
        else:
            return torch.tensor(self.vectors[self.vocab[word]])
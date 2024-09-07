""" Data Preprocessing """

from datasets import load_dataset
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import collections
from collections import Counter
from bs4 import BeautifulSoup
from word_mapping import word_mapping
import re
import torch
from torch.utils.data import Dataset, DataLoader


""" Text Cleaner """

stop_words = set(stopwords.words('english'))

def text_cleaner(text):
    
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([word_mapping[t] if t in word_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i) 
    text = " ".join(long_words).strip()
    def no_space(word, prev_word):
        return word in set(',!"";.''?') and prev_word!=" "
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    text = ''.join(out)

    return text


""" Tokenize function """

def tokenize(lines, token='word'):

    assert token in ('word', 'char'), 'Unknown token type: ' + token
    
    return [line.split() if token == 'word' else list(line) for line in lines]


""" Padding function """

def truncate_pad(line, num_steps, padding_token):

    if len(line) > num_steps:
        return line[:num_steps]    # truncate
    
    return line + [padding_token] * (num_steps - len(line))    # padding


""" The Vocabulary Class """

class Vocab:

    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):

        # flatten a 2D list
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        # count the token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # the list of unique tokens
        self.idx_to_token = ['<unk>'] + reserved_tokens + [token for token, freq in self.token_freqs if freq >= min_freq]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
    
    def __len__(self):
        return len(self.idx_to_token)


    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]


    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]


    def unk(self):  # Index for the unknown token
        return self.token_to_idx.get('<unk>', -1)
    

# fn to add eos and padding and also determine valid length of each data sample

def build_array_sum(lines, vocab, num_steps):

    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    
    return array, valid_len


# create the tensor dataset object 

def load_array(data_arrays, batch_size, is_train=True):

    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

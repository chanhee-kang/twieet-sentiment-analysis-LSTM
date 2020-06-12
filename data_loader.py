# -*- coding: utf-8 -*-
import pickle
import string
import torch
import random
import torchtext
import torch
import codecs
import random
import torch.utils.data as Data
import pandas as pd
import numpy as np
# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix, cuda=False):
    var = torch.LongTensor([to_ix[w] for w in seq.split(' ')])
    return var

def prepare_label(label,label_to_ix, cuda=False):
    var = torch.LongTensor([label_to_ix[label]])
    return var

def build_token_to_ix(sentences):
    token_to_ix = dict()
    for sent in sentences:
        if sent != sent:
            continue
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def clean_str(data):
    for example in data.examples:
        text = [x.lower() for x in vars(example)['text']]  # 소문자
        text = [x.replace("<br", "") for x in text]  # <br 제거
        text = [''.join(c for c in s if c not in string.punctuation) for s in text]  # 문장부호
        text = [s for s in text if s and not s == " " and not s == "  "]  # 공란제거
        vars(example)['text'] = text
    return data


def load_data():
    label_to_ix = {'negative': 0, 'neutral': 1, 'positive': 2}

    # already tokenized and there is no standard split
    # the size follow the Mou et al. 2016 instead

    test_data = []
    test = pd.read_csv(r"C:\Users\Administrator\Desktop\듀오비스\sentimental\test.csv")
    # for idx in range(len(test)):
    #     data.append(torchtext.data.Example.fromlist([test[idx], labels[idx]], datafields))
    # data = torchtext.data.Dataset(data, datafields)
    TEXT = torchtext.data.Field(tokenize='spacy')
    LABEL = torchtext.data.LabelField()

    datafields = [('text', TEXT), ('label', LABEL)]
    temp = []
    for data in test.values:
        if data[1] != data[1]:
            continue

        test_data.append(torchtext.data.Example.fromlist([data[1], label_to_ix[data[2]]],datafields))

    train_data = []
    train = pd.read_csv(r"C:\Users\Administrator\Desktop\듀오비스\sentimental\train.csv")
    for data in train.values:
        if data[2] != data[2]:
            continue
        train_data.append(torchtext.data.Example.fromlist([data[2], label_to_ix[data[3]]], datafields))
    train_data = torchtext.data.Dataset(train_data, datafields)
    test_data = torchtext.data.Dataset(test_data, datafields)

    train_data = clean_str(train_data)
    test_data = clean_str(test_data)
    train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

    TEXT.build_vocab(train_data, test_data,valid_data, max_size=50000)
    LABEL.build_vocab(train_data,test_data,valid_data)
    # word_to_ix = build_token_to_ix([s for s, _ in test_data + train_data])

    pickle.dump(TEXT, open("text.pkl", "wb"))
    pickle.dump(LABEL, open("label.pkl", "wb"))

    print('vocab size:',len(TEXT.vocab),'label size:',len(label_to_ix))
    print('loading data done!')
    return TEXT,LABEL,train_data,valid_data,test_data,label_to_ix



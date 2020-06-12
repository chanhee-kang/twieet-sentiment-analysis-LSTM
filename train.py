# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,r'C:\Users\Administrator\Desktop\듀오비스\sentimental')

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import os
import random
import torchtext
from tqdm import tqdm
import time
from model import Sentiment
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    # evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            batch.text = batch.text.to(device)
            batch.label = batch.label.to(device)
            predictions = model(batch.text)

            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, iterator, optimizer, criterion,device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()  # train_mode
    for batch in iterator:
        # initializing
        optimizer.zero_grad()
        # forward pass
        batch.text = batch.text.to(device)
        batch.label = batch.label.to(device)
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)


        acc = categorical_accuracy(predictions, batch.label)

        # backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
batch_size = 128

TEXT, LABEL, train_data, valid_data, test_data, label_to_ix = data_loader.load_data()
EMBEDDING_DIM = 400
HIDDEN_DIM = 400
EPOCH = 20
best_dev_acc = 0.0
OUTPUT_DIM = len(label_to_ix)
train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    device=device, sort=False)

model = Sentiment(len(TEXT.vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 2, 0.5)
model.to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def write_embeddings(path, embeddings, vocab):
    with open(path, 'w') as f:
        for i, embedding in enumerate(tqdm(embeddings)):
            word = vocab.itos[i]
            # skip words with unicode symbols
            if len(word) != len(word.encode()):
                continue
            vector = ' '.join([str(i) for i in embedding.tolist()])
            f.write(f'{word} {vector}\n')

best_valid_loss = float('inf')

for epoch in range(EPOCH):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, loss_function, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, loss_function, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pt')

model.load_state_dict(torch.load('best_model.pt'))

test_loss, test_acc= evaluate(model, test_iterator, loss_function, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

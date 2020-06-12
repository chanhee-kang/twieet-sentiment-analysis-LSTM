import torch
import torch.autograd as autograd
import torch.nn as nn


class Sentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()


        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * n_layers, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        emb = self.drop(self.embed(x))

        out, (h, c) = self.lstm(emb)
        h = self.drop(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))

        return self.fc(h.squeeze(0))
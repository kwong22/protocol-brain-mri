'''Defines the neural network, loss function, and metrics'''

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 dropout,
                 output_size,
                 embeds,
                 pad_idx=1
                 ):

        super(LSTMClassifier, self).__init__()

        self.dummy_param = nn.Parameter(torch.empty(0)) # to get this model's device

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pad_idx = pad_idx

        if embeds:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx).from_pretrained(embeds)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.hidden2out = nn.Linear(hidden_dim, output_size)

        self.dropout_layer = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        device = self.dummy_param.device
        return(autograd.Variable(torch.randn(self.num_layers * 2, batch_size, self.hidden_dim)).to(device),
               autograd.Variable(torch.randn(self.num_layers * 2, batch_size, self.hidden_dim)).to(device))

    def forward(self, text):
        self.hidden = self.init_hidden(text.shape[0]) # initialized randomly with each forward pass
        embeds = self.embedding(text.long())

        lens = torch.tensor([y.item() for y in map(lambda x: sum(x != self.pad_idx), text)],
                            dtype=torch.int64)

        embeds = pack_padded_sequence(embeds, lens.cpu(), batch_first=True)
        lstm_out, (hn, cn) = self.lstm(embeds, self.hidden)

        output = self.dropout_layer(hn[-1])
        output = self.hidden2out(output)

        return output


def loss_fn(outputs, labels, weight=None):
    return nn.CrossEntropyLoss(weight=weight)(outputs, labels)


def accuracy(outputs, labels):
    return (outputs.argmax(1) == labels).sum().item() / labels.shape[0]


metrics = {
    'accuracy': accuracy
}

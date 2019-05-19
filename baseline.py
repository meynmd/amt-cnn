import torch
import torch.nn as nn

class LanguageModeler(nn.Module):
    def __init__(self, rnn_size, rnn_layers):
        super(LanguageModeler, self).__init__()
        self.name = 'baseline LSTM dim {}'.format(rnn_size)
        self.rnn_dim = rnn_size
        self.rnn_layers = rnn_layers
        self.dropout = 0
        if self.rnn_layers > 1:
            self.dropout = 0.5

        self.net = nn.LSTM(input_size=88, hidden_size=self.rnn_dim, num_layers=self.rnn_layers, dropout=self.dropout)
        self.output = nn.Linear(self.rnn_dim, 88)


    def forward(self, x):
        b, _, h, w = x.shape
        x = x.squeeze(1).transpose(1, 2).transpose(0, 1)
        # (seq, batch, dim)
        z, _ = self.net(x)
        # (seq, batch, dim) => (seq, batch, 88)
        z = self.output(z)
        # (seq, batch, 88) => (batch, 88, seq)
        z = z.transpose(0, 1).transpose(1, 2)
        # (batch, 88, seq) => (batch, 1, 88, seq)
        return z[:, :, w//2:]



import torch
import torch.nn as nn


class GeneralGRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, input_size, hidden_size, n_layers, gru_dropout_p, bidirectional=False):
        super(GeneralGRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.feed_forward = nn.Linear(self.hidden_size, vocab_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.gru = nn.GRU(input_size, hidden_size, num_layers=n_layers, batch_first=True,
                          dropout=gru_dropout_p, bidirectional=bidirectional)
        # Note that batch_first is true, meaning that we can feed the output of embedding
        # straight into gru.
        # Also note that nn.GRU is *not* nn.GRUCell.

    # input_tensors should have shape (batch_size, window_size)
    def forward(self, input_tensors, initial_hidden, ff=True, contains_extra_input_data=False, extra_input_data=None):
        # embedding input: a LongTensor of shape (batch_size, window_size)
        # embedding output: a tensor of shape (batch_size, window_size, embedding_size)
        embedded = self.embedding(input_tensors)

        if contains_extra_input_data:
            embedded = torch.cat([embedded, extra_input_data], dim=2)

        # gru input:
        # 1. input tensor: tensor of shape (batch_size, window_size. input_size)
        # 2. hidden_tensors: tensor of shape (num_layers, batch_size, hidden_size)
        # Note that input_size = embedding_size

        # gru output: tuple containing
        # 1. output: tensor of shape (batch_size, window_size, hidden_size)
        # 2. hidden: tensor of shape (num_layers, batch_size, hidden_size)
        outputs, hidden = self.gru(embedded, initial_hidden)

        linear_output = outputs
        if ff:
            # linear_output has size  (batch_size, window_size, vocab_size)
            linear_output = self.feed_forward(outputs)

        # returns a tensor of size (batch_size, window_size, vocab_size)
        return linear_output, hidden
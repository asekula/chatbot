
import torch
import torch.nn as nn


class RegularFF(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs, dropout_p):
        super(RegularFF, self).__init__()

        # Model.
        self.feed_forward1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.feed_forward2 = nn.Linear(hidden_size, num_outputs)

    # input_tensors should have shape (batch_size, window_size)
    # hidden_tensors should have shape (num_layers, batch_size, hidden_size)
    # lengths is a *list* of size (batch_size), each element is the index of the last word
    # (could be 0 if sequence length is 1) note that this is one minus the actual length
    def forward(self, input_batch_tensor):
        # first layer has shape (batch_size, window_size, hidden_size) <- double check
        first_layer = self.relu(self.feed_forward1(input_batch_tensor))
        first_layer_dropout = self.dropout(first_layer)

        return self.feed_forward2(first_layer_dropout)


from hyperparameters import *

from chat_training_data import TrainingData
from general_gru import GeneralGRU
from helper_functions import make_input_padded_tensor, make_padded_tensor, expected_output
import math
import pickle
import os.path

import torch
import torch.nn as nn
from torch.autograd import Variable
import random

filename = 'all_response_chat_data'
continue_training = True


def word_dropout(words, dropout=0.25):
    dropped_out = []
    for word in words:
        if random.random() < dropout:
            dropped_out.append(1)
        else:
            dropped_out.append(word)


def train_on_batch(batch):
    # max length is window size, because make_(input_)padded_tensor functions truncate input at window_size.
    # so that even with padding, the length of the tensor is window_size
    output_lengths = [min(len(seq[1]) + 1, window_size) for seq in batch]
    # ^Not len(seq) + 1 for the STOP at the end -- we want to predict the STOP, not use it for prediction.

    # Make_padded_tensor converts the list of ints to a list of tensors that are padded to be of length 28.
    seqs = [make_input_padded_tensor(seq[0], window_size) for seq in batch]
    output_seqs = [make_padded_tensor(seq[1], window_size) for seq in batch]



    # Expected_output converts each seq to a seq shifted over by 1, with a padded STOP at the end.
    # Note that the sequence might not already have a padded STOP, if it's length is 27.
    # seq is a tensor of shape (window_size), so is expected_output(seq).
    expected_seqs = [expected_output(seq) for seq in output_seqs]

    word_dropout_output_seqs = [make_padded_tensor(word_dropout(seq[1]), window_size) for seq in batch]

    batch_tensor = torch.cat([seq.unsqueeze(0) for seq in seqs], dim=0)
    output_batch_tensor = torch.cat([seq.unsqueeze(0) for seq in output_seqs], dim=0)

    # encoder_output has size (batch_size, window_size, hidden_size * 2)
    # hidden_states has size (num_layers * 2, batch_size, hidden_size)
    _, hidden_states = encoder_network(Variable(batch_tensor),
                                       Variable(torch.zeros(encoder_network.n_layers * 2,
                                                            batch_tensor.size()[0],
                                                            encoder_network.hidden_size)),
                                       ff=False)

    # Converts hidden_states into a tensor of shape (num_layers, batch_size, hidden_size)
    # shiften hidden has size (num_layers * 2, batch_size, hidden_size)
    shifted_hidden = torch.cat([hidden_states[1:], hidden_states[0].unsqueeze(0)], dim=0)

    # joined hidden has size (num_layers * 2, batch_size, hidden_size)
    joined_hidden = torch.cat([hidden_states, shifted_hidden], dim=2)
    indices = [(x * 2) for x in range(n_layers)]
    index_tensor = Variable(torch.LongTensor(indices).unsqueeze(1).unsqueeze(2).expand(-1, len(batch), hidden_size * 2))
    corrected_hidden = torch.gather(joined_hidden, dim=0, index=index_tensor)

    # initial_hidden should have size (num_layers, batch_size, hidden_size)
    outputs, _ = decoder_network(Variable(output_batch_tensor), corrected_hidden, ff=True)

    flattened_output = outputs.view(len(batch) * window_size, vocab_size)
    flattened_expected_output = torch.cat(
        [seq.unsqueeze(0) for seq in expected_seqs], dim=0).view(len(batch) * window_size)

    # Note that reduce is false here so that we can select specific losses on which to backpropagate.
    loss = criterion(flattened_output, Variable(flattened_expected_output))

    # Only select the losses for input words that aren't STOP.
    word_indices = []
    for j in range(len(output_lengths)):
        word_indices.extend([(j * window_size) + x for x in range(output_lengths[j])])
        # +1 because we want to predict the last STOP.

    # torch.gather selects specific losses.
    gathered_loss = torch.gather(loss, dim=0, index=Variable(torch.LongTensor(word_indices), requires_grad=False))
    total_loss = sum(gathered_loss)
    num_loss = sum(output_lengths)

    return total_loss, num_loss


# Begin training code.

data = TrainingData(filename)
vocab_size = data.lang.vocab_size
print('Loaded and unked training data.')

print('Vocab size: ' + str(vocab_size))

# Explicit parameters to avoid messing up hyperparameter order.
encoder_network = GeneralGRU(vocab_size=vocab_size, embedding_size=embedding_size, input_size=embedding_size,
                             hidden_size=hidden_size, n_layers=n_layers, gru_dropout_p=gru_dropout_p,
                             bidirectional=True)
decoder_network = GeneralGRU(vocab_size=vocab_size, embedding_size=embedding_size,
                             hidden_size=2 * hidden_size, input_size=embedding_size,
                             n_layers=n_layers, gru_dropout_p=gru_dropout_p, bidirectional=False)

if continue_training:
    encoder_network.load_state_dict(torch.load("saved_model/" + filename + "_saved_encoder.ptc"))
    decoder_network.load_state_dict(torch.load("saved_model/" + filename + "_saved_decoder.ptc"))


criterion = nn.CrossEntropyLoss(reduce=False)

all_params = list(encoder_network.parameters()) + list(decoder_network.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, all_params), lr=learning_rate)
past_losses = []
loss_graph_data = []
num_batches = data.num_batches(batch_size)
dev_set = data.get_dev_set()
past_dev_perplexity = 1e10

for epoch in range(epochs):
    for i in range(num_batches):
        batch = data.get_batch(i, batch_size)

        optimizer.zero_grad()
        total_loss, num_loss = train_on_batch(batch)

        total_loss.backward()
        optimizer.step()

        past_losses.append(total_loss.data[0] / num_loss)
        loss_graph_data.append(total_loss.data[0] / num_loss)

        if (i + 1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   % (epoch + 1, epochs, i + 1, num_batches, math.exp(sum(past_losses) / len(past_losses))))
            past_losses = []

    total_loss, num_loss = train_on_batch(dev_set)
    dev_perplexity = math.exp(total_loss.data[0] / num_loss)
    print('Development set perplexity: ' + str(dev_perplexity))

    # Early stopping.
    if dev_perplexity > past_dev_perplexity:
        past_dev_perplexity = dev_perplexity
        break
    else:
        past_dev_perplexity = dev_perplexity

    stored_dev_perplexity = 1e11
    if os.path.exists('saved_model/' + filename + '_saved_dev_perplexity.p'):
        stored_dev_perplexity = pickle.load(open('saved_model/' + filename + '_saved_dev_perplexity.p', 'rb'))

    if stored_dev_perplexity > past_dev_perplexity:
        # Saves state dicts of encoder and decoder.
        torch.save(encoder_network.state_dict(), "saved_model/" + filename + "_saved_encoder.ptc")
        torch.save(decoder_network.state_dict(), "saved_model/" + filename + "_saved_decoder.ptc")
        pickle.dump(get_hyperparameter_dict(), open('saved_model/' + filename + '_hyperparameters.p', 'w'))
        pickle.dump(past_dev_perplexity, open('saved_model/' + filename + '_saved_dev_perplexity.p', 'w'))

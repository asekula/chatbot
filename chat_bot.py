
# This file is the repl that runs the trained chatbot.

from hyperparameters import *
from general_gru import GeneralGRU
from chat_training_data import TrainingData

import torch
from torch.autograd import Variable

from prompt_toolkit import prompt
from nltk.tokenize import sent_tokenize, word_tokenize
# Using both because that's how the data was tokenized -- not sure if sent_tokenize is redundant. It probably is.
from nltk.tokenize.moses import MosesDetokenizer
import math
import random

filename = 'all_response_chat_data'

def argmax(l):
    best_index = 0
    best = l[0]
    for i in range(1, len(l)):
        if l[i] > best:
            best = l[i]
            best_index = i
    return best_index


def random_sample(distr):
    if random.random() > 0.7:
        return argmax(distr)
    softmaxed = [math.exp(d) for d in distr]
    sum_softmax = sum(softmaxed)
    softmaxed = [x / sum_softmax for x in softmaxed]

    rand = random.random()
    i = 0
    while True:
        rand -= softmaxed[i]
        if rand <= 0:
            break
        i += 1
    return i

detokenizer = MosesDetokenizer()

data = TrainingData(filename)
vocab_size = data.lang.vocab_size
print('Loaded training data.')

print('Vocab size: ' + str(vocab_size))

# Explicit parameters to avoid messing up hyperparameter order.
encoder_network = GeneralGRU(vocab_size=vocab_size, embedding_size=embedding_size, input_size=embedding_size,
                             hidden_size=hidden_size, n_layers=n_layers, gru_dropout_p=gru_dropout_p,
                             bidirectional=True)
decoder_network = GeneralGRU(vocab_size=vocab_size, embedding_size=embedding_size,
                             hidden_size=2 * hidden_size, input_size=embedding_size,
                             n_layers=n_layers, gru_dropout_p=gru_dropout_p, bidirectional=False)

encoder_network.load_state_dict(torch.load("saved_model/" + filename + "_saved_encoder.ptc"))
decoder_network.load_state_dict(torch.load("saved_model/" + filename + "_saved_decoder.ptc"))

encoder_network.gru.dropout = 0.0
decoder_network.gru.dropout = 0.0


while True:

    input_line = prompt(u'User: ')
    if input_line == 'q':
        break

    sentences = sent_tokenize(input_line)
    words = [word_tokenize(sentence) for sentence in sentences]
    all_words = []
    for word_list in words:
        all_words.extend(word_list)

    all_words = [word.lower() for word in all_words]
    all_ids = [data.lang.get_word_id(word) for word in all_words] + [0]

    # seq is a LongTensor of shape (len(seq)) unsqueezing below because it should have a dimension for batch size
    seq = torch.LongTensor(all_ids)

    initial_hidden = Variable(torch.zeros(encoder_network.n_layers * 2, 1, encoder_network.hidden_size))
    # encoder_output has size (1, len(seq), hidden_size * 2)
    _, hidden_states = encoder_network(Variable(seq.unsqueeze(0)), initial_hidden, ff=False)

    # for chat bot purposes, we only care about the hidden states.

    shifted_hidden = torch.cat([hidden_states[1:], hidden_states[0].unsqueeze(0)], dim=0)
    # joined hidden has size (num_layers * 2, batch_size, hidden_size)
    joined_hidden = torch.cat([hidden_states, shifted_hidden], dim=2)
    indices = [(x * 2) for x in range(n_layers)]
    index_tensor = Variable(torch.LongTensor(indices).unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_size * 2))
    corrected_hidden = torch.gather(joined_hidden, dim=0, index=index_tensor)

    decoded_symbols = [0]
    current_hidden = corrected_hidden

    while len(decoded_symbols) < 100:  # doing a manual do-while with a break at the end as the while
        # initial_hidden should have size (num_layers, batch_size, hidden_size)
        input_tensors = Variable(torch.LongTensor([decoded_symbols[-1]]).unsqueeze(0))

        # input_tensors should have shape (batch_size, window_size)
        decoding, current_hidden = decoder_network(input_tensors, current_hidden, ff=True,
                                                   contains_extra_input_data=False)

        logits_list = decoding.squeeze(0).squeeze(0).data.tolist()

        symbol = random_sample(logits_list)
        decoded_symbols.append(symbol)

        if symbol == 0:
            break

    decoded_symbols = decoded_symbols[1:-1]
    decoded_words = [data.lang.id_to_word[i] for i in decoded_symbols]
    output_str = detokenizer.detokenize(decoded_words, return_str=True)
    print
    print('Chat bot: ' + output_str)
    print

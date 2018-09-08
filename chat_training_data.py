
import pickle
import random
import math
import torch
import numpy as np


class TrainingData:
    def __init__(self, filename, dev_split=0.15):  # filename is root of name, like 'user_response_chat_data'
        self.lang = pickle.load(open('data/' + filename + '_lang.p', 'rb'))
        self.inputs = []
        self.responses = []

        with open('data/' + filename + '_inputs.txt', 'rb') as f:
            for line in f.readlines():
                self.inputs.append([int(x) for x in line.split(' ')])
            f.close()

        with open('data/' + filename + '_responses.txt', 'rb') as f:
            for line in f.readlines():
                self.responses.append([int(x) for x in line.split(' ')])
            f.close()

        all_indices = range(len(self.inputs))
        random.shuffle(all_indices)
        max_dev_index = int(math.ceil(len(self.inputs) * dev_split))

        self.dev_indices = all_indices[:max_dev_index]
        self.train_indices = all_indices[max_dev_index:]

    def num_batches(self, batch_size):
        return (len(self.train_indices) / batch_size) + 1

    def get_batch(self, batch_index, batch_size):
        starting_index = batch_index * batch_size
        ending_index = min(len(self.train_indices), (batch_index + 1) * batch_size)
        indices = self.train_indices[starting_index:ending_index]
        return [(self.inputs[i], self.responses[i]) for i in indices]

    def get_dev_set(self):
        return [(self.inputs[i], self.responses[i]) for i in self.dev_indices]


if __name__ == '__main__':
    t = TrainingData('all_response_chat_data')

    print(t.lang.vocab_size)
    import sys
    sys.exit()

    a = [len(i) for i in t.inputs]
    b = [len(i) for i in t.responses]

    input_lens = np.array(a)
    response_lens = np.array(b)

    print(np.mean(input_lens))
    print(np.mean(response_lens))

    print(np.std(input_lens))
    print(np.std(response_lens))

    import matplotlib.pyplot as plt

    plt.hist(a, bins=range(100))
    plt.show()
    plt.hist(b, bins=range(100))
    plt.show()

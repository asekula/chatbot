
import pandas
from lang import Lang
import pickle

# Hyperparameters:
filename = 'all_response_chat_data'
unk_threshold = 3


def combine_lists(l):
    combined = []
    for inner_list in l:
        combined.extend(inner_list)
    return combined


df = pandas.read_csv('data/' + filename + '.csv', encoding='utf-8')
df.fillna('', inplace=True)
df = df.values

inputs = []
responses = []

for row in df:
    if row[1] != '' and row[2] != '':
        inputs.append(row[1])
        responses.append(row[2])

print(len(inputs))
print(len(responses))

lang = Lang()

for i in range(len(inputs)):
    lang.tokenize(inputs[i])
    lang.tokenize(responses[i])

prev_vocab_size = lang.vocab_size
lang.unk_data(unk_threshold=unk_threshold)

print("prev: " + str(prev_vocab_size))
print("curr: " + str(lang.vocab_size))

tokenized_inputs = []
tokenized_responses = []

for i in range(len(inputs)):
    tokenized_inputs.append(combine_lists(lang.tokenize(inputs[i], create_ids=False)))
    tokenized_responses.append(combine_lists(lang.tokenize(responses[i], create_ids=False)))


with open('data/' + filename + '_inputs.txt', 'w') as f:
    for input_list in tokenized_inputs:
        f.write(' '.join([str(x) for x in input_list]) + '\n')
    f.close()

with open('data/' + filename + '_responses.txt', 'w') as f:
    for response in tokenized_responses:
        f.write(' '.join([str(x) for x in response]) + '\n')
    f.close()

with open('data/' + filename + '_lang.p', 'w') as f:
    pickle.dump(lang, f)
    f.close()

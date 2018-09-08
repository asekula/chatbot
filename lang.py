
from nltk.tokenize import sent_tokenize, word_tokenize


# Keeps track of the word/id mappings, and vocab size. Tokenizes inputs. Can also unk data.
class Lang:
    def __init__(self, include_stop=True):
        # Note that this only deals with lowercased tokens.
        if include_stop:
            self.word_to_id = {"STOP": 0, "UNK": 1}
            self.id_to_word = ['STOP', 'UNK']
            self.vocab_size = 2
        else:
            self.word_to_id = {}
            self.id_to_word = []
            self.vocab_size = 0

        self.word_counts = {}  # This dict doesn't include STOP and UNK because we don't want to consider deleting them.

    # any_length_string is any length of string. note that this includes multiple sentences.
    # need to sentence-tokenize and word-tokenize.
    # returns a list of list of tokens, un-UNKed. Outermost list collects the sentences, inner lists collect word ids.
    # if create_ids is True, then any unknown words will be added to the list/dict. if False, then unknown words will
    # be marked as UNK
    # Note that this function will not pad stops anywhere.
    # This only adds to word counts if create_ids is true, i.e. if it's still in the pre-unking phase.
    def tokenize(self, any_length_string, create_ids=True):
        sentences = sent_tokenize(any_length_string)
        sentence_ids = []
        for sentence in sentences:
            word_tokens = [word.lower() for word in word_tokenize(sentence)]  # Note the word.lower() here.

            if create_ids:  # if we want to add unknown words to the dictionary
                for word in word_tokens:
                    if word not in self.word_to_id:
                        self.word_to_id[word] = self.vocab_size
                        self.id_to_word.append(word)
                        self.vocab_size += 1
                        self.word_counts[word] = 1
                    else:
                        self.word_counts[word] += 1

            sentence_ids.append([self.get_word_id(word) for word in word_tokens])

        return sentence_ids

    def whitespace_tokenize(self, line, create_ids=True, delimiter=' '):
        word_tokens = [word.lower() for word in line.split(delimiter)[:-1]] + ["STOP"]  # Note the word.lower() here.
        word_tokens = filter(lambda x: x != '', word_tokens)

        if create_ids:  # if we want to add unknown words to the dictionary
            for word in word_tokens:
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word.append(word)
                    self.vocab_size += 1
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1

        return [self.get_word_id(word) for word in word_tokens]


    # doesn't return anything. modifies word_to_id and id_to_word
    # any word count *less than or equal to* unk threshold gets UNKed.
    # note that word ids will change after this is called -- any external data stored as ids will be garbage
    def unk_data(self, unk_threshold=5):
        unked_word_to_id = {"STOP": 0, "UNK": 1}
        unked_id_to_word = ['STOP', 'UNK']
        unked_vocab_size = 2

        for word in self.word_counts:
            if self.word_counts[word] > unk_threshold:
                unked_word_to_id[word] = unked_vocab_size
                unked_id_to_word.append(word)
                unked_vocab_size += 1

        self.word_to_id = unked_word_to_id
        self.id_to_word = unked_id_to_word
        self.vocab_size = unked_vocab_size
        self.word_counts = {}  # To save memory.

    def get_word_id(self, word):
        if word in self.word_to_id:
            return self.word_to_id[word]
        else:
            return self.word_to_id['UNK']
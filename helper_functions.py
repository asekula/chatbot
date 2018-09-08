
import torch
from torch.autograd import Variable

# TODO: If the input list is too long, then truncate it depending on where the stops are being padded


# Returns list with padded stops at the front, with one padded stop at the end.
def make_input_padded_tensor(unbounded_int_list, window_sz):
    int_list = unbounded_int_list
    if len(int_list) >= (window_sz - 1):
        int_list = int_list[len(int_list) - window_sz + 1:]

    t = torch.LongTensor(int_list)
    zeros = [0] * (window_sz - len(int_list) - 1)
    zeros_tensor = torch.LongTensor(zeros)
    single_zero_tensor = torch.LongTensor([0])
    return torch.cat([zeros_tensor, t, single_zero_tensor], dim=0)


# pads stops at the end.
def make_padded_tensor(unbounded_int_list, window_sz):
    int_list = unbounded_int_list
    if len(int_list) >= (window_sz - 1):
        int_list = int_list[:window_sz - 1]

    t = torch.LongTensor(int_list)
    zeros = [0] * (window_sz - len(int_list) - 1)
    zeros_tensor = torch.LongTensor(zeros)
    single_zero_tensor = torch.LongTensor([0])
    return torch.cat([single_zero_tensor, t, zeros_tensor], dim=0)


def expected_output(seq_tensor):
    padded_stop = torch.LongTensor([0])
    return torch.cat([seq_tensor[1:], padded_stop], dim=0)


# parse is the variable list of the parse sequence.
# offset is an integer indicating where the pointer should start (doesn't necessarily start at 0 or 1, since we're
# front-padding).
# word_id is the id of WORD in the parse vocab.
# returns a Variable LongTensor of size (parse_window_size)
def attn_seq(parse, offset, parse_window_size, word_id):
    pointer = offset  # Starts looking at the encoding right after the padded stops.
    attn = []
    for elt in parse:
        attn.append(pointer)
        if elt == word_id:
            pointer += 1

    # To get the size of the attn tensor equal to the parse_window_size.
    # This might happen because parse is a list without padded STOPS.
    while len(attn) < parse_window_size: # Don't do != here. If it's greater than parse_window_size,
        #  exception thrown in next if statement.
        attn.append(pointer) # This should keep pointing at the last STOP.

    # Important (and maybe confusing): the input encoding is front-padded, but the decoding is end-padded.

    if len(attn) != parse_window_size:
        raise Exception("Attn list should have length parse_window_size.")

    return Variable(torch.LongTensor(attn))

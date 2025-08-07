import json
import numpy as np
from itertools import islice
import jsonlines


def trim_to_last_word(self, hypothesis, tokenizer):
    last_valid = len(hypothesis)
    while last_valid > 0 and tokenizer.decode([hypothesis[last_valid - 1]]).startswith("▁"):
        print(
            f"Token {hypothesis[last_valid - 1]} ({self.seamless_m4t_vocab.get(hypothesis[last_valid - 1], '')}) is a part of a word")
        last_valid -= 1
    return hypothesis[:last_valid]


def make_alignments(datap, args):
    alignments = []
    src = datap[args.src_key]
    tgt = datap[args.tgt_key]
    src_words = src.split(" ")
    tar_words = tgt.split(" ")

    # alligning the full English sentences with all Czech substrings combinations
    for x in range(0, len(src_words), args.words_per_prefix):
        alignments.append((" ".join(src_words[0:x + args.words_per_prefix]), tgt))
        # alignments.append((" ".join(src[0:x+5]), [" ".join(tar[0:y+5] for y in range(len(src)))]))
    return alignments

def make_pair(datap, args):
    return datap[args.src_key], datap[args.tgt_key]

def get_data(args):
    if args.dataset_path.endswith(".jsonl"):
        with jsonlines.open(args.dataset_path) as reader:
            data = list(islice(reader, 10000))
    else:
        with open(args.dataset_path, "r") as f:
            data = json.load(f)
    words = [make_pair(data[i], args) for i in range(len(data))]
    prefixes = [[words[0]]]
    for x in words[1:]:
        if x[0].startswith(prefixes[-1][-1][0]):
            prefixes[-1].append(x)
        else:
            prefixes.append([x])
    print([len(x) for x in prefixes])

    data = [x for x in prefixes if len(x[-1][0]) > 0]
    r = []
    for x in data:
        words = x[-1][0].split(" ")
        inds = list(range(len(words)))
        np.random.shuffle(inds)
        def swap(i, j):
            words[inds[i]], words[inds[j]] = words[inds[j]], words[inds[i]]
        for i in range(args.num_swaps):
            swap(i*2, i*2+1)
        r.append((words, x[-1][1].split(" ")))
    return r
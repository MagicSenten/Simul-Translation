import json
from itertools import islice

import jsonlines
import numpy as np

def make_pair(datap, args):
    """
    Extracts a source-target pair from the dataset.

    Args:
        datap (dict): A dictionary containing source and target text.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: A tuple containing the source and target text.
    """
    return datap[args.src_key], datap[args.tgt_key]


def get_data(args):
    """
    Loads an ordered prefix dataset and extracts only the full length prefix (the full sentence). Does args.num_swaps swaps of randomly selected pairs of words in the full length prefix to simulate non-projectivity.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        list: A list of tuples where each tuple contains shuffled source words and target words.
    """
    # Load data from a JSONL or JSON file
    if args.dataset_path.endswith(".jsonl"):
        with jsonlines.open(args.dataset_path) as reader:
            data = list(islice(reader, 10000))
    else:
        with open(args.dataset_path, "r") as f:
            data = json.load(f)

    # Create source-target word pairs
    words = [make_pair(data[i], args) for i in range(len(data))]
    prefixes = [[words[0]]]
    for x in words[1:]:
        if x[0].startswith(prefixes[-1][-1][0]):
            prefixes[-1].append(x)
        else:
            prefixes.append([x])
    print([len(x) for x in prefixes])

    # Filter and shuffle the data
    data = [x for x in prefixes if len(x[-1][0]) > 0]
    r = []
    for x in data:
        words = x[-1][0].split(" ")
        inds = list(range(len(words)))
        np.random.shuffle(inds)

        def swap(i, j):
            """
            Swaps two words in the list based on their indices.

            Args:
                i (int): Index of the first word.
                j (int): Index of the second word.
            """
            words[inds[i]], words[inds[j]] = words[inds[j]], words[inds[i]]

        for i in range(args.num_swaps):
            swap(i * 2, i * 2 + 1)
        r.append((words, x[-1][1].split(" ")))
    return r
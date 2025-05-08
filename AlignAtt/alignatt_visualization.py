import json
from argparse import Namespace
from itertools import chain
import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class States:
    def __init__(self):
        self.stable_hypothesis = []
        self.hypothesis = []


def trim_to_last_word(self, hypothesis, tokenizer):
    last_valid = len(hypothesis)
    while last_valid > 0 and tokenizer.decode([hypothesis[last_valid - 1]]).startswith("▁"):
        print(
            f"Token {hypothesis[last_valid - 1]} ({self.seamless_m4t_vocab.get(hypothesis[last_valid - 1], '')}) is a part of a word")
        last_valid -= 1
    return hypothesis[:last_valid]


def local_agreement(self, states, new_hypothesis, segment_finished, tokenizer):
    curr_len = len(states.stable_hypothesis)
    if not segment_finished:
        stable_len = 0
        for stable_len, (a, b) in enumerate(zip(states.hypothesis, new_hypothesis)):
            if a != b:
                break
        states.hypothesis = new_hypothesis

        # nothing new
        if stable_len <= curr_len:
            return ""

        if self.output_words_valid_words_only:
            new_stable_hypothesis = trim_to_last_word(
                new_hypothesis[:stable_len]
            )
        else:
            new_stable_hypothesis = new_hypothesis[:stable_len]
    else:
        new_stable_hypothesis = new_hypothesis

    if len(new_stable_hypothesis) > curr_len:
        states.stable_hypothesis = new_stable_hypothesis
        new_tokens = new_stable_hypothesis[curr_len:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return new_text
    else:
        return ""


def make_alignments(datap):
    alignments = []
    src = datap["czech"].split(" ")
    tar = datap["english"].split(" ")

    # alligning the full English sentences with all Czech substrings combinations
    for x in range(0, len(src), 4):
        alignments.append((" ".join(src[0:x + 5]), datap["english"]))
        # alignments.append((" ".join(src[0:x+5]), [" ".join(tar[0:y+5] for y in range(len(src)))]))
    return alignments

def visualize_attention(input_ids, output_ids, attentions, tokenizer, args):
    def sort_top(l, t):
        return [y - len(input_ids) for y in l[:-t] + sorted(l[-t:])]

    def get_range(vs):
        if len(vs) < 3:
            return ""
        ids = input_ids[min(vs[-3:]):max(vs[-3:])]
        r = tokenizer.decode(ids)
        return r
    # get the top attention positions for the last 5 output tokens (-1 means last input token)
    print([sort_top(y[0, args.heads, -1, :].mean(0).argsort(-1)[-10:].tolist(), 3) for y in attentions[:-10]])
    # print the corresponding tokens
    print([get_range(x[0, args.heads, -1, :].mean(0).argsort(-1)[-10:].tolist()) for x in attentions[-10:-5]])
    print(tokenizer.decode(output_ids[-10:-5]))
    print([get_range(x[0, args.heads, -1, :].mean(0).argsort(-1)[-10:].tolist()) for x in attentions[:-5]])
    print(tokenizer.decode(output_ids[:-5]))

def alignatt(attentions, args):
    for i in range(len(attentions)):
        # shape (batch_size, num_heads, generated_length, sequence_length)
        mean_attentions = attentions[i][0, args.heads, -1, :].mean(0)
        # shape (generated_length)
        top_pos = mean_attentions.argsort(-1)[-args.top_k:].cpu().numpy()
        top_pos[top_pos >= attentions[0].shape[-1]-args.skip_l] = 0
        #print(attentions[i].shape, top_pos, mean_attentions[-mean_attentions.shape[0]//8:])
        print(i, len(attentions), attentions[0].shape[-1] - top_pos)
        if np.sum(np.less_equal(attentions[0].shape[-1] - args.last_f, top_pos)) > args.count_in:
            return i
    print(attentions[0].shape[-1] - top_pos)
    return len(attentions)


def translate(model, tokenizer, input_text, stable_theory, args, verbose=False):
    '''
        - 'prefix' Refers to a substring, for each substring.
        - 'pt': Return as pytorch tensor.
    '''
    is_sent_end = input_text.endswith(".")
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    decoder_input_ids = tokenizer.encode(stable_theory, return_tensors="pt")
    """
      cross_attentions (tuple(tuple(torch.FloatTensor)), optional,
      returned when output_attentions=True) —
      Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor
      of shape (batch_size, num_heads, generated_length, sequence_length).
    """
    outputs = model.generate(input_ids=input_ids.to(args.device), decoder_input_ids=decoder_input_ids.to(args.device),
                             return_dict_in_generate=True, output_attentions=True, max_new_tokens=200)
    ca = outputs["cross_attentions"]
    output_ids = outputs["sequences"][0]
    if args.top_k > 0 or is_sent_end:
        assert all([x[0].shape[2] == 1 for x in ca[1:]])
        attentions = [sum(x[i] for i in args.layers) for x in ca[1:]]
        attentions = attentions[:len(output_ids) - decoder_input_ids.shape[1]]
        if verbose:
            visualize_attention(input_ids[0], output_ids[decoder_input_ids.shape[1]:], attentions, tokenizer, args)
        alignatt_result = decoder_input_ids.shape[1] + alignatt(attentions, args)
    else:
        alignatt_result = len(output_ids)

    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    decoded_align_att = tokenizer.decode(output_ids[:alignatt_result], skip_special_tokens=True)
    return tokenizer.tokenize(decoded_align_att)


def analyze_dataset(args):
    data = json.load(open("iwslt2024_cs_devset.json"))
    # flattens all the allignments into 1 dimension list
    prefixes = [make_alignments(data[i]) for i in range(len(data))]
    id = 0

    ''''
      the model names used
      first model will be compared to the other modesl
    '''
    names = ["Helsinki-NLP/opus-mt-cs-en",
             "facebook/nllb-200-3.3B",
             "facebook/nllb-200-1.3B",
             "facebook/nllb-200-distilled-600M",
             "utter-project/EuroLLM-1.7B",
             "utter-project/EuroLLM-9B"]
    tokenizer = AutoTokenizer.from_pretrained(names[id])
    model = AutoModelForSeq2SeqLM.from_pretrained(names[id]).to(args.device)
    first = True
    bleu = evaluate.load("bleu")
    total_bleu = 0
    total_delay = 0
    # The number of prefixes seen.
    cs = 0
    for x in prefixes[:10]:
        wordsen = x[-1][1].split(" ")
        # We prefix it with some text to not start the translation from nothing.
        helper_text = "Následující dokument obsahuje přepis proslovu z evropského parlamentu. "
        lht = len(helper_text)
        helper_text_en = "The following document contains a transcript of a speech from the European Parliament. "
        lhten = len(helper_text_en)
        # We give it some of the first target golden words to make it more stable for evaluation.
        start = 4
        lhten_tok = len(tokenizer.tokenize(helper_text_en))
        stable_theory = tokenizer.tokenize(helper_text_en + x[-1][1])[:lhten_tok+int(start*2.5)]
        previous_theory = stable_theory
        # How much is the english text longer than the czech text.
        frac = len(x[-1][1]) / len(x[-1][0]) + 0.5
        for t in range(int(start * frac), len(x)):
            # print(frac * len(x[t][0]),  len(stable_theory), len(x[t][0]), len(x[-1][1]))
            new_delay = (frac * len(x[t][0])) - len(stable_theory)
            total_delay += new_delay
            new_theory = translate(model, tokenizer, helper_text + x[t][0], stable_theory, args, False)
            for i in range(len(stable_theory), min(len(new_theory), len(previous_theory))):
                if new_theory[i] != previous_theory[i]:
                    break
                stable_theory += new_theory[i]
            print("****", len(new_theory)-len(stable_theory))
            print(x[t][0])
            print("".join(stable_theory).replace("▁", " ")[lhten:])
            print(x[-1][1])
            previous_theory = new_theory
        new_bleu = bleu.compute(predictions=["".join(stable_theory).replace("▁", " ")], references=[x[-1][1]])["bleu"]
        total_bleu += new_bleu
        cs += 1
        print(new_bleu, total_bleu / cs, new_delay, total_delay / cs)


# default 0.28967596904893456 0.21614738454887503 71.79527559055117 59.460164212070396
# -100 0.1857398572730801 0.24759903978731826 41.23529411764707 158.65644156457532

def main():
    argsdict = {"skip_l": 3, "layers": [4], "top_k": 10, "last_f": 6, "count_in": 5, "heads": list(range(6)), "device": "cuda"}
    args = Namespace(**argsdict)
    analyze_dataset(args)


if __name__ == "__main__":
    main()

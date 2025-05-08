import json
from argparse import Namespace
import torch
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import argparse
import jsonlines
from itertools import islice
def parse_args():
    parser = argparse.ArgumentParser()
    id = 0
    keys = ["czech", "english"] if False else ["pref_source", "pref_target"]
    parser.add_argument("--skip_l", type=int, default=0, help="Number of last positions in attention_frame_size to ignore")
    parser.add_argument("--layers", type=int, nargs='+', default=[4], help="List of layer indices")
    parser.add_argument("--top_attentions", type=int, default=10, help="Top attentions to use, set to zero to disable alignatt.")
    parser.add_argument("--attention_frame_size", type=int, default=10, help="The excluded frame of last positions size")
    parser.add_argument("--count_in", type=int, default=5, help="How many values in the top_attentions must be in attention_frame_size from end for the position to be bad.")
    parser.add_argument("--heads", type=int, nargs='+', default=list(range(6)), help="List of attention heads")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--words_per_prefix", type=int, default=2, help="Words per prefix shown")
    parser.add_argument("--forced_bos_token_text", type=str, default=None, help="Forced BOS token text")
    parser.add_argument("--model_id", type=int, default=id, help="Model ID")
    parser.add_argument("--num_beams", type=int, default=10, help="Setting the num_beams to a multiple of three turns on diverse beam search with num_beams//3 groups.")
    parser.add_argument("--src_key", type=str, default=keys[0], help="Source key")
    parser.add_argument("--tgt_key", type=str, default=keys[1], help="Target key")

    return parser.parse_args()

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
        top_pos = mean_attentions.argsort(-1)[-args.top_attentions:].cpu().numpy()
        top_pos[top_pos >= attentions[0].shape[-1] - args.skip_l] = 0
        # print(attentions[i].shape, top_pos, mean_attentions[-mean_attentions.shape[0]//8:])
        if np.sum(np.less_equal(attentions[0].shape[-1] - args.attention_frame_size, top_pos)) > args.count_in:
            print(i, len(attentions), attentions[0].shape[-1] - top_pos)
            return i
    print(len(attentions), "full", attentions[0].shape[-1] - top_pos)
    return len(attentions)


def translate(model, tokenizer, input_text, stable_theory, args, verbose=False):
    '''
        - 'prefix' Refers to a substring, for each substring.
        - 'pt': Return as pytorch tensor.
    '''
    is_sent_end = input_text.endswith(".")
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(stable_theory)).unsqueeze(0) if len(stable_theory) > 0 else None
    """
      cross_attentions (tuple(tuple(torch.FloatTensor)), optional,
      returned when output_attentions=True) —
      Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor
      of shape (batch_size, num_heads, generated_length, sequence_length).
    """
    config = GenerationConfig(num_beams=args.num_beams, num_beam_groups=args.num_beams//3 if args.num_beams % 3 == 0 else 1, diversity_penalty=0.1 if args.num_beams % 3 == 0 and args.num_beams > 3 else 0, no_repeat_ngram_size=2,
                              length_penalty=0.98)
    outputs = model.generate(input_ids=input_ids.to(args.device), decoder_input_ids=decoder_input_ids.to(
        args.device) if decoder_input_ids is not None else None,
                             return_dict_in_generate=True, output_attentions=True, max_new_tokens=20,
                             generation_config=config, renormalize_logits=True, forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.forced_bos_token_text) if args.forced_bos_token_text is not None else None)
    ca = outputs["cross_attentions"]
    output_ids = outputs["sequences"][0]
    if args.top_attentions > 0 and not is_sent_end and not decoder_input_ids is None:
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
    with jsonlines.open("prefixes_dataset.jsonl") as reader:
        data = list(islice(reader, 200))
    words = [make_pair(data[i], args) for i in range(len(data))]
    prefixes = [[words[0]]]
    for x in words[1:]:
        print(x[0], prefixes[-1][-1][0])
        if x[0].startswith(prefixes[-1][-1][0]):
            prefixes[-1].append(x)
        else:
            prefixes.append([x])
    print([len(x) for x in prefixes])
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
    tokenizer = AutoTokenizer.from_pretrained(names[args.model_id],  src_lang="ces_Latn", tgt_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(names[args.model_id], attn_implementation="eager").to(args.device)
    first = True
    bleu = evaluate.load("bleu")
    total_bleu = 0
    total_delay = 0
    # The total number of prefixes seen.
    cs = 0
    wait_for = 2
    for x in prefixes[:10]:
        if len(x) < wait_for+3:
            continue
        wordsen = x[-1][1].split(" ")
        # We prefix it with some text to not start the translation from nothing.
        helpt = False
        helper_text = "Následující dokument obsahuje přepis proslovu z evropského parlamentu. " if helpt else ""
        lht = len(helper_text)
        helper_text_en = "The following document contains a transcript of a speech from the European Parliament. " if helpt else ""
        lhten = len(helper_text_en)
        # We give it some of the first target golden words to make it more stable for evaluation.
        start = 0
        lhten_tok = len(tokenizer.tokenize(helper_text_en))
        stable_theory = tokenizer.tokenize(helper_text_en + x[-1][1])[:lhten_tok + int(start * 2.5)]
        previous_theory = stable_theory
        # How much is the english text longer than the czech text.
        frac = len(x[-1][1]) / len(x[-1][0]) + 0.5
        s = int(start * frac)
        for t in range(s, len(x)):
            # print(frac * len(x[t][0]),  len(stable_theory), len(x[t][0]), len(x[-1][1]))
            new_delay = (frac * len(x[t][0])) - len(stable_theory)
            total_delay += new_delay
            if t < wait_for:
                continue
            new_theory = translate(model, tokenizer, helper_text + x[t][0], stable_theory, args, False)
            # If we begin repeating the same tokens, we don't take the output.
            newtokens = new_theory[len(stable_theory):]
            if np.unique(newtokens).shape[0] < len(newtokens) // 2:
                print("repeating tokens")
                stable_theory += [tokenizer.tokenize(" ")]
                continue
            stop = min(len(new_theory), len(previous_theory))
            for i in range(len(stable_theory), stop):
                if any([new_theory[j] != previous_theory[j] for j in range(i, min(stop, i+1))]) or len(new_theory) > 500:
                    print(new_theory[i], previous_theory[i])
                    break
                stable_theory += [new_theory[i]]
            print("****", len(new_theory) - len(stable_theory))
            print(x[t][0])
            print("".join(stable_theory).replace("▁", " ")[lhten:])
            print("".join(new_theory).replace("▁", " ")[lhten:])
            print(x[-1][1])
            previous_theory = new_theory
        new_bleu = bleu.compute(predictions=["".join(new_theory if False else stable_theory).replace("▁", " ")], references=[x[-1][1]])["bleu"] if len(stable_theory) > 0 else 0
        total_bleu += new_bleu
        cs += 1
        print(new_bleu, total_bleu / cs, new_delay, total_delay / cs)

# default 0.28967596904893456 0.21614738454887503 71.79527559055117 59.460164212070396
# -100 0.1857398572730801 0.24759903978731826 41.23529411764707 158.65644156457532

def main():
    args = parse_args()
    analyze_dataset(args)


if __name__ == "__main__":
    main()

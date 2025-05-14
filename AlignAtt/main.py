import json
import os
from argparse import Namespace
import torch
import evaluate
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForCausalLM
import argparse
import jsonlines
from itertools import islice
from alignatt import alignatt, visualize_attention
from evaluation import SimuEval
def parse_args():
    parser = argparse.ArgumentParser()
    keys = ["czech", "english"] if True else (["source", "target"] if False else ["pref_source", "pref_target"])
    parser.add_argument("--dataset_path", default="../Data_preparation/iwslt2024_cs_devset.json", type=str, help="Path to the jsonl file with data.")
    parser.add_argument("--local_agreement_length", type=int, default=0, help="Number of next tokens it must agree with the previous theory in")
    parser.add_argument("--skip_l", type=int, default=0, help="Number of last positions in attention_frame_size to ignore")
    parser.add_argument("--layers", type=int, nargs='+', default=[3,4], help="List of layer indices")
    parser.add_argument("--top_attentions", type=int, default=3, help="Top attentions to use, set to 0 to disable alignatt.")
    parser.add_argument("--attention_frame_size", type=int, default=10, help="The excluded frame of last positions size")
    parser.add_argument("--count_in", type=int, default=2, help="How many values in the top_attentions must be in attention_frame_size from end for the position to be bad.")
    parser.add_argument("--wait_for", type=int, default=0, help="A static wait time to apply on top of alignatt everywhere")
    parser.add_argument("--wait_for_beginning", type=int, default=5, help="A wait time to apply at the beginning")
    parser.add_argument("--heads", type=int, nargs='+', default=list(range(6)), help="List of attention heads")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--words_per_prefix", type=int, default=2, help="Words per prefix shown")
    parser.add_argument("--forced_bos_token_text", type=str, default=None, help="Forced BOS token text")
    parser.add_argument("--model_id", type=int, default=0, help="Model ID")
    parser.add_argument("--num_beams", type=int, default=5, help="Setting the num_beams to a multiple of three turns on diverse beam search with num_beams//3 groups.")
    parser.add_argument("--num_swaps", type=int, default=0, help="Number of word pairs to blindly swap.")
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

def get_data(args):
    if args.dataset_path.endswith(".jsonl"):
        with jsonlines.open(args.dataset_path) as reader:
            data = list(islice(reader, 200))
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

    data = [x for x in prefixes if len(x[-1][0]) > 100]
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

def translate_LLM(model, tokenizer, input_text, stable_theory, args, verbose=False):
    '''
        - 'prefix' Refers to a substring, for each substring.
        - 'pt': Return as pytorch tensor.
    '''
    is_sent_end = input_text.endswith(".")
    decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids()).unsqueeze(0) if len(stable_theory) > 0 else None
    #decoder_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    turns = [[{"role": "system", "text": "You are a simultaneous translation API. Translate the czech partial output of an ASR system given to English. Only output the english sentence do not explain your output."},
             {"role": "user", "text": input_text}]]
    input_ids = tokenizer.apply_chat_template(turns, return_tensors="pt").input_ids
    inputlen = input_ids.shape[1]
    input_ids = torch.cat([input_ids, decoder_input_ids], 1)
    """
      cross_attentions (tuple(tuple(torch.FloatTensor)), optional,
      returned when output_attentions=True) —
      Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor
      of shape (batch_size, num_heads, generated_length, sequence_length).
    """
    config = GenerationConfig(num_beams=args.num_beams, num_beam_groups=args.num_beams//3 if args.num_beams % 3 == 0 else 1, diversity_penalty=0.1 if args.num_beams % 3 == 0 and args.num_beams > 3 else 0, no_repeat_ngram_size=2,
                              length_penalty=0.98)
    outputs = model.generate(input_ids=input_ids.to(args.device),
                             return_dict_in_generate=True, output_attentions=True, max_new_tokens=20,
                             generation_config=config, renormalize_logits=True, forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.forced_bos_token_text) if args.forced_bos_token_text is not None else None)
    ca = outputs["attentions"]
    output_ids = outputs["sequences"][0]
    if args.top_attentions > 0 and not decoder_input_ids is None and len(ca[1:]) > 0:
        print([x[0].shape for x in ca[1:]])
        raise Exception()
        assert all([x[0].shape[2] == 1 for x in ca[1:]])
        attentions = [sum(x[i][:, :, :, :inputlen] for i in args.layers) for x in ca[1:]]
        attentions = attentions[:len(output_ids) - decoder_input_ids.shape[1]]
        if verbose:
            visualize_attention(input_ids[0], output_ids[decoder_input_ids.shape[1]:], attentions, tokenizer, args)
        alignatt_result = decoder_input_ids.shape[1] + alignatt(attentions, args)
    else:
        alignatt_result = len(output_ids)

    decoded_align_att = tokenizer.decode(output_ids[:alignatt_result], skip_special_tokens=True)
    return tokenizer.tokenize(decoded_align_att)

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
    if args.top_attentions > 0 and not decoder_input_ids is None and len(ca[1:]) > 0:
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

def to_string(tokens):
    return "".join(tokens).replace("▁", " ").strip(" ")


def analyze_dataset(args, model, tokenizer, prefixes):

    ''''
      the model names used
      first model will be compared to the other modesl
    '''
    first = True
    bleu = evaluate.load("sacrebleu")
    total_bleu = 0
    total_latency = np.zeros(3)
    # The total number of prefixes seen.
    cs = 0
    metric = SimuEval()
    for datap in prefixes[:10]:
        print(datap)
        words = datap[0]
        gold_text = " ".join(datap[1])
        # We prefix it with some text to not start the translation from nothing.
        helpt = False
        helper_text = "Následující dokument obsahuje přepis proslovu z evropského parlamentu. " if helpt else ""
        helper_text_en = "The following document contains a transcript of a speech from the European Parliament. " if helpt else ""
        lhten = len(helper_text_en)
        # We give it some of the first target golden words to make it more stable for evaluation.
        start = 0
        lhten_tok = len(tokenizer.tokenize(helper_text_en))
        stable_theory = tokenizer.tokenize(helper_text_en + gold_text)[:lhten_tok + int(start * 2.5)]
        previous_theory = []
        new_theory = previous_theory
        new_bleu = 0
        output_theories = []
        inputs = []
        for t in range(1, len(words), 2):
            partial_input_text = " ".join(words[:t+1])
            if t >= args.wait_for_beginning:
                if args.isLLM:
                    translate_LLM(model, tokenizer, helper_text + partial_input_text, stable_theory, args, True)
                else:
                    new_theory = translate(model, tokenizer, helper_text + partial_input_text, stable_theory, args, True)
                # If we begin repeating the same tokens, we don't take the output.
                newtokens = new_theory[len(stable_theory):]
                if np.unique(newtokens).shape[0] < len(newtokens) // 2:
                    print("repeating tokens")
                    stable_theory += tokenizer.tokenize(" ")
                    continue
                if args.local_agreement_length > 0 and len(previous_theory) > 0:
                    stop = min(len(new_theory), len(previous_theory))
                    for i in range(len(stable_theory), stop):
                        if any([new_theory[j] != previous_theory[j] for j in range(i, min(stop, i+args.local_agreement_length))]):
                            print(new_theory[i], previous_theory[i])
                            break
                        stable_theory += [new_theory[i]]
                else:
                    stable_theory = new_theory
            print("****", len(new_theory) - len(stable_theory))
            print(partial_input_text)
            print(to_string(stable_theory)[lhten:])
            print(to_string(new_theory)[lhten:])
            print(gold_text)
            inputs.append(partial_input_text)
            output_theories.append(to_string(stable_theory)[lhten:])
            previous_theory = new_theory

        metric.update(inputs, output_theories, gold_text, tokenizer)
        if len(stable_theory) > 0:
            new_bleu = bleu.compute(predictions=[to_string(stable_theory)],
                                references=[gold_text])["score"]
        else:
            new_bleu = 0
        total_bleu += new_bleu
        cs += 1
        print(metric.eval(), new_bleu, total_bleu/cs)
        #print(new_bleu, total_bleu / cs, list(zip(["new_delay_chars", "new_delay_words", "new_delay_tokens"], new_delay)), list(zip(["avg_delay_chars", "avg_delay_words", "avg_delay_tokens"], total_latency / cs)))

    with open("results.jsonl", "a") as f:
        f.write(json.dumps({"bleu": total_bleu/cs, "args": vars(args), "all_metrics": metric.eval()})+"\n")
# default 0.28967596904893456 0.21614738454887503 71.79527559055117 59.460164212070396
# -100 0.1857398572730801 0.24759903978731826 41.23529411764707 158.65644156457532

def main():
    random.seed(42)
    args = parse_args()
    args.model_id = 4
    names = ["Helsinki-NLP/opus-mt-cs-en",
             "facebook/nllb-200-3.3B",
             "facebook/nllb-200-1.3B",
             "facebook/nllb-200-distilled-600M",
             "utter-project/EuroLLM-1.7B-Instruct",
             "utter-project/EuroLLM-9B-Instruct"]
    args.isLLM = "LLM" in names[args.model_id]
    print(args.isLLM, names[args.model_id])
    if args.isLLM:
        tokenizer = AutoTokenizer.from_pretrained(names[args.model_id])
        model = AutoModelForCausalLM.from_pretrained(names[args.model_id], attn_implementation="eager").to(args.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(names[args.model_id], src_lang="ces_Latn", tgt_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(names[args.model_id], attn_implementation="eager").to(args.device)
    prefixes = get_data(args)

    while True:
          args.wait_for_beginning = random.randint(1, 5)
          for x in range(4):
            args.num_beams = random.randint(1, 9)
            args.top_attentions = 0
            args.local_agreement_length = 0
            analyze_dataset(args, model, tokenizer, prefixes)
            args.top_attentions = 0
            args.local_agreement_length = 1
            analyze_dataset(args, model, tokenizer, prefixes)
            args.top_attentions = 0
            args.local_agreement_length = 2
            analyze_dataset(args, model, tokenizer, prefixes)
            for x in range(6):
              args.local_agreement_length = random.randint(0, 2)
              args.layers = [random.randint(1, 5)]
              args.count_in = random.randint(2, 5)
              args.attention_frame_size = random.randint(2, 5)
              args.top_attentions = args.count_in+random.randint(1, 3)
              analyze_dataset(args, model, tokenizer, prefixes)

main()
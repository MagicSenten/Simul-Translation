import json
import os
from argparse import Namespace
import torch
import random
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerBase
import argparse
from itertools import islice
import jsonlines
from alignatt import alignatt, visualize_attention
from simueval import SimuEval
def parse_args():
    parser = argparse.ArgumentParser()
    keys = ["czech", "english"] if False else (["source", "target"] if True else ["pref_source", "pref_target"])
    parser.add_argument("--dataset_path", default="../Data_preparation/cleaned_eval_dataset.jsonl", type=str, help="Path to the jsonl file with data.")
    parser.add_argument("--local_agreement_length", type=int, default=0, help="Number of next tokens it must agree with the previous theory in")
    parser.add_argument("--skip_l", type=int, default=0, help="Number of last positions in attention_frame_size to ignore")
    parser.add_argument("--layers", type=int, nargs='+', default=[3,4], help="List of layer indices")
    parser.add_argument("--top_attentions", type=int, default=1, help="Top attentions to use, set to 0 to disable alignatt.")
    parser.add_argument("--output_file", type=str, default="results.jsonl")
    parser.add_argument("--attention_frame_size", type=int, default=10, help="The excluded frame of last positions size")
    parser.add_argument("--count_in", type=int, default=1, help="How many values in the top_attentions must be in attention_frame_size from end for the position to be bad.")
    parser.add_argument("--wait_for", type=int, default=0, help="A static wait time to apply on top of alignatt everywhere")
    parser.add_argument("--wait_for_beginning", type=int, default=3, help="A wait time to apply at the beginning")
    parser.add_argument("--heads", type=int, nargs='+', default=list(range(6)), help="List of attention heads")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--words_per_prefix", type=int, default=2, help="Words per prefix shown")
    parser.add_argument("--forced_bos_token_text", type=str, default=None, help="Forced BOS token text")
    parser.add_argument("--model_id", type=int, default=0, help="Model ID")
    parser.add_argument("--num_beams", type=int, default=2, help="Setting the num_beams to a multiple of three turns on diverse beam search with num_beams//3 groups.")
    parser.add_argument("--num_swaps", type=int, default=0, help="Number of word pairs to blindly swap.")
    parser.add_argument("--src_key", type=str, default=keys[0], help="Source key")
    parser.add_argument("--tgt_key", type=str, default=keys[1], help="Target key")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--experiment_type", type=str, default="none")

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

def translate_LLM(model, tokenizer, input_text, stable_theory, args, computation_stats, verbose=False):
    '''
        - 'prefix' Refers to a substring, for each substring.
        - 'pt': Return as pytorch tensor.
    '''
    is_sent_end = input_text.endswith(".")
    decoder_input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(stable_theory)).unsqueeze(0) if len(stable_theory) > 0 else None

    input_text = f'<|im_start|>system\nYou are simultaneous interpreter from Czech to English, you translate incomplete sentences, please make sure you only translate what is explicitly stated in the input segment.<|im_end|>\n<|im_start|>user\nTranslate the following Czech source text to English.\nCzech: {input_text}\nEnglish: <|im_end|>\n<|im_start|>assistant\n'
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    all_input_ids = torch.cat([input_ids, decoder_input_ids], 1) if decoder_input_ids is not None else input_ids
    """
      cross_attentions (tuple(tuple(torch.FloatTensor)), optional,
      returned when output_attentions=True) —
      Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor
      of shape (batch_size, num_heads, generated_length, sequence_length).
    """
    bad_words = ["English:", "Czech:", "<0x0A>", "Reference:"]
    config = GenerationConfig(num_beams=args.num_beams, num_beam_groups=args.num_beams//3 if args.num_beams % 3 == 0 else 1, diversity_penalty=0.1 if args.num_beams % 3 == 0 and args.num_beams > 3 else 0, no_repeat_ngram_size=2,
                              length_penalty=0.98 if args.num_beams > 1 else 1.0, bad_words_ids= [tokenizer.encode(x) for x in bad_words])
    outputs = model.generate(input_ids=all_input_ids.to(args.device), generation_config=config,
                             return_dict_in_generate=True, output_attentions=True, max_new_tokens=min(2, input_ids.shape[1]*1.5-len(stable_theory)), renormalize_logits=True, forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.forced_bos_token_text) if args.forced_bos_token_text is not None else None, pad_token_id=tokenizer.pad_token_id)

    ca = outputs["attentions"]
    len_output_ids = len(outputs["sequences"][0])
    output_ids = outputs["sequences"][0][decoder_input_ids.shape[1]+1 if decoder_input_ids is not None else 0:].cpu()
    if args.top_attentions > 0 and not decoder_input_ids is None and len(ca[1:]) > 0:
        print([x[0].shape for x in ca[1:]])
        raise Exception()
        assert all([x[0].shape[2] == 1 for x in ca[1:]])
        attentions = [sum(x[-1-i][:, :, :, all_input_ids.shape[1]:] for i in args.layers) for x in ca[1:]]
        attentions = attentions[:len(output_ids) - decoder_input_ids.shape[1]]
        if verbose:
            visualize_attention(input_ids[0], output_ids[decoder_input_ids.shape[1]:], attentions, tokenizer, args)
        alignatt_result = alignatt(attentions, args)
    else:
        alignatt_result = len(output_ids)
    decoded_align_att = tokenizer.convert_ids_to_tokens(output_ids[:alignatt_result], skip_special_tokens=True)
    return decoded_align_att

def translate(model, tokenizer: PreTrainedTokenizerBase, input_text, stable_theory, computation_stats, args, verbose=False):
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
                              length_penalty=0.98 if args.num_beams > 1 else 1.0)
    outputs = model.generate(input_ids=input_ids.to(args.device), decoder_input_ids=decoder_input_ids.to(
        args.device) if decoder_input_ids is not None else None,
                             return_dict_in_generate=True, output_attentions=True, max_new_tokens=10,
                             generation_config=config, renormalize_logits=True, forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.forced_bos_token_text) if args.forced_bos_token_text is not None else None)
    ca = outputs["cross_attentions"]
    print(len(ca), len(ca[0]), len(ca[0][0]), len(ca[0][0][0]))
    outputsequence = outputs["sequences"][0].cpu()
    print(tokenizer.convert_ids_to_tokens(outputsequence))
    if False:
        print(tokenizer.convert_ids_to_tokens(decoder_input_ids[0]))
        assert len(ca) - 2 == len(outputsequence) - decoder_input_ids.shape[1] - 1, f"or {len(ca)} {len(outputsequence)} {decoder_input_ids.shape[1]}"
    if outputsequence[-1] == tokenizer.eos_token_id:
        outputsequence = outputsequence[:-1]
    len_output = len(outputsequence)
    output_ids = outputsequence[decoder_input_ids.shape[1] if decoder_input_ids is not None else 0:].cpu()
    if args.top_attentions > 0 and not decoder_input_ids is None and len(ca) > 1:
        assert all([x[0].shape[2] == 1 for x in ca[1:]])
        #assert len(ca) == len(output_ids), f"or {len(ca)} {len(output_ids)}"
        attentions = [sum(x[-1-i][:1, :, -1:] for i in args.layers) for x in ca]
        if verbose:
            visualize_attention(input_ids[0], output_ids, attentions, tokenizer, args)
        alignatt_relative = alignatt(attentions, args)
        #If for some reason the attentions are too short remove from allignatt result
        extra_length =  (len_output - decoder_input_ids.shape[1]) - len(attentions)
        if extra_length > 0:
            computation_stats["attentions_too_short{extra_length}"] = computation_stats.get("attentions_too_short", 0) + 1
            alignatt_relative = alignatt_relative - extra_length

        alignatt_relative = max(0, alignatt_relative)
        alignatt_is_zero = alignatt_relative == 0
        computation_stats["total_its"] = computation_stats.get("total_its", 0) + 1
        if alignatt_is_zero:
            computation_stats["alignatt_is_zero"] = computation_stats.get("alignatt_is_zero", 0) + 1
        if alignatt_is_zero and len(input_text.split(" ")) % 4 == 0:
            alignatt_relative = 1
        alignatt_result = alignatt_relative
    else:
        alignatt_result = len(output_ids)

    decoded_align_att = tokenizer.convert_ids_to_tokens(output_ids[:alignatt_result], skip_special_tokens=False)
    return decoded_align_att

def to_string(tokens, tokenizer: PreTrainedTokenizerBase):
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)

def analyze_dataset_from_jsonl(args, data):
    inputs = data["inputs"]
    outputs = data["outputs"]
    texts = data["texts"]
    metric = SimuEval()
    for input, output, text in zip(inputs, outputs, texts):
        for x in range(len(input)):
            metric.update(input[x], output[x], text)
    with open(args.output_file, "a") as f:
        f.write(json.dumps({"bleu": metric.eval()["bleu"], "all_metrics": metric.eval()})+"\n")



def analyze_dataset(args, model, tokenizer, prefixes):
    ''''
      the model names used
      first model will be compared to the other modesl
    '''
    print(vars(args))
    first = True
    total_latency = np.zeros(3)
    # The total number of prefixes seen.
    cs = 0
    metric = SimuEval()
    data = prefixes
    all_inputs = []
    all_outputs = []
    all_texts = []
    computation_stats = {}
    repeating_tokens_num = 0
    for sentid, datap in enumerate(data):
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
        output_theories = []
        inputs = []
        per = 1

        for t in range(1, len(words)+per, per):
            t = min(t, len(words))
            partial_input_text = " ".join(words[:t+1])
            if t >= args.wait_for_beginning or t == len(words):
                if args.isLLM:
                    new_theory = stable_theory + translate_LLM(model, tokenizer, helper_text + partial_input_text, stable_theory, args, args.verbose)
                else:
                    new_theory = stable_theory + translate(model, tokenizer, helper_text + partial_input_text, stable_theory, computation_stats, args, args.verbose)
                # If we begin repeating the same tokens, we don't take the output.
                newtokens = new_theory[len(stable_theory):]
                if np.unique(newtokens).shape[0] < len(newtokens) // 2:
                    repeating_tokens_num += 1
                    stable_theory += tokenizer.tokenize(" ")
                    continue
                if args.local_agreement_length > 0 and len(previous_theory) > 0:
                    stop = min(len(new_theory), len(previous_theory))
                    for i in range(len(stable_theory), stop):
                        if any([new_theory[j] != previous_theory[j] for j in range(i, min(stop, i+args.local_agreement_length))]):
                            break
                        stable_theory += [new_theory[i]]
                else:
                    stable_theory = new_theory
            if args.verbose:
                print("****", len(new_theory) - len(stable_theory))
                print(partial_input_text)
                print(to_string(stable_theory, tokenizer)[lhten:])
                print(to_string(new_theory, tokenizer)[lhten:])
                print(gold_text)
            inputs.append(partial_input_text)
            output_theories.append(to_string(stable_theory, tokenizer)[lhten:])
            previous_theory = new_theory

        metric.update(inputs, output_theories, gold_text)
        all_inputs.append(inputs)
        for i in range(len(output_theories)-1):
            assert output_theories[i+1].startswith(output_theories[i]), str(output_theories)
        all_outputs.append(output_theories)
        all_texts.append(gold_text)
        cs += 1
        print(f"sent{sentid} of{len(data)}", computation_stats, metric.eval(), vars(args))

    with open(args.output_file, "a") as f:
        f.write(json.dumps({"bleu": metric.eval()["bleu"], "total": cs, "computation_stats": computation_stats, "args": vars(args), "all_metrics": metric.eval(), "stuck_count": repeating_tokens_num, "data": {"inputs": all_inputs, "outputs": all_outputs, "texts": all_texts}}, ensure_ascii=False)+"\n")
# default 0.28967596904893456 0.21614738454887503 71.79527559055117 59.460164212070396
# -100 0.1857398572730801 0.24759903978731826 41.23529411764707 158.65644156457532


def run_local_agreement(args, model, tokenizer, prefixes):
    for num_beams in range(1, 4):
        for wait_for_beginning in range(1, 4):
            args.top_attentions = 0
            args.local_agreement_length = 1
            args.num_beams = num_beams
            args.wait_for_beginning = wait_for_beginning
            analyze_dataset(args, model, tokenizer, prefixes)


def run_align_att(args, model, tokenizer, prefixes):
    for layer in range(1, 4):
        for frame_size in range(1, 4):
            args.attention_frame_size = frame_size
            args.layers = [layer]
            analyze_dataset(args, model, tokenizer, prefixes)

def main():
    random.seed(42)
    args = parse_args()
    names = ["Helsinki-NLP/opus-mt-cs-en",
             "facebook/nllb-200-3.3B",
             "facebook/nllb-200-1.3B",
             "facebook/nllb-200-distilled-600M",
             "utter-project/EuroLLM-1.7B-Instruct",
             "utter-project/EuroLLM-9B-Instruct",
             "davidruda/opus-mt-cs-en-Prefix-Finetuned"]
    args.isLLM = "LLM" in names[args.model_id]
    print(args.isLLM, names[args.model_id])
    if args.isLLM:
        tokenizer = AutoTokenizer.from_pretrained(names[args.model_id], token = "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")
        args.forced_bos_token_text = None
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(names[args.model_id], attn_implementation="eager", quantization_config=quantization_config, token = "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").to(args.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(names[args.model_id], src_lang="ces_Latn", tgt_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(names[args.model_id], attn_implementation="eager").to(args.device)
        print(tokenizer.supported_language_codes)
    prefixes = get_data(args)
    if args.experiment_type == "none":
        analyze_dataset(args, model, tokenizer, prefixes)
    elif args.experiment_type == "alignatt":
        run_align_att(args, model, tokenizer, prefixes)
    elif args.experiment_type == "local_agreement":
        run_local_agreement(args, model, tokenizer, prefixes)

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
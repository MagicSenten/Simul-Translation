import argparse
import random

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig

from AlignAtt.analyze_dataset import analyze_dataset
from AlignAtt.get_data import get_data
from Evaluation.simueval import SimuEval


def parse_args():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments with default values and descriptions.
    """
    parser = argparse.ArgumentParser()
    keys = ["czech", "english"] if False else (["source", "target"] if True else ["pref_source", "pref_target"])
    parser.add_argument("--dataset_path", default="./Data_preparation/cleaned_eval_dataset.jsonl", type=str, help="Path to the jsonl file with data.")
    parser.add_argument("--local_agreement_length", type=int, default=0, help="Number of next tokens it must agree with the previous theory in.")
    parser.add_argument("--skip_l", type=int, default=0, help="Number of last positions in attention_frame_size to ignore.")
    parser.add_argument("--layers", type=int, nargs='+', default=[2], help="List of layer indices.")
    parser.add_argument("--top_attentions", type=int, default=0, help="Top attentions to use, set to 0 to disable alignatt.")
    parser.add_argument("--output_file", type=str, default="results.jsonl", help="Output file to save results.")
    parser.add_argument("--attention_frame_size", type=int, default=10, help="The excluded frame of last positions size.")
    parser.add_argument("--count_in", type=int, default=1, help="How many values in the top_attentions must be in attention_frame_size from end for the position to be bad.")
    parser.add_argument("--wait_for", type=int, default=0, help="A static wait time to apply on top of alignatt everywhere.")
    parser.add_argument("--wait_for_beginning", type=int, default=3, help="A wait time to apply at the beginning.")
    parser.add_argument("--heads", type=int, nargs='+', default=list(range(6)), help="List of attention heads.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--words_per_prefix", type=int, default=2, help="Words per prefix shown.")
    parser.add_argument("--forced_bos_token_text", type=str, default=None, help="Forced BOS token text.")
    parser.add_argument("--model_id", type=int, default=0, help="Model ID to select from the predefined list of models.")
    parser.add_argument("--num_beams", type=int, default=2, help="Number of beams for beam search. Setting to a multiple of three enables diverse beam search.")
    parser.add_argument("--num_swaps", type=int, default=0, help="Number of word pairs to blindly swap.")
    parser.add_argument("--src_key", type=str, default=keys[0], help="Source key for the dataset.")
    parser.add_argument("--tgt_key", type=str, default=keys[1], help="Target key for the dataset.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output.")
    parser.add_argument("--experiment_type", type=str, default="none", help="Type of experiment to run (e.g., 'simple', 'alignatt').")

    return parser.parse_args()


def analyze_dataset_wrapper(args, model, tokenizer, prefixes):
    """
    Wrapper function to call the analyze_dataset function.

    Args:
        args (argparse.Namespace): Parsed arguments.
        model: The model to use for analysis.
        tokenizer: The tokenizer to use for the model.
        prefixes: Data prefixes to analyze.
    """
    analyze_dataset(args, model, tokenizer, prefixes, SimuEval())


def run_param_search(args, model, tokenizer, prefixes):
    """
    Runs parameter search by iterating over combinations of num_beams and wait_for_beginning.

    Args:
        args (argparse.Namespace): Parsed arguments.
        model: The model to use for analysis.
        tokenizer: The tokenizer to use for the model.
        prefixes: Data prefixes to analyze.
    """
    for num_beams in range(1, 4):
        for wait_for_beginning in range(1, 4):
            args.num_beams = num_beams
            args.wait_for_beginning = wait_for_beginning
            analyze_dataset_wrapper(args, model, tokenizer, prefixes)

def run_align_att(args, model, tokenizer, prefixes):
    """
    Runs alignment attention experiments by iterating over layers and frame sizes.

    Args:
        args (argparse.Namespace): Parsed arguments.
        model: The model to use for analysis.
        tokenizer: The tokenizer to use for the model.
        prefixes: Data prefixes to analyze.
    """
    args.top_attentions = 1
    for layer in range(1, 4):
        for frame_size in range(1, 4):
            args.attention_frame_size = frame_size
            args.layers = [layer]
            analyze_dataset_wrapper(args, model, tokenizer, prefixes)


def main():
    """
    Main function to execute the script. Initializes the model, tokenizer, and data,
    and runs the specified experiment type.
    """
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
        # Load a causal language model with quantization for large models
        tokenizer = AutoTokenizer.from_pretrained(names[args.model_id])
        args.forced_bos_token_text = None
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(names[args.model_id], attn_implementation="eager", quantization_config=quantization_config).to(args.device)
    else:
        # Load a sequence-to-sequence model
        tokenizer = AutoTokenizer.from_pretrained(names[args.model_id], src_lang="ces_Latn", tgt_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(names[args.model_id], attn_implementation="eager").to(args.device)
        print(tokenizer.supported_language_codes)
    prefixes = get_data(args)
    if args.experiment_type == "simple":
        run_param_search(args, model, tokenizer, prefixes)
    elif args.experiment_type == "alignatt":
        run_align_att(args, model, tokenizer, prefixes)


if __name__ == "__main__":
    # Enable TensorFloat-32 (TF32) for faster matrix multiplications on CUDA
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
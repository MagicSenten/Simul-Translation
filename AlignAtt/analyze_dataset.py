import json

import numpy as np
from transformers import PreTrainedTokenizerBase

from .local_agreement import local_agreement
from .translate import translate, translate_LLM


class States:
    """
    A class to represent the state of hypotheses during analysis.

    Attributes:
        stable_hypothesis (list): A list to store the stable hypothesis.
        hypothesis (list): A list to store the current hypothesis.
    """
    def __init__(self):
        """
        Initializes the States class with empty stable_hypothesis and hypothesis lists.
        """
        self.stable_hypothesis = []
        self.hypothesis = []


def to_string(tokens, tokenizer: PreTrainedTokenizerBase):
    """
    Converts a list of tokens into a string using the tokenizer.

    Args:
        tokens (list): A list of tokens to convert.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for decoding.

    Returns:
        str: The decoded string representation of the tokens.
    """
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)


def analyze_dataset(args, model, tokenizer, prefixes, metric):
    """
    Analyzes a dataset by processing prefixes and evaluating model outputs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model: The model to use for translation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for the model.
        prefixes (list): A list of prefixes to analyze.
        metric: An evaluation metric object to update and evaluate results.

    Writes:
        A JSON object containing evaluation results, computation statistics, and processed data
        to the output file specified in `args.output_file`.
    """
    print(vars(args))
    # The total number of prefixes seen.
    cs = 0
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
                    # Translate using a large language model (LLM).
                    new_theory = stable_theory + translate_LLM(model, tokenizer, helper_text + partial_input_text, stable_theory, args, args.verbose)
                else:
                    # Translate using a sequence-to-sequence model.
                    new_theory = stable_theory + translate(model, tokenizer, helper_text + partial_input_text, stable_theory, computation_stats, args, args.verbose)
                # If we begin repeating the same tokens, we don't take the output.
                newtokens = new_theory[len(stable_theory):]
                if np.unique(newtokens).shape[0] < len(newtokens) // 2:
                    repeating_tokens_num += 1
                    stable_theory += tokenizer.tokenize(" ")
                    continue
                # Apply local agreement if enabled.
                stable_theory = local_agreement(new_theory, previous_theory, stable_theory, args) if args.local_agreement_length > 0 and len(previous_theory) > 0 else new_theory
            if args.verbose:
                print("****", len(new_theory) - len(stable_theory))
                print(partial_input_text)
                print(to_string(stable_theory, tokenizer)[lhten:])
                print(to_string(new_theory, tokenizer)[lhten:])
                print(gold_text)
            inputs.append(partial_input_text)
            output_theories.append(to_string(stable_theory, tokenizer)[lhten:])
            previous_theory = new_theory

        # Update the evaluation metric with the inputs, outputs, and gold text.
        metric.update(inputs, output_theories, gold_text)
        all_inputs.append(inputs)
        for i in range(len(output_theories)-1):
            assert output_theories[i+1].startswith(output_theories[i]), str(output_theories)
        all_outputs.append(output_theories)
        all_texts.append(gold_text)
        cs += 1
        print(f"sent{sentid} of{len(data)}", computation_stats, metric.eval(), vars(args))

    # Write the evaluation results and output data to the output file.
    with open(args.output_file, "a") as f:
        f.write(json.dumps({
            "bleu": metric.eval()["bleu"],
            "total": cs,
            "computation_stats": computation_stats,
            "args": vars(args),
            "all_metrics": metric.eval(),
            "stuck_count": repeating_tokens_num,
            "data": {
                "inputs": all_inputs,
                "outputs": all_outputs,
                "texts": all_texts
            }
        }, ensure_ascii=False) + "\n")
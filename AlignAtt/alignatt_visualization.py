import json
import nltk
import itertools
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class States:
    def __init__(self):
        self.stable_hypothesis = []
        self.hypothesis = []

def trim_to_last_word(self, hypothesis, tokenizer):
        last_valid = len(hypothesis)
        while last_valid > 0 and tokenizer.decode([hypothesis[last_valid - 1]]).startswith("▁"):
            print(f"Token {hypothesis[last_valid - 1]} ({self.seamless_m4t_vocab.get(hypothesis[last_valid - 1], '')}) is a part of a word")
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
    for x in range(len(src)):
        alignments.append((" ".join(src[0:x]), datap["english"]))
        #alignments.append((" ".join(src[0:x]), [" ".join(tar[0:y] for y in range(len(src)))]))
    return alignments


def analyze_dataset():
    data = json.load(open("iwslt2024_cs_devset.json"))
    # flattens all the allignments into 1 dimension list
    prefixes = list(itertools.chain.from_iterable([make_alignments(data[i]) for i in range(len(data))]))
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
    try:
      model = model
    except:
      model = AutoModelForSeq2SeqLM.from_pretrained(names[id])
    first = True
    # iterating on each hyphot
    for x in prefixes:
        '''
            - 'prefix' Refers to a substring, for each substring.
            - 'pt': Return as pytorch tensor.
        '''
        input_ids = tokenizer.encode(x[0], return_tensors="pt")
        words = x[1].split(" ")
        prefixes = [" ".join(words[:x]) for x in range(len(words))]
        decoder_inputs = [s for s in prefixes if len(s) >= len(x[0])//4][0]
        decoder_input_ids = tokenizer.encode(decoder_inputs, return_tensors="pt")
        """
          cross_attentions (tuple(tuple(torch.FloatTensor)), optional,
          returned when output_attentions=True) —
          Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor
          of shape (batch_size, num_heads, generated_length, sequence_length).
        """
        outputs = model.generate(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict_in_generate = True, output_attentions = True)
        ca = outputs["cross_attentions"]
        #get the attention matrix of the last token of shape (batch_size, num_heads, 1, sequence_length)
        last_ouput_ca = ca[-1]
        if first:
          print(input_ids.shape, decoder_input_ids.shape)
          print(outputs.keys())
          print(type(ca), type(ca[0]), [[y.shape[2] for y in x] for x in ca])
          print(last_ouput_ca[0].shape)
          first = False
        c = 5
        def sort_top(l, t):
            return [y - input_ids.shape[1] for y in l[:-t] + sorted(l[-t:])]
        def get_range(vs):
            if len(vs) < 3:
                return ""
            ids = input_ids[0, :][min(vs[-3:]):max(vs[-3:])]
            r =tokenizer.decode(ids)
            return r
        print("****************")
        #get the top attention positions for the last 5 output tokens (-1 means last input token)
        print([[sort_top(x[0, :, 0].mean(0).argsort(-1)[-c:].tolist(), 3) for x in y] for y in ca[-5:]])
        #print the corresponding tokens
        print([get_range(x[0, :, 0].mean(0).argsort(-1)[-c:].tolist()) for x in last_ouput_ca])
        output_ids = outputs["sequences"][0]
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(x[0])
        print(x[1])
        print(decoded)

def main():
  analyze_dataset()

if __name__ == "__main__":
  main()
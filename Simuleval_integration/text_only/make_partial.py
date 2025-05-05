import json
import nltk
from transformers import AutoTokenizer, M2M100ForConditionalGeneration


data = json.load(open("iwslt2024_cs_devset.json"))
def make_alignments(datap):
    alignments = []
    src = datap["czech"].split(" ")
    tar = datap["english"].split(" ")

    for x in range(len(src)):
        alignments.append((" ".join(src[0:x]), datap["english"]))
    return alignments


def main():
    for i in range(len(data)):
        prefixes = make_alignments(data[i])
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M")
        for x in prefixes[len(prefixes)//2:]:
            input_ids = tokenizer.encode(x[0], return_tensors="pt")
            outputs = model.generate(input_ids, forced_bos_token_id=tokenizer.encode("en_Latn")[0], cache_implementation = "hybrid")
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("****************")
            print("src", x[0])
            print("tar", x[1])
            print("out", decoded)

if __name__ == "__main__":
    main()
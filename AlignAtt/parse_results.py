import json
import os

import numpy as np
def write_results(name):
    with open(name) as file:
        data = [json.loads(x) for x in file.read().split("\n") if len(x) > 0]
    os.makedirs("../AlignAttOutputs/parsed", exist_ok=True)
    outp = os.path.join("../AlignAttOutputs/parsed", os.path.splitext(os.path.basename(name))[0]) + ".json"
    with open(outp, "w") as f:
        all_data = []
        for x in data:
            if x["args"]["local_agreement_length"] > 0:
                    all_data.append({"bleu":x["bleu"], "delay_words":x["all_metrics"]["delay_words_absolute"], "num_beams":x["args"]["num_beams"], "wait_for_beginning":x["args"]["wait_for_beginning"]})
            if x["args"]["local_agreement_length"] == 0:
                    all_data.append({"bleu":x["bleu"], "delay_words":x["all_metrics"]["delay_words_absolute"], "attention_frame_size": x["args"]["attention_frame_size"], "layers": x["args"]["layers"], "num_beams":x["args"]["num_beams"], "wait_for_beginning":x["args"]["wait_for_beginning"]})
        json.dump(sorted(all_data, key=lambda x: x["bleu"], reverse=True), f, indent=4, ensure_ascii=False)

    for x in data:
        x.update(x["args"])
    bleus = np.array([x["bleu"] for x in data])

    def print_top(data):
        top_attentions = np.array([x["top_attentions"] for x in data])
        top_attentions_inds = np.where(top_attentions > 0)[0]
        top_np_attentions_inds = np.where(top_attentions == 0)[0]
        if len(top_attentions_inds) == 0 or len(top_np_attentions_inds) == 0:
            print("not enough representatives")
            return
        best_att = top_attentions_inds[np.argmax(bleus[top_attentions_inds])]
        best_no_att = top_np_attentions_inds[np.argmax(bleus[top_np_attentions_inds])]
        print("best attention", data[best_att])
        print("best_no_att", data[best_no_att])

    distinct_beams = set(x["num_beams"] for x in data)

    for x in distinct_beams:
        print(f"********* num_beams = {x}")
        print_top([y for y in data if y["num_beams"] == x])

if __name__ == "__main__":
    print(os.getcwd())
    root = "../AlignAttOutputs/results"
    for x in os.listdir(root):
        if x.endswith(".jsonl"):
            write_results(os.path.join(root,x))
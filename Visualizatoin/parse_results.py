import itertools
import json
import os
from itertools import islice
import numpy as np
from collections import OrderedDict

def write_results(name):
    with open(name) as file:
        data = [json.loads(x) for x in file.read().split("\n") if len(x) > 0]
    os.makedirs("../AlignAttOutputs/parsed", exist_ok=True)
    outp = os.path.join("../AlignAttOutputs/parsed", os.path.splitext(os.path.basename(name))[0]) + ".json"
    with open(outp, "w") as f:
        all_data = []
        if outp.endswith("finetuned_alignatt.json"):
            data = [x for x in data if x["args"]["top_attentions"] > 0]
        for x in data:
            make_e = True
            def make_examples(count, reduce):
                return [(list(itertools.chain.from_iterable(islice(zip(x, y), 0, len(x), reduce))), z)
                for x, y, z in zip(x["data"]["inputs"], x["data"]["outputs"], x["data"]["texts"])][:count]

            def first_if_one(x):
                if len(x) == 1:
                    return x[0]
                return x
            print(outp)

            if x["args"]["top_attentions"] > 0:
                all_data.append(OrderedDict([("bleu",x["all_metrics"]["bleu"]), ("attention_frame_size", x["args"]["attention_frame_size"]), ("layers", first_if_one(x["args"]["layers"])), ("num_beams",x["args"]["num_beams"]), ("wait_for_beginning",x["args"]["wait_for_beginning"]), ("all_metrics",x["all_metrics"]), ("example_sentances", make_examples(2, 3))]))
            else:
                all_data.append(OrderedDict([("bleu",x["all_metrics"]["bleu"]), ("num_beams",x["args"]["num_beams"]), ("wait_for_beginning",x["args"]["wait_for_beginning"]), ("all_metrics",x["all_metrics"]), ("example_sentances", make_examples(2, 3))]))

            if not make_e:
                all_data[-1].pop("example_sentances")
        all_data_sorted = sorted(all_data, key=lambda x: x["bleu"], reverse=True)
        json.dump(all_data_sorted, f, indent=4, ensure_ascii=False)

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

    return (all_data_sorted[0], name)

if __name__ == "__main__":
    print(os.getcwd())
    root = "../AlignAttOutputs/results"
    best = []
    for x in os.listdir(root):
        if x.endswith(".jsonl"):
            p = os.path.join(root,x)
            best.append(write_results(p))
    best = sorted(best, key=lambda x: x[0]["bleu"], reverse=True)
    with open("best.json", "w") as f:
        json.dump(best, f, indent=4, ensure_ascii=False)
    with open("best_names.json", "w") as f:
        json.dump([(x[0]["all_metrics"]["bleu"], x[0]["all_metrics"]["AL"], os.path.splitext(os.path.basename(x[1]))[0]) for x in best], f, indent=4, ensure_ascii=False)
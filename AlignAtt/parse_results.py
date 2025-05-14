import json
import numpy as np
with open("results.jsonl") as file:
    data = [json.loads(x) for x in file.read().split("\n") if len(x) > 0]
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
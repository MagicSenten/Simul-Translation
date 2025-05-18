import json
import os
import numpy as np
with open("best.json", "r") as f:
    data = json.load(f)

def tostr(x):
    if isinstance(x, float):
        return f"{x:.2f}"
    else:
        return str(x)

def makebold(x, bold=True):
    return "**" + x + "**" if bold else x

def make_table(name, headers, file, data, ismin):
    texheader = " | experiment name | " + " | ".join(headers) + "|"
    text_hline = " | ---- | " + " | ".join(["----" for x in headers]) + "|"
    texdata = ""
    sorted_keys = sorted(data, key=lambda x: x)
    besti = [np.argmax([data[x][i] * (-1 if ismin[i] else 1) for x in sorted_keys])  for i in range(len(data[sorted_keys[0]]))]
    for i, label in enumerate(sorted_keys):
       texdata += f"| {label} | {' | '.join([makebold(tostr(y), x == i) for x,y in zip(besti, data[label][:len(headers)])])} | \n"

    print(f"## {name}", file=file)
    print(texheader, file=file)
    print(text_hline, file=file)
    print(texdata,end="", file=file)

data = {os.path.splitext(os.path.basename(x[1]))[0].replace("_", " "): x[0] for x in data}
def make_data(metric_key):
    return {k: (v[metric_key]["bleu"], v[metric_key]["wer"], v[metric_key]["AL"]) for k,v in data.items()}, [False, True, True]

with open("RESULTS_OVERALL.md", "w") as f:
    make_table("Results on all data.", ["bleu", "word error rate", "average lagging"], f, *make_data("all_metrics"))
    make_table("Results on all sentences shorter than 100 cahracters.", ["bleu", "word error rate", "average lagging"], f, *make_data("all_metrics_long"))
    make_table("Results on all sentences longer than 100 cahracters.", ["bleu", "word error rate", "average lagging"], f, *make_data("all_metrics_short"))
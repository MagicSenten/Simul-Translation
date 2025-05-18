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
    return "\\textbf{" + x + "}" if bold else x

def make_table(name, headers, data, ismin):
    textabular = f"l|{'r'*len(headers)}"
    texheader = " & " + " & ".join(headers) + "\\\\"
    texdata = "\\hline\n"
    sorted_keys = sorted(data, key=lambda x: x)
    besti = [np.argmax([data[x][i] * (-1 if ismin[i] else 1) for x in sorted_keys])  for i in range(len(data[sorted_keys[0]]))]
    for i, label in enumerate(sorted_keys):
       if label == "z":
          texdata += "\\hline\n"
       texdata += f"{label} & {' & '.join([makebold(tostr(y), x == i) for x,y in zip(besti, data[label][:len(headers)])])} \\\\\n"

    print("\\begin{table}[H]")
    print("\\begin{tabular}{"+textabular+"}")
    print(texheader)
    print(texdata,end="")
    print("\\end{tabular}")
    print(f"\\caption{{{name}}}")
    print("\\end{table}")

data = {os.path.splitext(os.path.basename(x[1]))[0].replace("_", " "): x[0] for x in data}
def make_data(metric_key):
    return {k: (v[metric_key]["bleu"], v[metric_key]["wer"], v[metric_key]["AL"]) for k,v in data.items()}, [False, True, True]

make_table("Results on all data.", ["bleu", "word error rate", "average lagging"], *make_data("all_metrics"))
make_table("Results on all sentences shorter than 100 cahracters.", ["bleu", "word error rate", "average lagging"], *make_data("all_metrics_long"))
make_table("Results on all sentences longer than 100 cahracters.", ["bleu", "word error rate", "average lagging"], *make_data("all_metrics_short"))
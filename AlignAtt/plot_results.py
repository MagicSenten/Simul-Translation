import json
import os
with open("best.json", "r") as f:
    data = json.load(f)

def make_table(headers, data):
    textabular = f"l|{'r'*len(headers)}"
    texheader = " & " + " & ".join(headers) + "\\\\"
    texdata = "\\hline\n"
    for label in sorted(data, key=lambda x: data[x][0], reverse=True):
       if label == "z":
          texdata += "\\hline\n"
       texdata += f"{label} & {' & '.join(map(str,data[label][:len(headers)]))} \\\\\n"

    print("\\begin{tabular}{"+textabular+"}")
    print(texheader)
    print(texdata,end="")
    print("\\end{tabular}")

data = {os.path.splitext(os.path.basename(x[1]))[0].replace("_", " "): x[0] for x in data}
data = {k: (v["all_metrics"]["bleu"], v["all_metrics"]["AL"], v["all_metrics"]["wer"], v["num_beams"], v["wait_for_beginning"], v["example_sentances"][0][len(v["example_sentances"][0])//2], v["example_sentances"][-1]) for k,v in data.items()}

make_table(["bleu", "word error rate", "average lagging", "number of beams","wait_for_beginning", "halfway example", "full example"], data)

make_table(["bleu", "word error rate", "average lagging", "number of beams","wait k for beginning"], data)
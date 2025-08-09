# Simul-Translation 📝➡️📝

### Simul-Translation is a research-oriented toolkit for Simultaneous Machine Translation (SimulMT) with support for dataset preparation, model training, fine-tuning, evaluation, and visualization. It includes modular components for alignment-based policies, local agreement strategies, and output analysis.
----
### 🔧 Installation
Each module contains its own requirements.txt.
To install dependencies for a module, run:
```
pip install -r <module>/requirements.txt
```
Alternitavely, to install all dependencies for all modules, run:
```
pip install -r requirements.txt
```
----
### 📜 Usage
#### Example workflow:

---
### 📂 Project Structure
```
Simul-Translation/
├── .gitignore
├── main.py
├── AlignAtt/
│   ├── README.md
│   ├── alignatt.py
│   ├── analyze_dataset.py
│   ├── get_data.py
│   ├── local_agreement.py
│   ├── requirements.txt
│   └── translate.py
├── AlignAttOutputs/
│   ├── output_logs_grid_search_local_agreement/
│   ├── parsed/
│   └── results/
├── Data_preparation/
│   ├── README.md
│   ├── cleaned_eval_dataset.jsonl
│   ├── create_dataset.py
│   ├── iwslt2024_cs_devset.json
│   └── requirements.txt
├── Evaluation/
│   ├── README.md
│   ├── requirements.txt
│   └── simueval.py
├── Finetuning/
│   ├── README.md
│   ├── finetune_nmt_backbone.py
│   ├── finetuning.log
│   └── requirements.txt
└── Visualization/
    ├── RESULTS.md
    ├── RESULTS_OVERALL.md
    ├── best.json
    ├── create_tables.py
    ├── parse_results.py
    └── plot_results.py

```

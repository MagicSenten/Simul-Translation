# 📝 Simultanous Translation CZ → EN

### Simul-Translation is a research-oriented toolkit for text-to-text Simultaneous Machine Translation (SimulMT) with support for dataset preparation, model training, fine-tuning, evaluation, and visualization. It includes modular components for alignment-based policies, local agreement strategies, and output analysis.

## 🛠️ Technologies Used
<p align="center">
  <a href="https://www.python.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" height="38"></a>
  <a href="https://pytorch.org/"><img src="https://pytorch.org/assets/images/pytorch-logo.png" height="40"></a>
  <a href="https://huggingface.co/docs/datasets"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="40"></a>
  <a href="https://numpy.org/"><img src="https://numpy.org/images/logo.svg" height="40"></a>
  <a href="https://wandb.ai/"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg" height="40"></a>
  <a href="https://github.com/google/sentencepiece"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/480px-Google_%22G%22_logo.svg.png" height="40"></a>
  <a href="https://arrow.apache.org/docs/python/index.html"><img src="https://arrow.apache.org/docs/_static/arrow-dark.png" height="40"></a>
</p>

----
### 🔧 Installation
Each module contains its own requirements.txt. To install dependencies for a module, run:
```python
pip install -r <module>/requirements.txt
```
Example:
```python
pip install -r Data_preparation/requirements.txt
```
*Alternitavely*, to install all dependencies for all modules, run:
```python
pip install -r requirements.txt
```

----
## ⚙️ Command-line Arguments

| Argument | Type | Default | Description |
|-----------------------------------------|------|---------|-------------|
| `--dataset_path`                        | str  | `./Data_preparation/cleaned_eval_dataset.jsonl` | Path to the JSONL dataset file. |
| `--local_agreement_length`              | int  | `0`     | Number of next tokens to agree with the previous theory. |
| `--skip_l`                              | int  | `0`     | Number of last positions in `attention_frame_size` to ignore. |
| `--layers`                              | int list | `[3, 4]` | Layer indices to use. |
| `--top_attentions`                      | int  | `0`     | Top attentions to use (0 disables AlignAtt). |
| `--output_file`                         | str  | `results.jsonl` | Output file for results. |
| `--attention_frame_size`                | int  | `10`    | Excluded frame of last positions size. |
| `--count_in`                            | int  | `1`     | Required top_attentions within `attention_frame_size` for position to be bad. |
| `--wait_for`                            | int  | `0`     | Static wait time applied globally. |
| `--wait_for_beginning`                  | int  | `3`     | Wait time applied at the beginning. |
| `--heads`                               | int list | `0 1 2 3 4 5` | Attention heads to use. |
| `--device`                              | str  | `cuda`  | Device (`cuda` or `cpu`). |
| `--words_per_prefix`                    | int  | `2`     | Words per prefix shown. |
| `--forced_bos_token_text`               | str  | `None`  | Forced BOS token text. |
| `--model_id`                            | int  | `0`     | Model ID from predefined list. |
| `--num_beams`                           | int  | `2`     | Number of beams for beam search (multiple of 3 for diverse beam search). |
| `--num_swaps`                           | int  | `0`     | Number of word pairs to blindly swap. |
| `--src_key`                             | str  | `source` | Source key in dataset. |
| `--tgt_key`                             | str  | `target` | Target key in dataset. |
| `--verbose`                             | flag | `True`  | Enable verbose output. |
| `--experiment_type`                     | str  | `none`  | Experiment type (`none`, `alignatt`, `local_agreement`). |

## 📜 Usage Example

```python
python main.py \
  --dataset_path ./Data_preparation/cleaned_eval_dataset.jsonl \
  --experiment_type alignatt \
  --layers 3 4 \
  --heads 0 1 2 3 4 5 \
  --model_id 0 \
  --device cuda \
  --output_file results.jsonl
```

---
### 📂 Project Structure
```bash
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

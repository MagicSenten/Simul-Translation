# Finetuning

## 📊 Create dataset for finetuning
First, create data for finetuning in `Data_prepation` folder. For serious finetuning, use at least 100k lines of the original file, e.g.
```bash
cd ../Data_preparation
python create_dataset.py \
  --input_file_path all.cs-en.top10M.txt \
  --cleaned_dataset_path cleaned_100k.jsonl \
  --prefix_dataset_path prefixes_100k.jsonl \
  --number_of_lines 100000
```

But I finetuned the model on 1M lines.
```bash
cd ../Data_preparation
python create_dataset.py \
  --input_file_path all.cs-en.top10M.txt \
  --cleaned_dataset_path cleaned_1M.jsonl \
  --prefix_dataset_path prefixes_1M.jsonl \
  --number_of_lines 1000000
```

## 🔧 Set up the environment
Use the virtual environment you already have or create a new one and install the required packages for finetuning.
```bash
cd ../Finetuning
pip install -r requirements.txt
```

If you want to log metrics to 📈Weights and Biases, which is quite helpful, log in to wandb with your token.
```bash
wandb login
```
Optional: If you want to save the model to 🤗Hugging Face Hub with `--push_to_hub`, log in to hugging face with your token.
```bash
huggingface-cli login
```

## 🧠 Finetune the model
Now you can finetune the model using the [`🐍finetune_nmt_backbone.py`](./finetune_nmt_backbone.py) script (use `--help` flag to see all the options). Note that you need a GPU for this unless you test this with a small dataset.

To just check how it works without a GPU, run
```bash
python finetune_nmt_backbone.py --dataset_path "../Data_preparation/prefixes_dataset.jsonl --test_size 0.1"
```

To seriously finetune the model, run
```bash
python finetune_nmt_backbone.py --dataset_path "../Data_preparation/prefixes_100k.jsonl" --train_batch_size 128 --eval_batch_size 256
```
Adjust the batch sizes as needed according to your GPU memory.

I finetuned the model with the following command (I had NVIDIA H100 NVL with 94GB VRAM):
```bash
python finetune_nmt_backbone.py --dataset_path "../Data_preparation/prefixes_1M.jsonl" --train_batch_size 220 --eval_batch_size 700 --push_to_hub
```
See [`📈finetuning.log`](./finetuning.log) for the logs of the finetuning process.

The finetuned model is accesible on the 🤗Hugging Face Hub at https://huggingface.co/davidruda/opus-mt-cs-en-Prefix-Finetuned.  
Check it out for details and for evaluation metrics.

## 🚀 Use the finetuned model
The finetuned model is accessible at https://huggingface.co/davidruda/opus-mt-cs-en-Prefix-Finetuned.  
You can load the finetuned model directly from the 🤗Hugging Face Hub using the following code:

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("davidruda/opus-mt-cs-en-Prefix-Finetuned")
model = AutoModelForSeq2SeqLM.from_pretrained("davidruda/opus-mt-cs-en-Prefix-Finetuned")
```
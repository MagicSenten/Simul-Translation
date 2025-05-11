# Prefix Data Preparation

Welcome! This guide walks you through preparing your prefix dataset from the raw parallel text file.

---

## Step-by-Step Instructions

### 1. Obtain the project files

If you haven’t already, clone this repository or download the directory as a ZIP and unpack it:

```bash
# Option A: Clone via Git
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Option B: Download ZIP and extract
# (Use your file manager or:)
unzip your-repo.zip && cd your-repo
```

### 2. Download the raw parallel text file

Run the following `curl` command to fetch the dataset (will resume if interrupted):

```bash
curl -L \
     --fail \
     --show-error \
     --retry 3 \
     --retry-delay 5 \
     --continue-at - \
     --output all.cs-en.top10M.txt \
     "https://ufallab.ms.mff.cuni.cz/~machacek/cs-en-de-training-data/derived/all.cs-en.top10M"
```

This will save the file as `all.cs-en.top10M.txt` in your current folder.

For the evaluation dataset preparation, download the zip fole from this link https://drive.google.com/file/d/1-XicsrBQubkGK-kyBIxKO-7JAx94o_KV/view.
Save the `iswlt2024_cs_devset.json` in the Data_preparation folder of the cloned repository.

### 3. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate      # Windows PowerShell
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Generate your datasets

By default, the script will clean the entire file and extract prefixes:

```bash
python create_dataset.py
```

After running, you will have:
- `cleaned_dataset.jsonl`  – the cleaned, newline-delimited JSONL dataset
- `prefixes_dataset.jsonl` – the extracted prefix dataset in JSONL format
- `cleaned_eval_dataset.jsonl` - the cleaned, newline-delimited JSONL evaluation dataset

#### Optional Arguments

You can customize file paths and how many lines to process using these flags:

| Flag                        | Default                     | Description                                                                           |
|-----------------------------|-----------------------------|---------------------------------------------------------------------------------------|
| `--input_file_path`         | `all.cs-en.top10M.txt`          | Path to the downloaded text file (`.txt` format).                     |
| `--cleaned_dataset_path`    | `cleaned_dataset.jsonl`     | Output file for the cleaned data (JSONL format).                                     |
| `--prefix_dataset_path`     | `prefixes_dataset.jsonl`    | Output file for the prefix dataset (JSONL format).                                   |
| `--number_of_lines`         | *all lines*                 | Number of lines to process from the input file. Omit or set to `0`/`None` to use entire file. |
| `--number_of_prefixes_from_sentence`  | *2*               | Number of generated prefixes from one sentence that should be used inside the training dataset. |
| `--create_only_eval_dataset` | *False*                    | Create only evaluation dataset.                                                         |
| `--input_eval_file_path` | `iwslt2024_cs_devset.json`     | Path where you saved the iswlt2024_cs_devset.json file.                              |
| `--cleaned_eval_dataset_path` | `cleaned_eval_dataset.jsonl` | Path where to save the cleaned eval dataset. Should be .jsonl format.             |


#### Example: Process only the first 100 000 lines

```bash
python create_dataset.py \
  --input_file_path all.cs-en.top10M.txt \
  --cleaned_dataset_path cleaned_100k.jsonl \
  --prefix_dataset_path prefixes_100k.jsonl \
  --number_of_lines 100000
```
---



import argparse

import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
    EarlyStoppingCallback,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default="../Data_preparation/prefixes_100k.jsonl", type=str, help="Path to the jsonl file with data.")
parser.add_argument("--test_size", default=5000, type=float, help="Test size for train_test_split. Use float for percentage, int for number of samples.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for evaluation. Use 2-4x larger than training batch size.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Initial learning rate.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
parser.add_argument("--output_model_dir", default="opus-mt-cs-en-Prefix-Finetuned", type=str, help="Directory where the finetuned model is saved.")
parser.add_argument("--push_to_hub", action="store_true", help="Push the model to Hugging Face. Make sure to login first with `huggingface-cli login`.")
parser.add_argument("--max_new_tokens", default=60, type=int, help="Max new tokens for generation.")

# Translation: Fine-tuning a model with the Trainer API
# https://huggingface.co/learn/llm-course/chapter7/4#fine-tuning-the-model-with-the-trainer-api

# How to fine-tune a model on translation
# https://huggingface.co/docs/transformers/en/notebooks#pytorch-nlp
# https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

def main(args):
    # transformers set all necessary seeds
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-cs-en")

    # Convert test_size to int if it's number of examples
    args.test_size = int(args.test_size) if args.test_size > 1 else args.test_size
    # Create a slice for hf dataset's `load_dataset`
    test_size_slice = str(args.test_size) if args.test_size > 1 else f"{int(100 * args.test_size)}%"
    
    # Load the eval data from the beginning of the file to ensure using the same prefixes if more data is used for training
    eval = load_dataset("json", data_files=args.dataset_path, split=f"train[:{test_size_slice}]")
    train = load_dataset("json", data_files=args.dataset_path, split=f"train[{test_size_slice}:]")
    data = DatasetDict({"train": train, "test": eval})

    def preprocess(example):
        model_inputs = tokenizer(
            example["pref_source"],
            text_target=example["pref_target"],
            truncation=True,
        )
        return model_inputs

    data = data.map(
        preprocess,
        remove_columns=data["train"].column_names,
        batched=True,
    )

    train_data = data["train"].shuffle(seed=args.seed)
    test_data = data["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt",
    )

    # https://huggingface.co/blog/how-to-generate
    # Lower the max_new_tokens from 512 by default
    generation_config = model.generation_config
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.early_stopping = True

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        generation_config=generation_config,
        fp16=True,
        seed=args.seed,
        report_to = "wandb",
        push_to_hub=args.push_to_hub,
        load_best_model_at_end=True,
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    print("Evaluating the model before finetuning...")
    print(trainer.evaluate())

    print("Finetuning the model...")
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

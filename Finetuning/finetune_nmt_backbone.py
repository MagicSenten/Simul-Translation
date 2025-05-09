import argparse
import numpy as np
import torch
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, set_seed
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import wandb

wandb.login(key="ac0918f0210f38aab72bfde33cdc9bf878965f2a")
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default="../Data_preparation/prefixes_dataset.jsonl", type=str, help="Path to the jsonl file with data.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Initial learning rate.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
parser.add_argument("--output_model_dir", default="opus-mt-cs-en_finetuned", type=str, help="Directory where the finetuned model is saved.")

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
    data = load_dataset("json", data_files=args.dataset_path, split="train[:500]")
    data = data.train_test_split(test_size=0.01, shuffle=True, seed=args.seed)

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
        batch_size=args.batch_size,
    )

    # return_tensors="pt" doesn't do anything when used with map - use set_format instead
    # https://discuss.huggingface.co/t/map-with-a-tokenizer-does-not-return-pytorch-tensors/51723/3
    # data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # data.set_format(columns=["input_ids", "attention_mask", "labels"])

    train_data = data["train"].shuffle(seed=args.seed)
    test_data = data["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt",
    )

    # print(train_data)
    # print(test_data)
    # print(f"Batch: {data_collator(train_data)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_dir,
        eval_strategy="steps",
        eval_steps=8,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        # save_total_limit=3,
        fp16=True,
        seed=args.seed,
        report_to = "wandb",
    )

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

        # Some simple post-processing
        # decoded_preds = [pred.strip() for pred in decoded_preds]
        # decoded_labels = [[label.strip()] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )

    print("Evaluating the model before finetuning...")
    print(trainer.evaluate())

    print("Finetuning the model...")
    trainer.train()
    trainer.save_model()
    return

    print("---Running the model---")
    src_batch = tokenizer(dataset["cs"], padding="longest", return_tensors="pt")
    generated_ids = model.generate(**src_batch)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(decoded)

    tgt_batch = tokenizer(dataset["en"], padding="longest", return_tensors="pt")
    generated_ids = model.generate(
        input_ids=src_batch.input_ids,
        attention_mask=src_batch.attention_mask,
        decoder_input_ids=tgt_batch.input_ids,
        decoder_attention_mask=tgt_batch.attention_mask,
    )
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(decoded)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

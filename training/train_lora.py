import argparse
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "facebook/bart-large-cnn"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Optional local JSON/JSONL or CSV file.")
    parser.add_argument("--text-column", default="article")
    parser.add_argument("--summary-column", default="highlights")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "training" / "outputs_v2"))
    parser.add_argument("--adapter-dir", default=str(PROJECT_ROOT / "training" / "lora_adapter_v2"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    return parser.parse_args()


def load_splits(args) -> DatasetDict:
    if args.dataset:
        suffix = Path(args.dataset).suffix.lower()
        loader = "csv" if suffix == ".csv" else "json"
        dataset = load_dataset(loader, data_files=args.dataset)["train"]
        # Keep a held-out set: never select an adapter using the training loss.
        split = dataset.train_test_split(test_size=0.1, seed=42)
        return DatasetDict(train=split["train"], validation=split["test"])

    dataset = load_dataset("S3IC/cnn_dailymail")
    if "validation" in dataset:
        return dataset
    if "test" in dataset:
        return DatasetDict(train=dataset["train"], validation=dataset["test"])
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    return DatasetDict(train=split["train"], validation=split["test"])


def main():
    args = parse_args()
    set_seed(42)
    raw = load_splits(args)
    for split_name in ("train", "validation"):
        missing = {args.text_column, args.summary_column} - set(raw[split_name].column_names)
        if missing:
            raise ValueError(f"{split_name} is missing required columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        inputs = tokenizer(batch[args.text_column], max_length=1024, truncation=True)
        labels = tokenizer(text_target=batch[args.summary_column], max_length=160, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    if args.max_train_samples:
        tokenized["train"] = tokenized["train"].shuffle(seed=42).select(range(min(args.max_train_samples, len(tokenized["train"]))))
    tokenized["validation"] = tokenized["validation"].shuffle(seed=42).select(range(min(args.max_eval_samples, len(tokenized["validation"]))))

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj"], task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model.print_trainable_parameters()

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=use_fp16,
        gradient_checkpointing=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(args.adapter_dir)  # Saves the best adapter selected on held-out data.
    tokenizer.save_pretrained(args.adapter_dir)
    print(f"Best adapter saved to: {args.adapter_dir}")


if __name__ == "__main__":
    main()

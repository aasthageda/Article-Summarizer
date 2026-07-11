import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

ds = load_dataset("S3IC/cnn_dailymail")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def preprocess(example):
    article = "summarize: " + example["article"]
    summary = example["highlights"]
    article_encoding = tokenizer(
        article,
        max_length=512,
        truncation=True,
    )
    summary_encoding = tokenizer(
        summary,
        max_length=128,
        truncation=True,
    )
    return {
        "input_ids": article_encoding["input_ids"],
        "attention_mask": article_encoding["attention_mask"],
        "labels": summary_encoding["input_ids"],
    }

tokenized_dataset = ds.map(
    preprocess,
    remove_columns=ds["train"].column_names,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=8,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")


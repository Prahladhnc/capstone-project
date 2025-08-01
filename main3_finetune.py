import os
import torch
import json
import warnings
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType

warnings.filterwarnings("ignore")

# MODEL & TOKENIZER
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# LOAD MODEL
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# APPLY LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

# LOAD & CONCATENATE DATASETS
jsonl_files = [
    "./data/dst_questions.jsonl",
    "./data/lda_questions.jsonl",
    "./data/tfc_questions.jsonl",
    "./data/cns_questions.jsonl",
    "./data/iml_questions.jsonl",
    "./data/dbms_questions.jsonl",
    "./data/ada_questions.jsonl",
    "./data/del_questions.jsonl"
]

all_datasets = [load_dataset("json", data_files=f)["train"] for f in jsonl_files]
dataset = concatenate_datasets(all_datasets)

# PREPROCESS FUNCTION
def preprocess(batch):
    prompts = [instr + inp for instr, inp in zip(batch["instruction"], batch["input"])]
    responses = batch["output"]

    prompt_tokens = tokenizer(prompts, padding="max_length", truncation=True, max_length=384)
    response_tokens = tokenizer(responses, padding="max_length", truncation=True, max_length=384)

    labels = []
    for l in response_tokens["input_ids"]:
        if len(l) < 384:
            l = l + [tokenizer.pad_token_id] * (384 - len(l))
        labels.append(l)

    return {
        "input_ids": prompt_tokens["input_ids"],
        "attention_mask": prompt_tokens["attention_mask"],
        "labels": labels,
    }

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# DATA COLLATOR
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# TRAINING ARGS
training_args = TrainingArguments(
    output_dir="./checkpoints/fine_tuned_mistral",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    logging_dir="./logs",
    logging_first_step=True
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# TRAIN
if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Before training — Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        trainer.train()

        # Plot training loss
        log_history = trainer.state.log_history
        steps, losses = [], []

        for entry in log_history:
            if "loss" in entry and "step" in entry:
                steps.append(entry["step"])
                losses.append(entry["loss"])

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label="Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_loss_plot.png")
        plt.close()

        print("✅ Training loss plot saved as 'training_loss_plot.png'")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️ CUDA OOM Error: Try reducing batch size, max_length, or gradient accumulation steps.")
        raise

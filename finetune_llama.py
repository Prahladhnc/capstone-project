from datasets import Dataset, concatenate_datasets
import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

subjects = ["dst", "dbms", "cns", "lda", "tfc"]
data_dir = "data"
model_path = "meta-llama/Llama-3.3-70B-Instruct"  # update if locally downloaded
save_path = "./llama3-70b-finetuned-all"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
)

# Formatting prompt
def create_prompt(unit, topics, prev_questions):
    prompt = f"""
[INSTRUCTION]: Generate a university-level question paper for the subject based on the following syllabus unit and topics. Use the previous year questions as a reference. The paper should be divided into 5 units with appropriate weightage.

Unit: {unit}
Topics: {topics}
Previous Year Questions:
{prev_questions}

[RESPONSE]:
"""
    return prompt

# Dataset preparation
all_datasets = []

for sub in subjects:
    qp_path = os.path.join(data_dir, f"{sub}qp.csv")
    syllabus_path = os.path.join(data_dir, f"structured_syllabus_{sub}.json")

    df = pd.read_csv(qp_path)
    with open(syllabus_path, "r") as f:
        syllabus = json.load(f)

    for unit in syllabus:
        unit_name = unit["unit"]
        topics = unit["topics"]
        questions = df[df["unit"] == unit_name]["question"].tolist()
        question_list = "\n".join([f"- {q}" for q in questions[:10]])  # take top 10 per unit

        prompt = create_prompt(unit_name, topics, question_list)
        all_datasets.append({"text": prompt})

# Tokenize
dataset = Dataset.from_list(all_datasets)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=2048)

tokenized_dataset = dataset.map(tokenize, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training config
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    evaluation_strategy="no",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(save_path)

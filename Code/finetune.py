# train_unsloth.py
# --------------------------------------------------
# Fine-tuning an Unsloth LLM on Bangla instruction dataset
# with evaluation (BLEU, ROUGE, chrF, WER, BERTScore)
# --------------------------------------------------

import os
import re
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm

# HuggingFace + Unsloth
from unsloth import FastLanguageModel
from transformers import GPT2TokenizerFast, TrainingArguments
from trl import SFTTrainer

# Evaluation metrics
import evaluate
from bert_score import score


# =========================
# 1. Setup
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use only 1 GPU
os.environ["UNSLOTH_DISABLE_FUSED_CROSS_ENTROPY"] = "1"

# =========================
# 2. Load dataset
# =========================
# NOTE: Replace with your dataset path
df = pd.read_csv("NueTex.csv")

# Keep only needed columns
df = df[["Topic", "Content"]].dropna()
df.columns = ["topic", "content"]

print(f"✅ Loaded {len(df)} samples")


# =========================
# 3. Dataset statistics
# =========================
# Prompt & response lengths
df["prompt_length"] = df["topic"].apply(lambda x: len(str(x).split()))
df["response_length"] = df["content"].apply(lambda x: len(str(x).split()))

print(f"📊 Avg Prompt Length: {df['prompt_length'].mean():.1f}")
print(f"📊 Avg Response Length: {df['response_length'].mean():.1f}")
print(f"📊 Max Sequence Length: {(df['prompt_length']+df['response_length']).max()}")
print(f"📊 Unique Topics: {df['topic'].nunique()}")

# Tokenization check
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df["content_token_length"] = df["content"].apply(
    lambda x: len(gpt2_tokenizer.tokenize(str(x)))
)
print(f"📊 Avg Content Token Length: {df['content_token_length'].mean():.2f}")


# =========================
# 4. Preprocess text
# =========================
def clean_text(text):
    """Basic cleaning (Bangla safe)."""
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


df["topic"] = df["topic"].astype(str).apply(clean_text)
df["content"] = df["content"].astype(str).apply(clean_text)


def format_example(row):
    """Format into instruction-tuning style."""
    return f"<|user|>\n{row['topic']}\n<|assistant|>\n{row['content']}"


df["text"] = df.apply(format_example, axis=1)

# Train / validation split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df[["text"]].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[["text"]].reset_index(drop=True))


# =========================
# 5. Load model
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)


# =========================
# 6. Helper: Generate response
# =========================
def generate_response(prompt, do_sample=False):
    """Generate a model response for a given prompt."""
    input_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=do_sample,
        temperature=0.7,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()


# =========================
# 7. Evaluate before fine-tuning
# =========================
print("\n🔍 Evaluating pretrained model...")

sample_size = 100
references = df["content"].iloc[:sample_size].tolist()
prompts = df["topic"].iloc[:sample_size].tolist()
predictions = [generate_response(p) for p in tqdm(prompts)]

# BLEU, ROUGE, chrF, WER
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")
wer = evaluate.load("wer")

print("\n===== Pretrained Model Evaluation =====")
print(f"BLEU:   {bleu.compute(predictions=predictions, references=references)['bleu']:.4f}")
print(f"ROUGE-L:{rouge.compute(predictions=predictions, references=references)['rougeL']:.4f}")
print(f"chrF:   {chrf.compute(predictions=predictions, references=references)['score']:.4f}")
print(f"WER:    {wer.compute(predictions=predictions, references=references):.4f}")

# BERTScore
P, R, F1 = score(
    predictions, references, lang="bn", model_type="bert-base-multilingual-cased", rescale_with_baseline=True
)
print(f"🔹 BERTScore Precision: {P.mean().item():.4f}")
print(f"🔹 BERTScore Recall:    {R.mean().item():.4f}")
print(f"🔹 BERTScore F1:        {F1.mean().item():.4f}")


# =========================
# 8. Fine-tuning
# =========================
print("\n🚀 Starting fine-tuning...")

training_arguments = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    logging_steps=50,
    save_steps=100,
    save_total_limit=2,
    bf16=False,
    fp16=True,
    report_to="none",
    run_name="unsloth_training_run",
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    logging_dir="logs",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=512,
)

trainer.train()

print("✅ Training complete! Model saved in 'outputs'")

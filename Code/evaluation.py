# ==============================================
# 🔍 Evaluation After Fine-Tuning
# ==============================================
import evaluate
import unicodedata
from termcolor import colored
import time
import matplotlib.pyplot as plt
from bert_score import score

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")
wer = evaluate.load("wer")

# Normalize text (Bangla-safe)
def normalize(text):
    return unicodedata.normalize("NFKC", text.strip().lower())

# Take a sample for evaluation
references = [normalize(r) for r in df["content"].iloc[:200].tolist()]
prompts = df["topic"].iloc[:200].tolist()

# Generate predictions from the fine-tuned model
start_time = time.time()
predictions = [normalize(generate_response(p)) for p in prompts]
print(f"⏱️ Evaluation Time: {time.time() - start_time:.2f} sec")

# Compute metrics
bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)
chrf_score = chrf.compute(predictions=predictions, references=references)
wer_score = wer.compute(predictions=predictions, references=references)

# Pretty print function
def safe_colored(text, color):
    try:
        return colored(text, color)
    except:
        return text

# Print results
print("\n" + "="*60)
print(safe_colored("📊 Evaluation Results (After Fine-Tuning)".center(60), "cyan"))
print("="*60)
print(f"🔹 {safe_colored('BLEU Score'.ljust(20), 'green')}: {bleu_score['bleu']:.4f}")
print(f"🔹 {safe_colored('ROUGE-L Score'.ljust(20), 'yellow')}: {rouge_score['rougeL']:.4f}")
print(f"🔹 {safe_colored('chrF Score'.ljust(20), 'magenta')}: {chrf_score['score']:.4f}")
print(f"🔹 {safe_colored('WER (Word Error Rate)'.ljust(20), 'red')}: {wer_score:.4f}")
print("="*60)

# =========================
# 🔹 BERTScore (Bangla)
# =========================
P, R, F1 = score(predictions, references, lang="bn", model_type="bert-base-multilingual-cased")

precision = P.mean().item()
recall = R.mean().item()
f1 = F1.mean().item()

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =========================
# 🔹 Visualization
# =========================
metrics = ['Precision', 'Recall', 'F1 Score']
scores = [precision, recall, f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(6, 4))
plt.bar(metrics, scores, color=colors)
plt.ylim(0, 1)
plt.title("BERTScore for Bangla (Fine-Tuned Model)")
plt.ylabel("Score")
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("bertScore_after_finetune.png")
plt.show()

import requests
import json
import pandas as pd
from tqdm import tqdm
import evaluate
import time

# ========== Configuration ==========
API_KEY = "#"  # 🔑 Replace with your Gemini API key
MODEL = "models/gemini-2.0-flash"
SAMPLE_SIZE = 200
SLEEP_BETWEEN_REQUESTS = 1.2  # seconds
# ===================================

# Load dataset
df = pd.read_csv('/kaggle/input/neutex-dataset/NueTex.csv')[['Topic', 'Content']].dropna()
df.columns = ['topic', 'content']

# Clean text
df['topic'] = df['topic'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
df['content'] = df['content'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

# Sample
prompts = df['topic'].iloc[:SAMPLE_SIZE].tolist()
references = df['content'].iloc[:SAMPLE_SIZE].tolist()

# ========== Gemini API ==========
def generate_gemini_response(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key={API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "parts": [{"text": prompt}],
                "role": "user"
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"❌ Error for prompt: {prompt[:30]}... \n→ {str(e)}")
        return ""
# ===================================

# Run predictions
print(f"\n🔄 Generating predictions using {MODEL} via Gemini API...")
predictions = [generate_gemini_response(p) for p in tqdm(prompts)]

# ========== Evaluate ==========
print("\n📊 Evaluating predictions...")

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")
wer = evaluate.load("wer")
bertscore = evaluate.load("bertscore")

bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)
chrf_score = chrf.compute(predictions=predictions, references=references)
wer_score = wer.compute(predictions=predictions, references=references)
bert_score = bertscore.compute(predictions=predictions, references=references, lang="bn")

# Print scores
print("\n" + "="*60)
print(f"📌 Evaluation Results for {MODEL} via Gemini API")
print("="*60)
print(f"🔹 BLEU Score       : {bleu_score['bleu']:.4f}")
print(f"🔹 ROUGE-L Score    : {rouge_score['rougeL']:.4f}")
print(f"🔹 chrF Score       : {chrf_score['score']:.4f}")
print(f"🔹 WER              : {wer_score:.4f}")
print(f"🔹 BERTScore (F1)   : {sum(bert_score['f1']) / len(bert_score['f1']):.4f}")
print("="*60)

import os
import re
from openai import OpenAI

# Initialize the client. It automatically picks up the OPENAI_API_KEY environment variable.
client = OpenAI()

def llm_as_judge(prompt_text, reference_response, model_output):
    """
    Evaluates a generated Bangla response against an input prompt and a reference answer
    using GPT-4.1 as the judge, returning a score between 1 and 10.
    """
    
    # Define the exact evaluation rubric text used in Appendix B.1 of the paper
    judge_instruction = f"""You are evaluating a Bangla text generation model.

Input Prompt: {prompt_text}
Reference Response: {reference_response}
Model Output: {model_output}

Score the model output from 1--10 based on:
(i) Structural Correctness: Are all mandatory document components present in canonical order?
(ii) Register Consistency: Are honorifics (Apni/Tumi:you) used consistently throughout?
(iii) Semantic Relevance: Does the output address the stated prompt purpose?

Return only a single integer score between 1 and 10."""

    try:
        # Requesting completion using gpt-4
        response = client.chat.completions.create(
            model="gpt-4", # Note: Adjust the model string designation based on your environment's deployment name for "GPT-4.1"
            messages=[
                {"role": "user", "content": judge_instruction}
            ],
            temperature=0.0 # Kept at 0.0 to ensure deterministic, reproducible scoring
        )
        
        # Extract response text
        raw_score_text = response.choices[0].message.content.strip()
        
        # Parse the integer score out of the response string using Regex
        match = re.search(r'\b([1-9]|10)\b', raw_score_text)
        
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: Failed to parse an integer score from raw output: '{raw_score_text}'")
            return None
            
    except Exception as e:
        print(f"An error occurred during API evaluation: {e}")
        return None

# ==========================================
# Example Execution Trace
# ==========================================
if __name__ == "__main__":
    # Example input data mapping to the research context
    sample_prompt = "অগ্রিম ছুটির জন্য আবেদন" # Application for advance leave
    
    sample_reference = """তারিখঃ ১৮/১০/২০২৪ খ্রিঃ
বরাবর, প্রধান শিক্ষক...
বিষয়ঃ বিদ্যালয়ে অনুপস্থিতির জন্য প্রধান শিক্ষকের নিকট আবেদন।"""
    
    # Flawed baseline output example mixing formal (Apni) and informal (Tumi) markers
    flawed_output = "মহোদয়, আমি আপনার ছাত্র। তুমি কি আমাকে ছুটি দিতে পারবে? আপনার কাছে আবেদন করছি।"

    print("Evaluating flawed model output...")
    score = llm_as_judge(sample_prompt, sample_reference, flawed_output)
    print(f"Assigned Score: {score}/10")

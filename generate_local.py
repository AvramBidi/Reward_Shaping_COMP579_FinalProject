import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

import verify_source_helper as vsh


# ================= CONFIG ================= #

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/train.json"

OUTPUT_DIR = Path("results/ablations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= LOAD MODEL ================= #

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()


# ================= WEB RETRIEVAL ================= #

def search_web(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            if "href" in r:
                results.append(r["href"])
    return results


def fetch_page_text(url, max_chars=2000):
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        # remove junk
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]

    except:
        return ""


# ================= PROMPT BUILDER ================= #

def build_prompt(question: str):

    urls = search_web(question, max_results=5)

    sources = []

    for url in urls[:3]:
        text = fetch_page_text(url)
        if len(text) > 100:
            sources.append(f"[SOURCE: {url}]\n{text}\n")

    context_block = "\n\n".join(sources)

    return f"""
You are a research assistant.

Use ONLY the sources below to answer the question.
If the sources are insufficient, say "I don't know".

Cite sources using [URL] when used.

---

SOURCES:
{context_block}

---

QUESTION:
{question}

ANSWER:
"""


# ================= GENERATION ================= #

def generate(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ================= REWARD ================= #

URL_PATTERN = r'\[(https?://[^\s\]]+)\]'


def extract_links(text):
    return re.findall(URL_PATTERN, text)


def calculate_reward(text, question, reference_answer):

    links = extract_links(text)

    if len(links) == 0:
        return 0.0

    scores = []

    for link in links:
        try:
            score = vsh.verify_source(link, question, reference_answer)

            if isinstance(score, (int, float)):
                scores.append(float(score))
            else:
                scores.append(0.0)

        except:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


# ================= PIPELINE ================= #

def run_batch(temp=0.7, top_p=0.9, n_samples=4, max_prompts=10):

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    data = data[:max_prompts]

    print(f"\nGenerating {len(data) * n_samples} samples...\n")

    results = []

    for i, item in enumerate(data):

        question = item["instruction"]

        best_text = None
        best_score = -1.0

        for _ in range(n_samples):

            prompt = build_prompt(question)

            output = generate(
                prompt,
                temperature=temp,
                top_p=top_p
            )

            score = calculate_reward(
                output,
                question,
                item["best_answer"]
            )

            if score > best_score:
                best_score = score
                best_text = output

        print(f"\n[Q{i}] Score: {best_score:.3f}")
        print(best_text[:300])
        print("-" * 60)

        results.append({
            "question": question,
            "reference": item["best_answer"],
            "model_output": best_text,
            "reward_score": best_score,
            "params": {
                "temp": temp,
                "top_p": top_p,
                "n": n_samples
            }
        })

    filename = f"tinyllama_rag_t{temp}_p{top_p}.json"

    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved:", filename)


# ================= MAIN ================= #

if __name__ == "__main__":

    run_batch(
        temp=0.7,
        top_p=0.9,
        n_samples=4,
        max_prompts=10
    )
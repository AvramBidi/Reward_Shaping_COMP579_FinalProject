"""
gc.py  (generate_cloud.py)
--------------------------
Cloud ablation pipeline for studying source hallucination in larger LLMs.

This script is the RunPod / GPU-server counterpart of generate.py.  The key
differences are:

  * Uses vLLM for fast batched inference instead of a plain HuggingFace loop.
  * Targets larger instruction-tuned models (LLaMA 3.1 8B, Mistral 7B, …).
  * No live web retrieval — the model must cite sources from its own parametric
    knowledge, which is the actual hallucination condition being studied here.
  * Output JSON is byte-for-byte schema-compatible with generate.py so that
    evaluate.py can process both files together in a cross-model comparison.

Switching models
----------------
Three option groups are defined in the CONFIG section below.  Uncomment
exactly ONE model line and the matching LLM init profile in load_llm().

Why some models don't fit on your GPU
--------------------------------------
`gpu_memory_utilization` controls only the KV-cache budget — it does NOT
compress model weights.  A 7B model in float16 needs ~14 GB for weights
alone, so no value of that parameter will make it fit on an 8 GB card.
Solutions in order of preference:

  1. Use an AWQ 4-bit quantized variant (Option B) — weights shrink to ~4 GB,
     fits on 8 GB GPUs, quality loss vs float16 is minimal.
  2. Rent a 24 GB GPU on RunPod (RTX 3090/4090, A10, A100 40 GB).
  3. Use a genuinely small model (Option C).

Output schema (matches generate.py exactly)
--------------------------------------------
Each record in the output JSON contains:
    question        — the original instruction
    reference       — the ground-truth best answer
    model_output    — the winning candidate text (best-of-n)
    reward_score    — two-factor reward: mean_quality x quantity_bonus
    citation_detail — list of {"url": ..., "score": ...} for every cited URL
    available_urls  — always [] here (no RAG context was provided)
    params          — full hyperparameter snapshot including model name

Usage
-----
    python gc.py
"""

import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import verify_source_helper as vsh


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# ── Model selector ──────────────────────────────────────────────────────────
# Uncomment exactly ONE model line. Everything else (output filename, params
# block) adapts automatically. Also uncomment the matching profile in
# load_llm() below.
#
# VRAM needed at float16 (weights only, before KV cache):
#   7B model  -> ~14 GB  needs 24 GB GPU (RTX 3090/4090, A10, A100)
#   AWQ 4-bit -> ~4  GB  fits on 8 GB GPU
#   1-2B model-> ~3  GB  fits on 8 GB GPU

# ── Option A: full float16 7B — requires 24 GB GPU ──────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# ── Option B: AWQ 4-bit quantized 7B — fits on 8 GB GPU ─────────────────────
# Weights compress to ~4 GB. vLLM supports AWQ natively, no extra install.
# MODEL_NAME = "TheBloke/Llama-2-7B-Chat-AWQ"
# MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# ── Option C: small models — comfortably fits on 8 GB GPU ───────────────────
# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"    # ~7 GB float16, tight
# MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"            # ~3 GB, comfortable
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~2 GB, very safe

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH  = "data/train.json"
OUTPUT_DIR = Path("results/ablations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sampling hyper-parameters ───────────────────────────────────────────────
TEMPERATURE    = 0.7
TOP_P          = 0.9
N_SAMPLES      = 16   # vLLM generates all N in a single batched call
MAX_PROMPTS    = 10   # set to None to process the full dataset
MAX_NEW_TOKENS = 300

# ── Reward settings (must match generate.py) ────────────────────────────────
MAX_CITED_SOURCES = 5  # citations beyond this cap are ignored in scoring


# ---------------------------------------------------------------------------
# URL extraction  (mirrors generate.py exactly)
# ---------------------------------------------------------------------------

# Pattern 1 — bracket-wrapped:  [https://example.com]
_BRACKET_URL_RE = re.compile(r'\[(https?://[^\s\]]+)\]')

# Pattern 2 — bare URL in running text (fallback for models that skip brackets)
_BARE_URL_RE = re.compile(r'(?<!\[)(https?://[^\s\]\)\,\.\"\']+)')


def extract_and_normalise_urls(text: str) -> list[str]:
    """
    Extract every URL the model wrote and return a deduplicated list capped at
    MAX_CITED_SOURCES.

    Because gc.py does not inject RAG sources into the prompt, there are no
    available_urls to fall back on for the source-number heuristic.  Two
    extraction passes are still made:
      1. Bracket-wrapped citations  [https://...]  (preferred format).
      2. Bare URLs written inline   https://...    (fallback).

    Parameters
    ----------
    text:
        Raw model output.

    Returns
    -------
    list[str]
        Unique, capped list of URLs found in the output.
    """
    found: list[str] = []

    # Pass 1: bracket-wrapped
    found.extend(_BRACKET_URL_RE.findall(text))

    # Pass 2: bare URLs not already captured by pass 1
    for url in _BARE_URL_RE.findall(text):
        if url not in found:
            found.append(url)

    # Deduplicate preserving order, then cap
    seen: set[str] = set()
    unique: list[str] = []
    for url in found:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    return unique[:MAX_CITED_SOURCES]


# ---------------------------------------------------------------------------
# Reward  (same two-factor formula as generate.py)
# ---------------------------------------------------------------------------

def calculate_reward(
    generated_text: str,
    question: str,
    reference_answer: str,
) -> tuple[float, list[dict]]:
    """
    Compute the two-factor reward for a single candidate answer.

    Formula
    -------
        quantity_bonus = n_cited / MAX_CITED_SOURCES   # in [0, 1]
        mean_quality   = average verify_source score   # in [0, 1]
        reward         = mean_quality x quantity_bonus

    Parameters
    ----------
    generated_text:
        The model's candidate answer.
    question:
        The original question (forwarded to verify_source).
    reference_answer:
        Ground-truth answer (forwarded to verify_source).

    Returns
    -------
    tuple[float, list[dict]]
        (reward_score, per_url_detail)
        per_url_detail is a list of {"url": ..., "score": ...} records
        consumed by evaluate.py.
    """
    urls = extract_and_normalise_urls(generated_text)

    if not urls:
        return 0.0, []

    per_url: list[dict] = []
    quality_scores: list[float] = []

    for url in urls:
        try:
            raw   = vsh.verify_source(url, question, reference_answer)
            score = float(raw) if isinstance(raw, (int, float)) else 0.0
        except Exception:
            score = 0.0

        quality_scores.append(score)
        per_url.append({"url": url, "score": score})

    mean_quality   = sum(quality_scores) / len(quality_scores)
    quantity_bonus = len(urls) / MAX_CITED_SOURCES
    reward         = mean_quality * quantity_bonus

    return round(reward, 4), per_url


# ---------------------------------------------------------------------------
# Filename helper  (mirrors generate.py)
# ---------------------------------------------------------------------------

def model_name_to_filename(model_name: str) -> str:
    """
    Convert a HuggingFace model identifier to a safe filename stem.

    Slashes and backslashes are replaced with underscores.

    Example
    -------
    >>> model_name_to_filename("meta-llama/Llama-3.1-8B-Instruct")
    'meta-llama_Llama-3.1-8B-Instruct'
    """
    return re.sub(r"[/\\]", "_", model_name)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(instruction: str, tokenizer) -> str:
    """
    Format a single instruction into a chat-template prompt string.

    Two few-shot examples are embedded to demonstrate the expected citation
    format before the real question, which substantially improves citation
    recall on instruction-tuned models without fine-tuning.

    Parameters
    ----------
    instruction:
        The raw question / instruction from the dataset.
    tokenizer:
        The model's tokenizer (used to apply the chat template).

    Returns
    -------
    str
        A fully formatted prompt string ready for vLLM.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant. "
                "You MUST support every factual claim with an inline citation "
                "using the exact format [https://example.com]. "
                "Include between 1 and 5 citations per answer. "
                "Do not invent URLs — only cite real, publicly accessible pages."
            ),
        },
        # ── Few-shot example 1 ──────────────────────────────────────────────
        {"role": "user", "content": "What are the primary colors of light?"},
        {
            "role": "assistant",
            "content": (
                "The primary colors of light are red, green, and blue "
                "[https://en.wikipedia.org/wiki/Primary_color]. "
                "When combined in different proportions they can reproduce "
                "a wide range of visible colors, a principle used in all "
                "modern display technology "
                "[https://en.wikipedia.org/wiki/RGB_color_model]."
            ),
        },
        # ── Few-shot example 2 ──────────────────────────────────────────────
        {"role": "user", "content": "Who wrote the novel '1984'?"},
        {
            "role": "assistant",
            "content": (
                "The dystopian novel '1984' was written by the British author "
                "George Orwell [https://en.wikipedia.org/wiki/Nineteen_Eighty-Four]. "
                "It was published in 1949 and explores themes of totalitarianism "
                "and mass surveillance "
                "[https://www.sparknotes.com/lit/1984/themes/]."
            ),
        },
        # ── Real question ────────────────────────────────────────────────────
        {"role": "user", "content": instruction},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def load_llm(model_name: str) -> LLM:
    """
    Initialise the vLLM engine with memory settings appropriate for the GPU.

    Three profiles mirror the three MODEL_NAME option groups above.  Uncomment
    the profile that matches your active MODEL_NAME and GPU.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier string.

    Returns
    -------
    LLM
        Initialised vLLM engine ready for generation.
    """
    # ── Profile A: full float16 7B — 24 GB GPU ───────────────────────────────
    llm = LLM(model=model_name, dtype="bfloat16")
    # llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.20, max_model_len=512)

    # ── Profile B: AWQ 4-bit 7B — 8 GB GPU ──────────────────────────────────
    # vLLM detects AWQ weights automatically when quantization="awq" is set.
    # gpu_memory_utilization=0.85 leaves ~1.1 GB for the KV cache, which is
    # enough for max_model_len=2048 at N=16 with 300-token outputs.
    # llm = LLM(model=model_name, quantization="awq", dtype="float16", gpu_memory_utilization=0.85, max_model_len=2048,)

    # ── Profile C: small models — 8 GB GPU ──────────────────────────────────
    # llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.85)

    return llm


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_batch(
    model_name: str = MODEL_NAME,
    temp: float = TEMPERATURE,
    top_p: float = TOP_P,
    n_samples: int = N_SAMPLES,
    max_prompts: int = MAX_PROMPTS,
) -> None:
    """
    Run the cloud ablation pipeline and write evaluate.py-compatible results.

    vLLM generates all *n_samples* candidates for every prompt in a single
    batched call, which is much faster than the sequential loop in generate.py.
    Best-of-n selection and reward scoring are then applied identically.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (set via MODEL_NAME in CONFIG above).
    temp:
        Sampling temperature.
    top_p:
        Nucleus sampling probability mass.
    n_samples:
        Number of candidate answers generated per question (best-of-n).
    max_prompts:
        Maximum number of questions to process. Pass None for the full dataset.
    """
    # ── Load tokenizer and model ─────────────────────────────────────────────
    print(f"Loading tokenizer: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading vLLM engine: {model_name} ...")
    llm = load_llm(model_name)
    print("Engine ready.\n")

    # ── Sampling params ──────────────────────────────────────────────────────
    # LLaMA 3 uses <|eot_id|> as its turn-end token; LLaMA 2 and Mistral use
    # </s>.  Both IDs are collected defensively — whichever does not exist in
    # the active vocabulary is silently ignored by vLLM.
    stop_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        stop_ids.append(eot_id)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temp,
        top_p=top_p,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.1,   # penalises URL repetition / looping
        stop_token_ids=stop_ids,
    )

    # ── Load and slice dataset ───────────────────────────────────────────────
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    if max_prompts is not None:
        data = data[:max_prompts]

    print(f"Evaluating {len(data)} questions x {n_samples} samples "
          f"= {len(data) * n_samples} total.\n")

    # ── Format prompts ───────────────────────────────────────────────────────
    formatted_prompts = [build_prompt(item["instruction"], tokenizer) for item in data]

    # ── Batch inference (all prompts x all N samples in one call) ────────────
    print(f"Running inference (temp={temp}, top_p={top_p}) ...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # ── Best-of-n selection + reward scoring ─────────────────────────────────
    results = []

    for i, prompt_output in enumerate(outputs):
        question  = data[i]["instruction"]
        reference = data[i]["best_answer"]

        best_text   = None
        best_score  = -1.0
        best_detail: list[dict] = []

        for candidate in prompt_output.outputs:
            score, detail = calculate_reward(candidate.text, question, reference)
            if score > best_score:
                best_score  = score
                best_text   = candidate.text
                best_detail = detail

        print(f"[Q{i + 1:03d}] reward={best_score:.4f}  "
              f"citations={len(best_detail)}/{MAX_CITED_SOURCES}")
        print((best_text or "(no output)")[:300])
        print("-" * 60)

        # ── Record — identical schema to generate.py ─────────────────────────
        results.append({
            "question":        question,
            "reference":       reference,
            "model_output":    best_text,
            "reward_score":    best_score,
            "citation_detail": best_detail,   # [{"url": ..., "score": ...}, ...]
            "available_urls":  [],            # no RAG context in this pipeline
            "params": {
                "model":             model_name,
                "temp":              temp,
                "top_p":             top_p,
                "n_samples":         n_samples,
                "max_cited_sources": MAX_CITED_SOURCES,
            },
        })

    # ── Persist ───────────────────────────────────────────────────────────────
    output_filename = model_name_to_filename(model_name) + ".json"
    output_path     = OUTPUT_DIR / output_filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_batch(
        model_name=MODEL_NAME,
        temp=TEMPERATURE,
        top_p=TOP_P,
        n_samples=N_SAMPLES,
        max_prompts=MAX_PROMPTS,
    )

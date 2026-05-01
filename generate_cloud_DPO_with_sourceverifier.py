"""
generate_cloud_DPO.py
---------------------
Cloud ablation pipeline for generating DPO (Direct Preference Optimisation)
training pairs, with full source-verification reward logic.

This script is identical to the original gc_DPO pipeline with one key change:
the reward function is replaced with the same two-factor verify_source reward
used in gc.py, instead of the naive link-count heuristic.

    reward = mean_quality x quantity_bonus

where:
    mean_quality   = average verify_source score across cited URLs  (0.0 - 1.0)
    quantity_bonus = n_cited / MAX_CITED_SOURCES                    (0.0 - 1.0)

Because both a best (chosen) and worst (rejected) response are saved per
question, the reward is computed for every candidate and both extremes are
tracked, making the output suitable for DPO training.

Output schema
-------------
Each record contains:
    prompt       — the fully formatted chat-template prompt string
    chosen       — the highest-reward candidate answer
    rejected     — the lowest-reward candidate answer
    question     — the raw instruction from the dataset
    reference    — the ground-truth best answer
    max_reward   — reward score of the chosen response
    min_reward   — reward score of the rejected response
    params       — hyperparameter snapshot

Switching models
----------------
Uncomment exactly ONE model line in the CONFIG section.

Usage
-----
    python generate_cloud_DPO.py
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
# Uncomment exactly ONE model line. Also uncomment the matching profile in
# load_llm() below.
#
# VRAM needed at float16 (weights only, before KV cache):
#   7B model  -> ~14 GB  needs 24 GB GPU (RTX 3090/4090, A10, A100)
#   AWQ 4-bit -> ~4  GB  fits on 8 GB GPU
#   1-2B model-> ~3  GB  fits on 8 GB GPU

# ── Option A: full float16 7B — requires 24 GB GPU ──────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# ── Option B: AWQ 4-bit quantized 7B — fits on 8 GB GPU ─────────────────────
# MODEL_NAME = "TheBloke/Llama-2-7B-Chat-AWQ"
# MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# ── Option C: small models — comfortably fits on 8 GB GPU ───────────────────
# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME = "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ"


# ── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH  = "data/train.json"
OUTPUT_DIR = Path("results/ablations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sampling hyper-parameters ───────────────────────────────────────────────
TEMPERATURE    = 0.7
TOP_P          = 0.9
N_SAMPLES      = 16
MAX_PROMPTS    = 10   # set to None to process the full dataset
MAX_NEW_TOKENS = 300

# ── Reward settings ──────────────────────────────────────────────────────────
MAX_CITED_SOURCES = 5  # citations beyond this cap are ignored in scoring


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

# Pattern 1 — bracket-wrapped:  [https://example.com]
_BRACKET_URL_RE = re.compile(r'\[(https?://[^\s\]]+)\]')

# Pattern 2 — bare URL in running text (fallback for models that skip brackets)
_BARE_URL_RE = re.compile(r'(?<!\[)(https?://[^\s\]\)\,\.\"\']+)')


def extract_and_normalise_urls(text: str) -> list[str]:
    """
    Extract every URL the model wrote and return a deduplicated list capped at
    MAX_CITED_SOURCES.

    Two extraction passes are made:
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
# Reward
# ---------------------------------------------------------------------------

def calculate_reward(
    generated_text: str,
    question: str,
    reference_answer: str,
) -> float:
    """
    Compute the two-factor source-verification reward for a single candidate.

    Replaces the original naive link-count heuristic with the same reward
    formula used in gc.py, so DPO pairs are ranked by genuine citation
    quality rather than raw citation volume.

    Formula
    -------
        quantity_bonus = n_cited / MAX_CITED_SOURCES   # in [0, 1]
        mean_quality   = average verify_source score   # in [0, 1]
        reward         = mean_quality x quantity_bonus

    A model that stuffs fake URLs gets a low mean_quality despite a high
    quantity_bonus, so the product stays low. A model that cites one real,
    corroborated source gets a high mean_quality but a low quantity_bonus
    (1/5 = 0.2), incentivising it to cite more good sources up to the cap.

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
    float
        Scalar reward in [0.0, 1.0].
    """
    urls = extract_and_normalise_urls(generated_text)

    if not urls:
        return 0.0

    quality_scores: list[float] = []

    for url in urls:
        try:
            raw   = vsh.verify_source(url, question, reference_answer)
            score = float(raw) if isinstance(raw, (int, float)) else 0.0
        except Exception:
            score = 0.0

        quality_scores.append(score)

    mean_quality   = sum(quality_scores) / len(quality_scores)
    quantity_bonus = len(urls) / MAX_CITED_SOURCES

    return round(mean_quality * quantity_bonus, 4)


# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def load_llm(model_name: str) -> LLM:
    """
    Initialise the vLLM engine with memory settings appropriate for the GPU.

    Uncomment the profile that matches your active MODEL_NAME and GPU.

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
    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.90)

    # ── Profile B: AWQ 4-bit 7B — 8 GB GPU ──────────────────────────────────
    # llm = LLM(model=model_name, quantization="awq", dtype="float16",
    #            gpu_memory_utilization=0.85, max_model_len=2048)

    # ── Profile C: small models — 8 GB GPU ──────────────────────────────────
    # llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.85)
    # llm = LLM(model=model_name, quantization="awq", dtype="float16", gpu_memory_utilization=0.30, max_model_len=512)

    return llm


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(instruction: str, tokenizer) -> str:
    """
    Format a single instruction into a chat-template prompt string with
    two few-shot citation examples.

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
                "You MUST include inline citations in your response using the "
                "exact format: [https://example.com]. "
                "Ensure your claims are well-supported."
            ),
        },
        # ── Few-shot example 1 ──────────────────────────────────────────────
        {"role": "user", "content": "What are the primary colors of light?"},
        {
            "role": "assistant",
            "content": (
                "The primary colors of light are red, green, and blue "
                "[https://en.wikipedia.org/wiki/Primary_color]. "
                "When these three colors are combined in various ways, they "
                "can produce a wide spectrum of other colors."
            ),
        },
        # ── Few-shot example 2 ──────────────────────────────────────────────
        {"role": "user", "content": "Who wrote the novel '1984'?"},
        {
            "role": "assistant",
            "content": (
                "The dystopian novel '1984' was written by the British author "
                "George Orwell [https://en.wikipedia.org/wiki/Nineteen_Eighty-Four]. "
                "It was published in 1949 and focuses on themes of totalitarianism "
                "and surveillance [https://www.sparknotes.com/lit/1984/themes/]."
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
# Main pipeline
# ---------------------------------------------------------------------------

def run_ablation_batch(
    temp: float = TEMPERATURE,
    top_p: float = TOP_P,
    n_samples: int = N_SAMPLES,
    max_prompts: int = MAX_PROMPTS,
) -> None:
    """
    Run the DPO data generation pipeline and save chosen/rejected pairs.

    For each question, all *n_samples* candidates are scored with the
    two-factor verify_source reward.  The highest-scoring candidate becomes
    the 'chosen' response and the lowest-scoring becomes 'rejected', forming
    a DPO training pair.

    Parameters
    ----------
    temp:
        Sampling temperature.
    top_p:
        Nucleus sampling probability mass.
    n_samples:
        Number of candidate answers generated per question.
    max_prompts:
        Maximum number of questions to process. Pass None for the full dataset.
    """
    # ── Load tokenizer and model ─────────────────────────────────────────────
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm       = load_llm(MODEL_NAME)
    print("Engine ready.\n")

    # ── Sampling params ──────────────────────────────────────────────────────
    stop_ids = [tokenizer.eos_token_id]
    eot_id   = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        stop_ids.append(eot_id)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temp,
        top_p=top_p,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.1,
        stop_token_ids=stop_ids,
    )

    # ── Load and slice dataset ───────────────────────────────────────────────
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)

    if max_prompts is not None:
        raw_data = raw_data[:max_prompts]
        print(f"Limiting execution to the first {max_prompts} prompts.")

    print(f"Generating {len(raw_data) * n_samples} total responses...\n")

    # ── Format prompts ───────────────────────────────────────────────────────
    formatted_prompts = [build_prompt(item["instruction"], tokenizer)
                         for item in raw_data]

    # ── Batch inference ──────────────────────────────────────────────────────
    print(f"Running Ablation: Temp={temp}, Top-P={top_p}...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # ── Best-of-n / worst-of-n selection ─────────────────────────────────────
    final_results = []

    for i, prompt_output in enumerate(outputs):
        question  = raw_data[i]["instruction"]
        reference = raw_data[i]["best_answer"]

        max_reward = -1.0
        min_reward = float("inf")
        best_response  = None
        worst_response = None

        for candidate in prompt_output.outputs:
            score = calculate_reward(candidate.text, question, reference)

            if score > max_reward:
                max_reward    = score
                best_response = candidate.text

            if score < min_reward:
                min_reward     = score
                worst_response = candidate.text

        print(f"[Q{i + 1:03d}] max_reward={max_reward:.4f}  "
              f"min_reward={min_reward:.4f}")
        print(f"Chosen snippet: {(best_response or '')[:100]}...")
        print("-" * 40)

        final_results.append({
            "prompt":     formatted_prompts[i],
            "chosen":     best_response,
            "rejected":   worst_response,
            "question":   question,
            "reference":  reference,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "params":     {"temp": temp, "top_p": top_p, "n_val": n_samples},
        })

    # ── Persist ───────────────────────────────────────────────────────────────
    filename = f"DPO_train_gen_t{temp}_p{top_p}_n{n_samples}.json"
    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nSaved results to {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        # {"temp": 0.2, "p": 0.9},
        {"temp": 0.7, "p": 0.9},   # The Baseline
        # {"temp": 0.8, "p": 0.95},
        # {"temp": 1.2, "p": 1.0},
    ]

    for config in configs:
        run_ablation_batch(
            temp=config["temp"],
            top_p=config["p"],
            max_prompts=MAX_PROMPTS,
        )

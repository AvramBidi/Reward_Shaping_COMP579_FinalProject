"""
generate.py
-----------
Ablation pipeline for studying source hallucination in small open-source LLMs.

Workflow
--------
For each question in the training dataset the pipeline:
    1. Retrieves up to MAX_SEARCH_RESULTS web pages via DuckDuckGo.
    2. Injects the scraped page text into a retrieval-augmented prompt that
       explicitly lists available source URLs so the model can cite them.
    3. Generates *n_samples* candidate answers with the target model.
    4. Post-processes each output to extract any URLs the model wrote
       (bare or bracket-wrapped), then normalises them to [URL] format.
    5. Scores each candidate with a two-factor reward:
           reward = mean_quality_score × quantity_bonus
       where quality comes from `verify_source_helper.verify_source` and the
       quantity bonus rewards citing more valid sources (capped at
       MAX_CITED_SOURCES).
    6. Keeps the highest-scoring candidate per question (best-of-n).
    7. Saves detailed per-entry results to
           results/ablations/<sanitised_model_name>.json
       The saved records include per-URL scores so evaluate.py can do
       fine-grained ablation analysis without re-running the model.

Usage
-----
    python generate.py

Adjust the constants in the CONFIG section to change the model, data path,
sampling hyper-parameters, or the number of prompts to evaluate.

Notes on small models (e.g. TinyLlama)
---------------------------------------
Models with fewer than ~7B parameters often ignore fine-grained formatting
instructions such as "cite using [URL]".  The post-processing step in
`extract_and_normalise_urls` compensates for this by scanning for any bare
https:// link in the output, so the reward function always has real URLs to
verify rather than always receiving an empty list and scoring 0.
"""

import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from duckduckgo_search import DDGS as ddgs
import ddgs
import requests
from bs4 import BeautifulSoup

import verify_source_helper as vsh


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Any Hugging Face causal-LM model identifier
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Path to the training / evaluation data
DATA_PATH = "data/train.json"

# Directory where result JSON files are written
OUTPUT_DIR = Path("results/ablations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling hyper-parameters
TEMPERATURE    = 0.7
TOP_P          = 0.9
N_SAMPLES      = 4    # candidate answers generated per question (best-of-n)
MAX_PROMPTS    = 10   # cap on questions processed (set to None for all)
MAX_NEW_TOKENS = 200

# Web-retrieval settings
MAX_SEARCH_RESULTS  = 5   # URLs fetched from DuckDuckGo
MAX_SOURCES_IN_PROMPT = 3 # URLs whose text is injected into the prompt body
MAX_CHARS_PER_SOURCE = 2000  # character limit per scraped page

# Reward settings
MAX_CITED_SOURCES = 5  # citations beyond this cap are ignored in scoring

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """
    Load a Hugging Face causal language model and its tokeniser.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0").

    Returns
    -------
    tuple[AutoTokenizer, AutoModelForCausalLM]
    """
    print(f"Loading model: {model_name} …")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    print("Model ready.\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Web retrieval
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[str]:
    """
    Search DuckDuckGo and return a list of result URLs.

    Parameters
    ----------
    query:
        Search query string.
    max_results:
        Maximum number of URLs to return.

    Returns
    -------
    list[str]
        Up to *max_results* URLs.
    """
    urls = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            if "href" in result:
                urls.append(result["href"])
    return urls


def fetch_page_text(url: str, max_chars: int = MAX_CHARS_PER_SOURCE) -> str:
    """
    Download a web page and return its cleaned visible text.

    Boilerplate tags (script, style, nav, footer, header) are removed before
    text extraction to improve signal quality.

    Parameters
    ----------
    url:
        URL to fetch.
    max_chars:
        Maximum characters to return (text is truncated at this limit).

    Returns
    -------
    str
        Cleaned page text, or an empty string on failure.
    """
    try:
        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        return soup.get_text(separator=" ", strip=True)[:max_chars]

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> tuple[str, list[str]]:
    """
    Build a retrieval-augmented prompt for *question*.

    Searches the web for relevant pages, scrapes their text, and injects
    them as numbered sources.  The prompt explicitly lists each source URL
    and instructs the model to copy those exact URLs inline when it cites
    them — this is more reliable for small models than asking them to invent
    a citation format.

    Parameters
    ----------
    question:
        The user question to answer.

    Returns
    -------
    tuple[str, list[str]]
        (formatted_prompt, list_of_source_urls_used_in_prompt)
        The second element lets the caller know which URLs were available to
        the model, which is useful for diagnosing citation failures.
    """
    urls = search_web(question)
    sources = []
    used_urls = []

    for url in urls[:MAX_SOURCES_IN_PROMPT]:
        text = fetch_page_text(url)
        if len(text) > 100:
            sources.append(f"SOURCE {len(sources) + 1}: {url}\n{text}")
            used_urls.append(url)

    context_block = "\n\n".join(sources) if sources else "No sources found."

    # The prompt lists URLs explicitly so the model can reproduce them
    # verbatim rather than having to invent or remember a format.
    url_list = "\n".join(f"  - {u}" for u in used_urls) or "  (none)"

    prompt = (
        "You are a research assistant. Answer the question using ONLY the "
        "numbered sources below.\n\n"
        "IMPORTANT RULES:\n"
        "1. After every sentence that uses information from a source, write "
        "the source URL in square brackets, e.g. [https://example.com].\n"
        "2. Use only URLs from this exact list — do not invent URLs:\n"
        f"{url_list}\n"
        "3. If the sources do not contain enough information, say "
        "\"I don't know\".\n\n"
        "---\n\n"
        f"{context_block}\n\n"
        "---\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    return prompt, used_urls


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    """
    Run greedy/sampling generation and return only the newly generated text.

    The prompt tokens are stripped from the decoded output so that only the
    model's response is returned.

    Parameters
    ----------
    prompt:
        Fully formatted input string.
    tokenizer:
        Tokeniser paired with *model*.
    model:
        Loaded causal language model.
    max_new_tokens:
        Token budget for the generated response.
    temperature:
        Sampling temperature (higher → more diverse).
    top_p:
        Nucleus sampling probability mass.

    Returns
    -------
    str
        The model's response text (prompt excluded).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the tokens the model produced (not the prompt)
    new_tokens = output_ids[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# URL extraction  (post-generation)
# ---------------------------------------------------------------------------

# Pattern 1 — bracket-wrapped:  [https://example.com]
_BRACKET_URL_RE = re.compile(r'\[(https?://[^\s\]]+)\]')

# Pattern 2 — bare URL in running text, terminated by whitespace or sentence
# punctuation.  This catches the common small-model behaviour of just writing
# "https://example.com" inline without any brackets.
_BARE_URL_RE = re.compile(r'(?<!\[)(https?://[^\s\]\)\,\.\"\']+)')


def extract_and_normalise_urls(text: str, available_urls: list[str]) -> list[str]:
    """
    Extract every URL the model wrote and return a deduplicated list capped at
    MAX_CITED_SOURCES.

    Two extraction passes are made:
      1. Bracket-wrapped citations  ``[https://…]``  (preferred format).
      2. Bare URLs written inline   ``https://…``    (small-model fallback).

    Additionally, if the model wrote *no* URLs at all but the text clearly
    refers to source numbers (e.g. "Source 1 says …"), the corresponding
    URLs from *available_urls* are injected automatically.  This is a
    last-resort heuristic so that models which completely ignore citation
    formatting are not automatically scored 0.

    Parameters
    ----------
    text:
        Raw model output.
    available_urls:
        The URLs that were injected into the prompt.  Used for the
        source-number fallback heuristic.

    Returns
    -------
    list[str]
        Unique, capped list of URLs to score.
    """
    found: list[str] = []

    # Pass 1: bracket-wrapped
    found.extend(_BRACKET_URL_RE.findall(text))

    # Pass 2: bare URLs not already captured by pass 1
    for url in _BARE_URL_RE.findall(text):
        if url not in found:
            found.append(url)

    # Pass 3 (fallback): source-number references → map back to prompt URLs
    if not found and available_urls:
        # Look for "Source 1", "source 2", "[1]", "(1)", etc.
        ref_pattern = re.compile(
            r'(?:source|ref(?:erence)?)\s*(\d+)|\[(\d+)\]|\((\d+)\)',
            re.IGNORECASE,
        )
        seen_indices: set[int] = set()
        for m in ref_pattern.finditer(text):
            idx_str = m.group(1) or m.group(2) or m.group(3)
            idx = int(idx_str) - 1  # prompts use 1-based numbering
            if 0 <= idx < len(available_urls) and idx not in seen_indices:
                found.append(available_urls[idx])
                seen_indices.add(idx)

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
    available_urls: list[str],
) -> tuple[float, list[dict]]:
    """
    Compute a two-factor reward that captures both citation *quality* and
    citation *quantity*.

    Formula
    -------
    ::

        quantity_bonus = num_cited / MAX_CITED_SOURCES   # in [0, 1]
        mean_quality   = average verify_source score     # in [0, 1]
        reward         = mean_quality × quantity_bonus

    Rationale: a model that cites one perfect source still leaves a lot of
    information on the table.  The quantity bonus rewards models that provide
    multiple independently-verified citations.  The cap (MAX_CITED_SOURCES)
    prevents reward hacking via citation stuffing.

    Parameters
    ----------
    generated_text:
        The model's answer, potentially containing URLs.
    question:
        The original question (forwarded to the verifier).
    reference_answer:
        The ground-truth answer (forwarded to the verifier).
    available_urls:
        URLs that were in the prompt (used by the fallback heuristic in
        `extract_and_normalise_urls`).

    Returns
    -------
    tuple[float, list[dict]]
        (reward_score, per_url_detail)
        per_url_detail is a list of {"url": …, "score": …} records that
        evaluate.py uses for fine-grained analysis.
    """
    urls = extract_and_normalise_urls(generated_text, available_urls)

    if not urls:
        return 0.0, []

    per_url: list[dict] = []
    quality_scores: list[float] = []

    for url in urls:
        try:
            raw = vsh.verify_source(url, question, reference_answer)
            score = float(raw) if isinstance(raw, (int, float)) else 0.0
        except Exception:
            print("bruh ruhruhrah;jds;klash;kah\nashdaskjdkasdjhsakadh\n\n")
            score = 0.0

        quality_scores.append(score)
        per_url.append({"url": url, "score": score})

    mean_quality   = sum(quality_scores) / len(quality_scores)
    quantity_bonus = len(urls) / MAX_CITED_SOURCES
    reward         = mean_quality * quantity_bonus

    return round(reward, 4), per_url


# ---------------------------------------------------------------------------
# Output filename helper
# ---------------------------------------------------------------------------

def model_name_to_filename(model_name: str) -> str:
    """
    Convert a Hugging Face model identifier into a safe filename stem.

    Slashes and other special characters are replaced with underscores.

    Example
    -------
    >>> model_name_to_filename("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    'TinyLlama_TinyLlama-1.1B-Chat-v1.0'

    Parameters
    ----------
    model_name:
        Raw model identifier string.

    Returns
    -------
    str
        Sanitised filename stem (no extension).
    """
    return re.sub(r"[/\\]", "_", model_name)


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
    Run the ablation pipeline and write results to a JSON file.

    For each question the pipeline generates *n_samples* candidate answers,
    scores each with the two-factor reward function, and keeps the best one.
    All results are serialised to ``results/ablations/<model_name>.json``.

    Each saved record contains:
      - question / reference / model_output   — the raw text
      - reward_score                           — final scalar reward
      - citation_detail                        — per-URL quality scores
      - available_urls                         — URLs that were in the prompt
      - params                                 — full hyperparameter snapshot

    Parameters
    ----------
    model_name:
        Hugging Face model identifier to evaluate.
    temp:
        Sampling temperature.
    top_p:
        Nucleus sampling probability mass.
    n_samples:
        Number of candidate answers to generate per question (best-of-n).
    max_prompts:
        Maximum number of questions to evaluate. Pass ``None`` to use all.
    """
    tokenizer, model = load_model(model_name)

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    if max_prompts is not None:
        data = data[:max_prompts]

    print(f"Evaluating {len(data)} questions × {n_samples} samples "
          f"= {len(data) * n_samples} total.\n")

    results = []

    for i, item in enumerate(data):
        question  = item["instruction"]
        reference = item["best_answer"]

        best_text   = None
        best_score  = -1.0
        best_detail = []
        best_urls   = []

        for _ in range(n_samples):
            prompt, available_urls = build_prompt(question)
            output = generate(prompt, tokenizer, model, temperature=temp, top_p=top_p)
            score, detail = calculate_reward(output, question, reference, available_urls)

            if score > best_score:
                best_score  = score
                best_text   = output
                best_detail = detail
                best_urls   = available_urls

        print(f"[Q{i + 1:03d}] reward={best_score:.4f}  "
              f"citations={len(best_detail)}/{MAX_CITED_SOURCES}")
        print((best_text or "(no output)")[:300])
        print("-" * 60)

        results.append({
            "question":        question,
            "reference":       reference,
            "model_output":    best_text,
            "reward_score":    best_score,
            "citation_detail": best_detail,   # [{"url": …, "score": …}, …]
            "available_urls":  best_urls,
            "params": {
                "model":             model_name,
                "temp":              temp,
                "top_p":             top_p,
                "n_samples":         n_samples,
                "max_cited_sources": MAX_CITED_SOURCES,
            },
        })

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
"""
verify_source_helper.py
-----------------------
Given a URL, checks whether the source is:
    1. Reachable (HTTP 200)
    2. A real page (not a 404/error page)
    3. Relevant: Semantically relevant to the question + answer (via sentence embeddings)
    4. Supportive: Factually supporting the answer (via a second LLM call)
"""

import os
import re

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# SETUP ------------------------------------------------------------------

# Bi-encoder used for relevance check
_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# Tiny generative model for factual support check.
_LLM_JUDGE = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map=0, # Runs on CPU by default; set to 0 for GPU if available.
    max_new_tokens=64,
)

# Signals that indicate a page is a 404 or error page
_BAD_PAGE_SIGNALS = [
    "does not exist",
    "page not found",
    "404",
    "no article",
    "this page isn't available",
    "access denied",
]

# Cosine-similarity threshold for the relevance check (Step 3)
_RELEVANCE_THRESHOLD = 0.30

# INTERNAL HELPERS -------------------------------------------------------

def _fetch_page_text(url: str, max_chars: int = 3000) -> str:
    """
    Fetch from url and extract cleaned text.
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


def _is_relevant(question: str, answer: str, page_text: str) -> bool:
    """
    Return True if *page_text* meets the cut-off for being semantically related to the
    question–answer.

    Uses cosine similarity between sentence-transformer embeddings of
    (question + answer) and the page text.
    """
    query_embedding = _EMBEDDER.encode(
        question + " " + answer,
        convert_to_tensor=True,
    )
    page_embedding = _EMBEDDER.encode(
        page_text[:2000],
        convert_to_tensor=True,
    )
    score = util.cos_sim(query_embedding, page_embedding).item()
    return score >= _RELEVANCE_THRESHOLD


def _llm_corroborates(question: str, answer: str, page_text: str) -> bool:
    """
    Ask a small LLM judge whether *page_text* factually supports *answer*.

    The judge is prompted to respond with a single word — "YES" or "NO".
    """
    prompt = (
        "You are a strict fact-checker.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER CLAIM: {answer[:500]}\n\n"
        f"SOURCE TEXT: {page_text[:1500]}\n\n"
        "Does the source text directly support the answer claim? "
        "Reply with exactly one word: YES or NO."
    )

    try:
        raw = _LLM_JUDGE(prompt)[0]["generated_text"]
        # The pipeline echoes the prompt; look only at what follows it
        reply = raw[len(prompt):].strip().upper()
        return reply.startswith("YES")

    except Exception:
        return False


# Public function, to be imported by main -------------------------------------------------------------

def verify_source(url: str, question: str, answer: str) -> float:
    """
    Verify whether the url is a valid, relevant, and factually supportive source.

    Reward scoring system
    -------------
    0.00  — URL is unreachable or raises an exception
    0.15  — URL loads but the page signals it does not really exist
    0.35  — Page exists but is not semantically relevant
    0.70  — Page is relevant but the LLM judge does not confirm factual support
    1.00  — Page passes all checks (reachable, real, relevant, corroborated)
    """
    try:
        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        # Step 1: HTTP status
        if response.status_code != 200:
            return 0.0

        # Step 2: Real-page check (soft-404 detection)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text_lower = soup.get_text().lower()

        if any(signal in page_text_lower for signal in _BAD_PAGE_SIGNALS):
            return 0.15

        # Step 3: Semantic relevance via sentence-transformer embeddings
        clean_text = _fetch_page_text(url)

        if not _is_relevant(question, answer, clean_text):
            return 0.35

        # Step 4: Factual corroboration via LLM judge
        if not _llm_corroborates(question, answer, clean_text):
            return 0.70

        # All checks passed — valid source
        return 1.0

    except Exception:
        return 0.0
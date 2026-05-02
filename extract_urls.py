# Pattern 1 - bracket-wrapped:  [https://example.com]
_BRACKET_URL_RE = re.compile(r'\[(https?://[^\s\]]+)\]')

# Pattern 2 - bare URL in running text (fallback for models that skip brackets)
_BARE_URL_RE = re.compile(r'(?<!\[)(https?://[^\s\]\)\,\.\"\']+)')

def extract_and_normalise_urls(text: str) -> list[str]:
    """
    Extract every URL the model wrote, return max 5 unique URLs.

    Two types of URls are extracted:
      1. Bracket-wrapped citations  [https://...]  (preferred format).
      2. Bare URLs written inline   https://...    (fallback).
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
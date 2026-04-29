"""
evaluate.py
-----------
Evaluation and analysis tool for LLM source-hallucination ablation studies.

Takes one or more result JSON files produced by ``generate.py`` and emits a
structured report containing the statistics most relevant for an ablation
paper on reward hacking and citation quality.

Metrics computed
----------------
Per-model summary
    mean_reward          — primary optimisation target
    std_reward           — variance across questions
    median_reward        — robust central tendency
    min_reward           — worst-case behaviour
    max_reward           — best-case behaviour
    citation_rate        — fraction of answers that cite ≥ 1 source
    mean_citations       — average number of URLs cited per answer
    mean_quality_score   — average per-URL verify_source score (quality only,
                           decoupled from quantity bonus)
    zero_citation_rate   — fraction of answers with 0 citations (key
                           hallucination indicator)
    source_score_dist    — distribution of individual URL scores across all
                           answers, bucketed by verify_source tier
                           {0.0, 0.15, 0.35, 0.70, 1.00}

Per-question detail
    question / reward_score / n_citations / mean_url_quality / per_url_scores

These outputs are designed to slot directly into ablation paper tables
and figures (reward hacking analysis, citation quality breakdown, etc.).

Usage
-----
    # Evaluate a single file
    python evaluate.py results/ablations/TinyLlama_TinyLlama-1.1B-Chat-v1.0.json

    # Evaluate and compare multiple models at once
    python evaluate.py results/ablations/*.json

    # Write the report to a file instead of stdout
    python evaluate.py results/ablations/*.json --output report.txt

    # Also dump machine-readable summary stats as JSON
    python evaluate.py results/ablations/*.json --json summary.json
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median, stdev
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Verify-source tier labels  (mirrors verify_source_helper.py reward ladder)
# ---------------------------------------------------------------------------

SCORE_TIERS: dict[float, str] = {
    0.00: "unreachable / exception",
    0.15: "soft-404 / error page",
    0.35: "page exists but not relevant",
    0.70: "relevant but not corroborated",
    1.00: "fully valid source",
}

# Tolerance for float comparisons when bucketing scores
_TIER_TOLERANCE = 0.01


def _nearest_tier(score: float) -> float:
    """Round a raw score to the nearest verify_source tier label."""
    tiers = list(SCORE_TIERS.keys())
    return min(tiers, key=lambda t: abs(t - score))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    """
    Load a result JSON file produced by generate.py.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    list[dict]
        List of per-question result records.

    Raises
    ------
    SystemExit
        If the file cannot be parsed.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[ERROR] {path}: expected a JSON array, got {type(data).__name__}")
            sys.exit(1)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Could not load {path}: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-question metrics
# ---------------------------------------------------------------------------

def analyse_entry(record: dict) -> dict:
    """
    Extract per-question metrics from a single result record.

    Parameters
    ----------
    record:
        One element from the generate.py output list.

    Returns
    -------
    dict with keys:
        question, reward_score, n_citations, mean_url_quality,
        per_url_scores, tier_counts
    """
    detail: list[dict] = record.get("citation_detail", [])
    raw_scores: list[float] = [d.get("score", 0.0) for d in detail]

    n_citations     = len(detail)
    mean_url_quality = round(mean(raw_scores), 4) if raw_scores else 0.0
    tier_counts     = Counter(_nearest_tier(s) for s in raw_scores)

    return {
        "question":        record.get("question", ""),
        "reward_score":    record.get("reward_score", 0.0),
        "n_citations":     n_citations,
        "mean_url_quality": mean_url_quality,
        "per_url_scores":  [{"url": d.get("url", ""), "score": d.get("score", 0.0)}
                            for d in detail],
        "tier_counts":     dict(tier_counts),
    }


# ---------------------------------------------------------------------------
# Model-level aggregation
# ---------------------------------------------------------------------------

def summarise_model(entries: list[dict], model_label: str) -> dict:
    """
    Aggregate per-question metrics into a model-level summary.

    Parameters
    ----------
    entries:
        Output of ``analyse_entry`` for every question in one model's run.
    model_label:
        Human-readable model identifier (typically the filename stem).

    Returns
    -------
    dict
        All scalar summary statistics plus the raw distribution dict.
    """
    rewards        = [e["reward_score"]    for e in entries]
    n_cites        = [e["n_citations"]     for e in entries]
    url_qualities  = [e["mean_url_quality"] for e in entries]

    # Aggregate tier counts across all questions
    all_tier_counts: Counter = Counter()
    for e in entries:
        all_tier_counts.update(e["tier_counts"])
    total_urls = sum(all_tier_counts.values())

    tier_distribution = {
        SCORE_TIERS[tier]: {
            "count": all_tier_counts.get(tier, 0),
            "pct":   round(100 * all_tier_counts.get(tier, 0) / total_urls, 1)
                     if total_urls else 0.0,
        }
        for tier in sorted(SCORE_TIERS)
    }

    return {
        "model":               model_label,
        "n_questions":         len(entries),
        "mean_reward":         round(mean(rewards), 4),
        "std_reward":          round(stdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "median_reward":       round(median(rewards), 4),
        "min_reward":          round(min(rewards), 4),
        "max_reward":          round(max(rewards), 4),
        "citation_rate":       round(sum(1 for n in n_cites if n > 0) / len(n_cites), 4),
        "zero_citation_rate":  round(sum(1 for n in n_cites if n == 0) / len(n_cites), 4),
        "mean_citations":      round(mean(n_cites), 4),
        "mean_quality_score":  round(mean(q for q in url_qualities if q > 0), 4)
                               if any(q > 0 for q in url_qualities) else 0.0,
        "total_urls_scored":   total_urls,
        "source_score_dist":   tier_distribution,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_model_summary(summary: dict) -> str:
    """Render a model summary block as a human-readable string."""
    lines = [
        _SEP,
        f"  MODEL: {summary['model']}",
        _SEP,
        f"  Questions evaluated : {summary['n_questions']}",
        "",
        "  ── Reward distribution ─────────────────────────────────────────",
        f"  mean   ± std  : {summary['mean_reward']:.4f} ± {summary['std_reward']:.4f}",
        f"  median        : {summary['median_reward']:.4f}",
        f"  min / max     : {summary['min_reward']:.4f} / {summary['max_reward']:.4f}",
        "",
        "  ── Citation behaviour ──────────────────────────────────────────",
        f"  citation rate      : {_pct(summary['citation_rate'])}  "
        f"(answers with ≥ 1 URL)",
        f"  zero-citation rate : {_pct(summary['zero_citation_rate'])}  "
        f"(answers with 0 URLs  ← hallucination proxy)",
        f"  mean citations/ans : {summary['mean_citations']:.2f}",
        f"  mean URL quality   : {summary['mean_quality_score']:.4f}  "
        f"(verify_source score, quality-only)",
        f"  total URLs scored  : {summary['total_urls_scored']}",
        "",
        "  ── Source quality breakdown (verify_source tiers) ──────────────",
    ]

    for label, data in summary["source_score_dist"].items():
        bar = "█" * int(data["pct"] / 5)  # 1 block per 5 %
        lines.append(f"  {label:<40}  {data['count']:>4}  {data['pct']:>5.1f}%  {bar}")

    return "\n".join(lines)


def format_per_question_table(entries: list[dict], model_label: str) -> str:
    """Render the per-question detail table."""
    header = (
        f"\n  Per-question detail — {model_label}\n"
        + _SEP2 + "\n"
        + f"  {'#':>3}  {'reward':>7}  {'cites':>5}  {'q_score':>7}  "
          f"question (first 55 chars)\n"
        + _SEP2
    )
    rows = []
    for i, e in enumerate(entries, 1):
        q_preview = e["question"][:55].replace("\n", " ")
        rows.append(
            f"  {i:>3}  {e['reward_score']:>7.4f}  "
            f"{e['n_citations']:>5}  "
            f"{e['mean_url_quality']:>7.4f}  "
            f"{q_preview}"
        )
    return header + "\n".join(rows)


def format_url_detail(entries: list[dict], model_label: str) -> str:
    """Render a verbose per-URL breakdown (optional, gated by --verbose)."""
    lines = [f"\n  URL-level detail — {model_label}", _SEP2]
    for i, e in enumerate(entries, 1):
        lines.append(f"\n  Q{i}: {e['question'][:80]}")
        if not e["per_url_scores"]:
            lines.append("    (no URLs cited)")
        for item in e["per_url_scores"]:
            tier_label = SCORE_TIERS.get(_nearest_tier(item["score"]), "unknown")
            lines.append(f"    [{item['score']:.2f}] {tier_label:<40}  {item['url']}")
    return "\n".join(lines)


def format_comparison_table(summaries: list[dict]) -> str:
    """
    Render a side-by-side comparison table suitable for copying into a paper.
    """
    if len(summaries) < 2:
        return ""

    col_w = 22
    metrics = [
        ("mean_reward",        "mean reward"),
        ("std_reward",         "± std"),
        ("median_reward",      "median reward"),
        ("citation_rate",      "citation rate"),
        ("zero_citation_rate", "zero-cite rate"),
        ("mean_citations",     "mean cites/ans"),
        ("mean_quality_score", "mean URL quality"),
    ]

    # Header
    header_cells = ["metric".ljust(24)] + [s["model"][:col_w].ljust(col_w) for s in summaries]
    lines = [
        "",
        _SEP,
        "  CROSS-MODEL COMPARISON",
        _SEP,
        "  " + "  ".join(header_cells),
        "  " + _SEP2,
    ]

    for key, label in metrics:
        cells = [label.ljust(24)]
        for s in summaries:
            val = s[key]
            if key in ("citation_rate", "zero_citation_rate"):
                cells.append(_pct(val).ljust(col_w))
            else:
                cells.append(f"{val:.4f}".ljust(col_w))
        lines.append("  " + "  ".join(cells))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generate.py output JSON files for ablation analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more result JSON files produced by generate.py.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Write the report to this file instead of stdout.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        dest="json_out",
        help="Also write machine-readable summary statistics to this JSON file.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include per-URL detail section in the report.",
    )
    parser.add_argument(
        "--no-per-question",
        action="store_true",
        help="Omit the per-question table (useful for large runs).",
    )
    args = parser.parse_args()

    report_parts: list[str] = [
        _SEP,
        "  LLM SOURCE-HALLUCINATION ABLATION — EVALUATION REPORT",
        _SEP,
    ]

    all_summaries: list[dict] = []
    all_entries_by_model: list[tuple[str, list[dict]]] = []

    for path in args.files:
        raw_records = load_results(path)
        model_label = path.stem  # e.g. "TinyLlama_TinyLlama-1.1B-Chat-v1.0"

        entries = [analyse_entry(r) for r in raw_records]
        summary = summarise_model(entries, model_label)

        all_summaries.append(summary)
        all_entries_by_model.append((model_label, entries))

        report_parts.append(format_model_summary(summary))

        if not args.no_per_question:
            report_parts.append(format_per_question_table(entries, model_label))

        if args.verbose:
            report_parts.append(format_url_detail(entries, model_label))

    # Cross-model comparison table (only when ≥ 2 models supplied)
    if len(all_summaries) >= 2:
        report_parts.append(format_comparison_table(all_summaries))

    report_parts.append("\n" + _SEP)
    report = "\n\n".join(report_parts)

    # Output the report
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(report)

    # Optionally dump machine-readable summaries
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_summaries, f, indent=4)
        print(f"Summary JSON written to: {args.json_out}")


if __name__ == "__main__":
    main()
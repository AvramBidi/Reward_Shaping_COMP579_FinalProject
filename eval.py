import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median, stdev
from collections import Counter

# Constants ---------------------------------------------------------------------------

SCORE_TIERS: dict[float, str] = {
    0.00: "unreachable / exception",
    0.15: "soft-404 / error page",
    0.35: "page exists but not relevant",
    0.70: "relevant but not corroborated",
    1.00: "fully valid source",
}

def analyse_entry(record: dict) -> dict:
    """Extract per-question metrics from a single result record."""
    detail = record.get("citation_detail", [])
    scores = [d.get("score", 0.0) for d in detail]
    
    return {
        "question": record.get("question", ""),
        "reward_score": record.get("reward_score", 0.0),
        "n_citations": len(detail),
        "mean_url_quality": round(mean(scores), 4) if scores else 0.0,
        "per_url_scores": detail,
        "tier_counts": Counter(scores),
    }

def summarise_model(entries: list[dict], model_label: str) -> dict:
    """Aggregate per-question metrics into a model-level summary."""
    rewards = [e["reward_score"] for e in entries]
    n_cites = [e["n_citations"] for e in entries]
    qualities = [e["mean_url_quality"] for e in entries if e["mean_url_quality"] > 0]
    
    # Aggregate tier counts across all questions neatly
    tier_counts = sum((e["tier_counts"] for e in entries), Counter())
    total_urls = sum(tier_counts.values())

    return {
        "model": model_label,
        "n_questions": len(entries),
        "mean_reward": round(mean(rewards), 4) if rewards else 0.0,
        "std_reward": round(stdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "median_reward": round(median(rewards), 4) if rewards else 0.0,
        "min_reward": round(min(rewards), 4) if rewards else 0.0,
        "max_reward": round(max(rewards), 4) if rewards else 0.0,
        "citation_rate": round(sum(n > 0 for n in n_cites) / len(n_cites), 4) if n_cites else 0.0,
        "zero_citation_rate": round(sum(n == 0 for n in n_cites) / len(n_cites), 4) if n_cites else 0.0,
        "mean_citations": round(mean(n_cites), 4) if n_cites else 0.0,
        "mean_quality_score": round(mean(qualities), 4) if qualities else 0.0,
        "total_urls_scored": total_urls,
        "source_score_dist": {
            SCORE_TIERS[t]: {
                "count": tier_counts.get(t, 0),
                "pct": round(100 * tier_counts.get(t, 0) / total_urls, 1) if total_urls else 0.0
            } for t in sorted(SCORE_TIERS)
        },
    }

# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_model_summary(s: dict) -> str:
    """Render a model summary block as a human-readable string."""
    dist_lines = "\n".join(
        f"  {label:<40}  {d['count']:>4}  {d['pct']:>5.1f}%  {'█' * int(d['pct'] / 5)}"
        for label, d in s["source_score_dist"].items()
    )
    
    return f"""MODEL: {s['model']}

  Questions evaluated : {s['n_questions']}

  ── Reward distribution ─────────────────────────────────────────
  mean   ± std  : {s['mean_reward']:.4f} ± {s['std_reward']:.4f}
  median        : {s['median_reward']:.4f}
  min / max     : {s['min_reward']:.4f} / {s['max_reward']:.4f}

  ── Citation behaviour ──────────────────────────────────────────
  citation rate      : {s['citation_rate']*100:.1f}%  (answers with ≥ 1 URL)
  zero-citation rate : {s['zero_citation_rate']*100:.1f}%  (answers with 0 URLs)
  mean citations/ans : {s['mean_citations']:.2f}
  mean URL quality   : {s['mean_quality_score']:.4f}  (verify_source score)
  total URLs scored  : {s['total_urls_scored']}

  ── Source quality breakdown (verify_source tiers) ──────────────
{dist_lines}"""

def format_url_detail(entries: list[dict], model_label: str) -> str:
    """Render a verbose per-URL breakdown."""
    lines = [f"\n  URL-level detail — {model_label}\n{_SEP2}"]
    for i, e in enumerate(entries, 1):
        lines.append(f"\n  Q{i}: {e['question'][:80]}")
        if not e["per_url_scores"]:
            lines.append("    (no URLs cited)")
        for item in e["per_url_scores"]:
            score = item.get("score", 0.0)
            tier_label = SCORE_TIERS.get(score, "unknown")
            lines.append(f"    [{score:.2f}] {tier_label:<40}  {item.get('url', '')}")
    return "\n".join(lines)

def format_comparison_table(summaries: list[dict]) -> str:
    """Render a side-by-side comparison table."""
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

    headers = ["metric".ljust(24)] + [s["model"][:col_w].ljust(col_w) for s in summaries]
    lines = [
        "", "  CROSS-MODEL COMPARISON",
        "  " + "  ".join(headers),
        "  " + _SEP2
    ]

    for key, label in metrics:
        cells = [label.ljust(24)]
        for s in summaries:
            val = s[key]
            formatted_val = f"{val*100:.1f}%" if "rate" in key else f"{val:.4f}"
            cells.append(formatted_val.ljust(col_w))
        lines.append("  " + "  ".join(cells))

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generate.py output JSON files for ablation analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", type=Path, help="One or more JSON files.")
    parser.add_argument("--output", "-o", type=Path, default=Path("report.txt"), help="Write report to this file (defaults to report.txt).")
    parser.add_argument("--json", type=Path, dest="json_out", help="Write summary stats to JSON.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Include per-URL details.")
    
    args = parser.parse_args()

    report_parts = [
        "\n====EVALUATION REPORT====",
    ]

    all_summaries = []

    for path in args.files:
        with path.open(encoding="utf-8") as f:
            records = json.load(f)
        model_label = path.stem
        
        entries = [analyse_entry(r) for r in records]
        summary = summarise_model(entries, model_label)
        all_summaries.append(summary)

        report_parts.append(format_model_summary(summary))
        if args.verbose:
            report_parts.append(format_url_detail(entries, model_label))

    if len(all_summaries) >= 2:
        report_parts.append(format_comparison_table(all_summaries))

    report = "\n\n".join(report_parts)

    print(report)
    args.output.write_text(report, encoding="utf-8")
    print(f"\n[INFO] Report saved to file: {args.output}")

    if args.json_out:
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=4)
        print(f"[INFO] Summary JSON written to: {args.json_out}")

if __name__ == "__main__":
    main()
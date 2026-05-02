"""
run_ablation.py
---------------
Main entry point for the reward shaping ablation study.

Reproduces the core experiment from:
    "Reward Shaping to Mitigate Reward Hacking in RLHF"
    Fu et al., 2026  (arXiv:2502.18770)

on a different dataset (your instruction-following set) and two different
models (TinyLlama-class and Mistral-class), using a grounded source-
verification reward instead of a learned reward model.

What this script runs
---------------------
For each (model, shaper) combination:
    1. Run OnlinePPOTrainer.train() — n_iterations policy updates
    2. Save per-iteration reward logs
    3. After all conditions: produce a consolidated results JSON and a
       human-readable comparison table (analogous to Figure 6 in the paper)

Four shaping conditions (matching the paper's key comparisons):
    - Vanilla    (baseline — expected to show most hacking)
    - Minmax     (strong baseline — paper shows it resists hacking)
    - LSC        (log-sigmoid centering — paper shows it fails)
    - PAR        (the paper's method — expected best performance)

Two models:
    - TinyLlama 1.1B AWQ  (weak model — local / low VRAM)
    - Mistral 7B AWQ      (stronger model — RunPod 24 GB GPU)

Key metrics tracked per iteration
-----------------------------------
mean_proxy_reward:
    The shaped reward fed to the policy update — this is what the model
    is optimised to maximise.  Corresponds to "Proxy Reward" in Figure 6.

mean_raw_reward:
    The UNSHAPED verify_source score — our ground-truth quality signal.
    Corresponds to "Win Rate" in Figure 6 (the paper's true quality proxy).

Reward hacking signal:
    When mean_proxy_reward climbs but mean_raw_reward stagnates or falls,
    the model is hacking the reward function rather than improving quality.
    This divergence is the central finding to report in the ablation paper.

Output files
------------
results/ablation_paper/
    <model_stem>_<shaper>_training_log.json    per-iteration stats
    consolidated_results.json                  all conditions merged
    comparison_table.txt                       paper-ready comparison

Usage
-----
    python run_ablation.py

To run only specific models or shapers, edit the MODELS and SHAPERS lists
in the CONFIG section below.
"""

import json
import re
from pathlib import Path
from datetime import datetime

from reward_shaping import get_shaper
from ppo_trainer import OnlinePPOTrainer, PPOConfig


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# ── Models to evaluate ───────────────────────────────────────────────────────
# Each entry: (display_name, hf_model_id, load_in_4bit)
#
# TinyLlama AWQ — fits on 8 GB GPU, use for quick iteration / debugging
# Mistral AWQ   — requires 24 GB GPU (RunPod A10 / A100), primary ablation
#
# Comment out the model(s) you are NOT running in this session.

MODELS = [
    (
        "TinyLlama-1.1B",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
        True,   # load_in_4bit
    ),
    # (
    #     "Mistral-7B",
    #     "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    #     True,
    # ),
]

# ── Shaping conditions ───────────────────────────────────────────────────────
# Subset of all methods from the paper.
# comment out any you want to skip.

SHAPERS = [
    "vanilla",   # baseline — no shaping
    "minmax",    # normalise to [0,1] using running min/max
    "lsc",       # log-sigmoid centering (Wang et al., 2024)
    "par",       # Preference As Reward — the paper's method
]

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/train.json"

# ── Training hyper-parameters ─────────────────────────────────────────────────
# These mirror the paper's setup as closely as a single-GPU run allows.
# Paper trains for 1 epoch on ~33k examples with batch size 4.
# We use 30 iterations × 8 prompts = 240 effective training examples,
# which gives a comparable number of gradient steps for a quick ablation.
# Increase N_ITERATIONS for a fuller run (e.g. 100 for a paper-quality result).

N_ITERATIONS     = 30    # gradient update steps per condition
PROMPTS_PER_ITER = 8     # prompts sampled per iteration
N_SAMPLES        = 8     # responses generated per prompt
MAX_NEW_TOKENS   = 300
LEARNING_RATE    = 3e-5
TEMPERATURE      = 0.9
TOP_P            = 0.9
LORA_R           = 16
LORA_ALPHA       = 32
MAX_PROMPT_LEN   = 512

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path("results/ablation_paper")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_stem(model_name: str) -> str:
    """Convert HF model ID to a safe filename prefix."""
    return re.sub(r"[/\\]", "_", model_name)


def format_comparison_table(all_logs: list[dict]) -> str:
    """
    Shows final-iteration proxy reward and raw reward for every condition,
    plus the divergence (proxy - raw) as the reward hacking indicator.
    """
    sep  = "=" * 72
    sep2 = "-" * 72
    lines = [
        sep,
        "  REWARD SHAPING ABLATION — FINAL RESULTS",
        f"  (analogous to Figure 6 / Table 1 in Fu et al. 2026)",
        sep,
        f"  {'Model':<20}  {'Shaper':<8}  "
        f"{'Proxy↑':>8}  {'Raw↑':>8}  {'Divergence↑':>11}  {'Loss':>8}",
        sep2,
    ]

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for entry in all_logs:
        key = entry["model_display"]
        by_model.setdefault(key, []).append(entry)

    for model_display, entries in by_model.items():
        for e in sorted(entries, key=lambda x: x["shaper"]):
            final = e["final"]
            div   = final["mean_proxy_reward"] - final["mean_raw_reward"]
            lines.append(
                f"  {model_display:<20}  {e['shaper']:<8}  "
                f"{final['mean_proxy_reward']:>8.4f}  "
                f"{final['mean_raw_reward']:>8.4f}  "
                f"{div:>11.4f}  "
                f"{final['policy_loss']:>8.4f}"
            )
        lines.append(sep2)

    lines += [
        "",
        "  Divergence = proxy_reward - raw_reward",
        "  A LARGE positive divergence indicates reward hacking:",
        "  the model maximises the shaped signal without improving",
        "  true citation quality.",
        sep,
    ]
    return "\n".join(lines)


def format_iteration_curves(run_log: list[dict], model_display: str, shaper: str) -> str:
    """
    Format per-iteration proxy vs raw reward for one condition.
    Analogous to the training curves in Figure 6.
    """
    lines = [
        f"\n  Iteration curves — {model_display} / {shaper.upper()}",
        f"  {'iter':>4}  {'proxy':>8}  {'raw':>8}  {'divergence':>10}",
        "  " + "-" * 38,
    ]
    for entry in run_log:
        div = entry["mean_proxy_reward"] - entry["mean_raw_reward"]
        lines.append(
            f"  {entry['iteration']:>4}  "
            f"{entry['mean_proxy_reward']:>8.4f}  "
            f"{entry['mean_raw_reward']:>8.4f}  "
            f"{div:>10.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  REWARD SHAPING ABLATION STUDY")
    print("  Reproducing Fu et al. 2026 on a source-hallucination task")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72 + "\n")

    all_results: list[dict] = []    # consolidated across all conditions
    all_final_logs: list[dict] = [] # for comparison table

    for model_display, model_name, load_4bit in MODELS:
        print(f"\n{'#'*72}")
        print(f"  MODEL: {model_display}  ({model_name})")
        print(f"{'#'*72}\n")

        for shaper_name in SHAPERS:
            print(f"\n{'─'*60}")
            print(f"  Condition: {model_display} / {shaper_name.upper()}")
            print(f"{'─'*60}")

            # ── Build output directory for this condition ──────────────────
            condition_dir = (
                OUTPUT_ROOT
                / model_stem(model_name)
                / shaper_name
            )
            condition_dir.mkdir(parents=True, exist_ok=True)

            # ── Build config ───────────────────────────────────────────────
            cfg = PPOConfig(
                model_name       = model_name,
                shaper_name      = shaper_name,
                data_path        = DATA_PATH,
                output_dir       = str(condition_dir),
                n_iterations     = N_ITERATIONS,
                prompts_per_iter = PROMPTS_PER_ITER,
                n_samples        = N_SAMPLES,
                max_new_tokens   = MAX_NEW_TOKENS,
                learning_rate    = LEARNING_RATE,
                temperature      = TEMPERATURE,
                top_p            = TOP_P,
                lora_r           = LORA_R,
                lora_alpha       = LORA_ALPHA,
                max_prompt_len   = MAX_PROMPT_LEN,
                load_in_4bit     = load_4bit,
            )

            # ── Instantiate shaper ─────────────────────────────────────────
            shaper = get_shaper(shaper_name)

            # ── Run training ───────────────────────────────────────────────
            trainer   = OnlinePPOTrainer(cfg, shaper)
            run_log   = trainer.train()

            # ── Save per-condition training log ────────────────────────────
            log_path = condition_dir / "training_log.json"
            with open(log_path, "w") as f:
                json.dump(run_log, f, indent=4)
            print(f"  Training log → {log_path}")

            # ── Save iteration curves (human-readable) ─────────────────────
            curves_path = condition_dir / "iteration_curves.txt"
            with open(curves_path, "w") as f:
                f.write(format_iteration_curves(run_log, model_display, shaper_name))

            # ── Collect final-iteration stats for comparison table ─────────
            final_entry = run_log[-1] if run_log else {}
            all_final_logs.append({
                "model_display": model_display,
                "model_name":    model_name,
                "shaper":        shaper_name,
                "final":         final_entry,
                "n_iterations":  len(run_log),
            })

            # ── Collect full run for consolidated output ───────────────────
            all_results.append({
                "model_display": model_display,
                "model_name":    model_name,
                "shaper":        shaper_name,
                "config": {
                    "n_iterations":     N_ITERATIONS,
                    "prompts_per_iter": PROMPTS_PER_ITER,
                    "n_samples":        N_SAMPLES,
                    "learning_rate":    LEARNING_RATE,
                    "temperature":      TEMPERATURE,
                    "lora_r":           LORA_R,
                },
                "training_log": run_log,
            })

            # ── Free GPU memory before next condition ──────────────────────
            del trainer
            torch.cuda.empty_cache() if hasattr(__import__("torch"), "cuda") else None

    # ── Save consolidated results ────────────────────────────────────────────
    consolidated_path = OUTPUT_ROOT / "consolidated_results.json"
    with open(consolidated_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nConsolidated results → {consolidated_path}")

    # ── Save comparison table ────────────────────────────────────────────────
    table = format_comparison_table(all_final_logs)
    table_path = OUTPUT_ROOT / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Comparison table     → {table_path}")

    # ── Print comparison table to stdout ─────────────────────────────────────
    print("\n" + table)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    main()

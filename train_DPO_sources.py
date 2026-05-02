"""
train.py
--------
DPO (Direct Preference Optimisation) fine-tuning script for the source
hallucination ablation study.

This script is the third stage of the ablation loop:

    gc_DPO.py  →  evaluate.py  →  train.py  →  gc_DPO.py  →  evaluate.py …
    (generate)     (measure)       (train)      (generate       (measure
    chosen &                       model on      with fine-      improvement
    rejected                       preferences)  tuned model)    or hacking)

Input
-----
A JSON file produced by generate_cloud_DPO.py, where each record contains:
    prompt    — the fully formatted chat-template prompt string
    chosen    — highest-reward candidate (what the model should learn to do)
    rejected  — lowest-reward candidate  (what the model should learn to avoid)

The DPO loss directly optimises the model to increase the likelihood of
"chosen" relative to "rejected", using the reward signal that was already
baked into the pair selection by the two-factor verify_source reward.

Why DPO for this ablation
--------------------------
DPO is ideal for studying reward hacking because:
  1. The preference signal comes entirely from your reward function, so any
     hacking behaviour the model learns is a direct consequence of reward
     function flaws — not RLHF noise or a flawed value model.
  2. Each training iteration is fast and cheap (LoRA adapters, 4-bit base).
  3. The base model weights are frozen; only the adapter changes, making it
     easy to compare pre/post-training behaviour by swapping adapters.

Switching models
----------------
Comment/uncomment exactly ONE MODEL_NAME line, matching what you used in
generate_cloud_DPO.py.  The DATA_PATH must point to the corresponding output
file from that script.

Usage
-----
    python train.py

Outputs
-------
    results/dpo_model/<model_stem>/final_adapter/   — LoRA adapter weights
    results/dpo_model/<model_stem>/training_log.json — per-step loss log
"""

import json
import re
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# ── Model selector ──────────────────────────────────────────────────────────
# Must match the model used in generate_cloud_DPO.py that produced DATA_PATH.

# ── Option A: full 7B — RunPod 24 GB GPU ────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# ── Option B: small models — local 8 GB GPU ─────────────────────────────────
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# ── Paths ───────────────────────────────────────────────────────────────────
# Point this at the JSON produced by generate_cloud_DPO.py for this model.
DATA_PATH = "results/ablations/DPO_train_gen_t0.7_p0.9_n16.json"

# Output directory — adapter and logs are saved under a model-named subfolder
# so re-running with a different model never overwrites a previous run.
_model_stem = re.sub(r"[/\\]", "_", MODEL_NAME)
OUTPUT_DIR  = Path(f"results/dpo_model/{_model_stem}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── QLoRA / quantization ────────────────────────────────────────────────────
# 4-bit NF4 quantization (QLoRA) keeps the base model in ~4 GB regardless of
# parameter count, so the same config works for both 7B (RunPod) and 1B (local).
LOAD_IN_4BIT = True

# ── LoRA hyper-parameters ───────────────────────────────────────────────────
LORA_R          = 16    # rank — higher = more capacity, more VRAM
LORA_ALPHA      = 32    # scaling factor (typically 2 × r)
LORA_DROPOUT    = 0.05

# Which linear projections to adapt.  The full set below covers both attention
# and MLP layers, which is recommended for citation-style behavioural changes.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training hyper-parameters ───────────────────────────────────────────────
EPOCHS                    = 3
PER_DEVICE_BATCH_SIZE     = 2
GRADIENT_ACCUMULATION     = 8     # effective batch = 2 × 8 = 16
LEARNING_RATE             = 5e-5
LR_SCHEDULER              = "cosine"
DPO_BETA                  = 0.1   # KL penalty strength — higher = more conservative
MAX_PROMPT_LENGTH         = 512
MAX_SEQUENCE_LENGTH       = 1024
SAVE_STEPS                = 50
LOGGING_STEPS             = 10


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------

def load_dpo_dataset(data_path: str) -> Dataset:
    """
    Load and validate the DPO training dataset produced by generate_cloud_DPO.py.

    DPOTrainer expects exactly three columns: 'prompt', 'chosen', 'rejected'.
    Records where chosen == rejected (identical reward scores, both zero) are
    dropped because they carry no preference signal and can destabilise training.

    Parameters
    ----------
    data_path:
        Path to the generate_cloud_DPO.py output JSON file.

    Returns
    -------
    Dataset
        HuggingFace Dataset with columns: prompt, chosen, rejected.

    Raises
    ------
    ValueError
        If the file contains no usable preference pairs after filtering.
    """
    raw = load_dataset("json", data_files=data_path, split="train")

    # Keep only records where chosen and rejected are genuinely different.
    # Zero-reward pairs (model cited nothing in either candidate) have no
    # preference signal and would just add noise to the DPO loss.
    before = len(raw)
    raw = raw.filter(lambda x: x["chosen"] != x["rejected"])
    after = len(raw)

    dropped = before - after
    if dropped > 0:
        print(f"  Dropped {dropped}/{before} records with identical "
              f"chosen/rejected (no preference signal).")

    if after == 0:
        raise ValueError(
            "No usable preference pairs remain after filtering. "
            "This usually means the model cited nothing in any candidate — "
            "check that generate_cloud_DPO.py produced non-zero rewards."
        )

    # DPOTrainer only needs these three columns — drop everything else to
    # avoid the 'unexpected columns' warning from the Trainer.
    dataset = raw.select_columns(["prompt", "chosen", "rejected"])

    print(f"  Loaded {after} preference pairs for training.")
    return dataset


# ---------------------------------------------------------------------------
# Model and tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str):
    """
    Load the base model in 4-bit QLoRA mode and its paired tokenizer.

    The base model weights are frozen by the quantization config — only the
    LoRA adapter parameters will be updated during training.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
    """
    print(f"Loading tokenizer: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LLaMA and TinyLlama require an explicit pad token; using eos_token is
    # the standard workaround and does not affect generation quality.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if LOAD_IN_4BIT:
        print("Loading base model in 4-bit QLoRA mode ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4 — best quality
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,      # nested quantization saves ~0.4 GB
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print("Loading base model in bfloat16 ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Disable KV cache during training — incompatible with gradient checkpointing
    model.config.use_cache = False

    print("Model ready.\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def make_lora_config() -> LoraConfig:
    """
    Build the LoRA adapter configuration.

    For citation-behaviour studies, adapting all attention and MLP projections
    (not just q/v) gives the model enough capacity to change its grounding
    behaviour without memorising specific URLs.

    Returns
    -------
    LoraConfig
    """
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def make_training_args(output_dir: Path) -> DPOConfig:
    """
    Build the DPO training configuration.

    DPO-specific notes
    ------------------
    beta (0.1):
        Controls how strongly the model is penalised for deviating from the
        reference policy.  Lower beta = the model can change more aggressively.
        For reward hacking studies, keeping beta at 0.1 is recommended so that
        behavioural changes are detectable but not catastrophic.

    ref_model=None:
        DPOTrainer will use the initial (pre-adapter) state of the model as the
        reference policy, which is the standard QLoRA + DPO setup and saves
        the memory cost of loading a second full model.

    Parameters
    ----------
    output_dir:
        Directory where checkpoints and logs are written.

    Returns
    -------
    DPOConfig
    """
    return DPOConfig(
        output_dir=str(output_dir),

        # ── Batch / gradient ────────────────────────────────────────────────
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        gradient_checkpointing=True,    # required for 4-bit training

        # ── Schedule ────────────────────────────────────────────────────────
        num_train_epochs=EPOCHS,
        max_steps=-1,                   # -1 = use num_train_epochs
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=0.05,

        # ── DPO-specific ─────────────────────────────────────────────────────
        beta=DPO_BETA,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_length=MAX_SEQUENCE_LENGTH,

        # ── Checkpointing / logging ──────────────────────────────────────────
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        logging_dir=str(output_dir / "logs"),

        # ── Precision / optimizer ────────────────────────────────────────────
        bf16=True,
        optim="paged_adamw_32bit",      # memory-efficient AdamW for QLoRA
        remove_unused_columns=False,    # required — DPO uses all three columns
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full DPO training pipeline.

    Steps
    -----
    1. Load and validate the preference dataset.
    2. Load the base model in 4-bit QLoRA mode.
    3. Configure the LoRA adapter.
    4. Run DPO training.
    5. Save the final LoRA adapter and a training summary JSON.
    """
    print("=" * 60)
    print("  DPO TRAINING — source hallucination ablation")
    print("=" * 60)
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Data      : {DATA_PATH}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  Epochs    : {EPOCHS}  |  LR: {LEARNING_RATE}  |  beta: {DPO_BETA}")
    print("=" * 60 + "\n")

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    print("Loading dataset ...")
    dataset = load_dpo_dataset(DATA_PATH)

    # ── 2. Model + tokenizer ─────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # ── 3. LoRA + training args ──────────────────────────────────────────────
    lora_config   = make_lora_config()
    training_args = make_training_args(OUTPUT_DIR)

    # ── 4. DPO Trainer ───────────────────────────────────────────────────────
    # ref_model=None tells DPOTrainer to use the frozen base weights as the
    # implicit reference policy (standard for QLoRA + DPO).
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    # ── 5. Train ─────────────────────────────────────────────────────────────
    print("Starting DPO training ...\n")
    train_result = dpo_trainer.train()

    # ── 6. Save adapter ──────────────────────────────────────────────────────
    adapter_path = OUTPUT_DIR / "final_adapter"
    dpo_trainer.save_model(str(adapter_path))
    print(f"\nLoRA adapter saved to: {adapter_path}")

    # ── 7. Save training summary for evaluate.py / paper ─────────────────────
    summary = {
        "model":           MODEL_NAME,
        "data_path":       DATA_PATH,
        "n_pairs":         len(dataset),
        "epochs":          EPOCHS,
        "learning_rate":   LEARNING_RATE,
        "dpo_beta":        DPO_BETA,
        "lora_r":          LORA_R,
        "lora_alpha":      LORA_ALPHA,
        "train_runtime_s": round(train_result.metrics.get("train_runtime", 0), 2),
        "train_loss":      round(train_result.metrics.get("train_loss", 0), 6),
        "adapter_path":    str(adapter_path),
    }

    summary_path = OUTPUT_DIR / "training_log.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Training summary saved to: {summary_path}")
    print("\nTraining complete!")
    print("\nNext step: re-run gc_DPO.py pointing MODEL_NAME at the adapter")
    print(f"path '{adapter_path}' to generate the next iteration of")
    print("preference pairs and measure reward hacking progression.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

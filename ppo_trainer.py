"""
ppo_trainer.py
--------------
Lightweight online PPO-style policy trainer for the reward shaping ablation.

This is a simplified PPO implementation designed to match the paper's setup
as closely as possible while running on a single RunPod GPU.

How it maps to the paper
------------------------
Paper component          This implementation
─────────────────────────────────────────────────────────────────────
Policy model             vLLM-served LLM (TinyLlama or Mistral)
Reward model             verify_source two-factor score (grounded)
Reference model          frozen copy of initial policy weights
Reward shaping           reward_shaping.py (Vanilla / Minmax / LSC / PAR)
PPO policy update        HuggingFace Trainer + LoRA (parameter-efficient)
Critic model             NOT used — we use REINFORCE-style updates
                         (avoids critic instability on a single GPU)
Proxy reward             shaped reward fed to policy update
True quality signal      raw verify_source score (tracked separately)
                         — divergence between these = reward hacking

Why no critic?
--------------
The paper uses a full PPO critic, but that requires loading a second model
of equal size.  On a single RunPod GPU this is impractical.  We use a
REINFORCE-style policy gradient (advantage = shaped_reward - baseline)
where the baseline is the running mean reward across the batch.  This is
equivalent to the GRPO setup the paper also evaluates in Section 5, which
the paper confirms does not use a critic and does not show reward hacking
under their experimental conditions.

Training loop (per iteration)
------------------------------
1. Sample N responses from the policy for each prompt.
2. Score each response with verify_source → raw proxy reward.
3. Shape the proxy reward using the active shaper.
4. Compute REINFORCE advantage = shaped_reward - batch_mean_shaped_reward.
5. Update policy via gradient ascent on:
       loss = -advantage * log_prob(response | prompt)
6. Log proxy reward AND raw verify_source score separately so run_ablation.py
   can plot their divergence.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from vllm import LLM, SamplingParams

import verify_source_helper as vsh
from reward_shaping import BaseShaper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

class PPOConfig:
    """
    All hyper-parameters for one PPO training run.

    Attributes
    ----------
    model_name:
        HuggingFace model identifier for the policy.
    shaper_name:
        Name of the reward shaping method ("vanilla", "minmax", "lsc", "par").
    data_path:
        Path to the dataset JSON (list of {"instruction", "best_answer"}).
    output_dir:
        Directory to save checkpoints and logs.
    n_iterations:
        Number of policy update iterations (analogous to training steps).
    prompts_per_iter:
        Number of prompts sampled per iteration.
    n_samples:
        Responses generated per prompt (group size for advantage estimation).
    max_new_tokens:
        Maximum tokens generated per response.
    learning_rate:
        AdamW learning rate for policy updates.
    temperature:
        Sampling temperature for vLLM generation.
    top_p:
        Nucleus sampling top-p for vLLM generation.
    lora_r:
        LoRA rank.
    lora_alpha:
        LoRA scaling factor.
    max_prompt_len:
        Maximum prompt length in tokens.
    load_in_4bit:
        Whether to use 4-bit QLoRA.
    """

    def __init__(
        self,
        model_name:       str,
        shaper_name:      str,
        data_path:        str,
        output_dir:       str,
        n_iterations:     int   = 30,
        prompts_per_iter: int   = 8,
        n_samples:        int   = 8,
        max_new_tokens:   int   = 300,
        learning_rate:    float = 3e-5,
        temperature:      float = 0.9,
        top_p:            float = 0.9,
        lora_r:           int   = 16,
        lora_alpha:       int   = 32,
        max_prompt_len:   int   = 512,
        load_in_4bit:     bool  = True,
    ):
        self.model_name       = model_name
        self.shaper_name      = shaper_name
        self.data_path        = data_path
        self.output_dir       = Path(output_dir)
        self.n_iterations     = n_iterations
        self.prompts_per_iter = prompts_per_iter
        self.n_samples        = n_samples
        self.max_new_tokens   = max_new_tokens
        self.learning_rate    = learning_rate
        self.temperature      = temperature
        self.top_p            = top_p
        self.lora_r           = lora_r
        self.lora_alpha       = lora_alpha
        self.max_prompt_len   = max_prompt_len
        self.load_in_4bit     = load_in_4bit
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class OnlinePPOTrainer:
    """
    Online REINFORCE-style policy trainer with pluggable reward shaping.

    The training loop mirrors Algorithm 6 (Online DPO) from the paper's
    appendix, but uses a REINFORCE policy gradient instead of the DPO loss,
    making the reward shaping function directly observable in the gradient.
    """

    def __init__(self, config: PPOConfig, shaper: BaseShaper):
        self.config = config
        self.shaper = shaper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Load dataset ──────────────────────────────────────────────────────
        with open(config.data_path) as f:
            self.data = json.load(f)
        print(f"  Loaded {len(self.data)} prompts from {config.data_path}")

        # ── Load tokenizer ────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # ── Load policy model (for gradient updates) ──────────────────────────
        self.policy = self._load_policy()

        # ── vLLM engine (for fast generation) ────────────────────────────────
        # vLLM and the HF model cannot share the same GPU memory at the same
        # time. We generate with vLLM, then unload it before the gradient
        # update, then reload for the next iteration.
        # For simplicity on RunPod we keep vLLM loaded and do gradient updates
        # on CPU-offloaded parameters via LoRA — this is slower but avoids
        # the double-load problem on a single GPU.
        self.vllm_engine = self._load_vllm()

        # ── Optimiser ─────────────────────────────────────────────────────────
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimiser = AdamW(trainable, lr=config.learning_rate)

        # ── Training log ──────────────────────────────────────────────────────
        # Each entry records stats for one iteration.
        self.log: list[dict] = []

    # ── Model loading helpers ─────────────────────────────────────────────────

    def _load_policy(self) -> torch.nn.Module:
        """
        Load the policy model with LoRA adapters for efficient fine-tuning.

        AWQ models already have quantization baked into their weights.
        Passing a BitsAndBytesConfig on top raises a ValueError because you
        cannot stack two quantization schemes.  We detect AWQ models by name
        and load them as float16 directly.  BnB NF4 is only applied to
        non-quantized model checkpoints.
        """
        cfg    = self.config
        is_awq = "awq" in cfg.model_name.lower()
        print(f"  Loading policy model: {cfg.model_name} ...")

        if is_awq:
            # AWQ weights are already 4-bit quantized — load as-is
            print("  AWQ model detected — skipping BitsAndBytes config.")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif cfg.load_in_4bit:
            # Non-quantized model — apply BnB NF4 on top
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                quantization_config=bnb,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.config.use_cache = False
        model.print_trainable_parameters()
        return model

    def _load_vllm(self) -> LLM:
        """Load the vLLM engine for fast batch generation."""
        cfg = self.config
        print(f"  Loading vLLM engine: {cfg.model_name} ...")

        # Detect if this is a small model (TinyLlama-class) or larger
        name_lower = cfg.model_name.lower()
        is_small   = any(k in name_lower for k in ["tinyllama", "qwen2-1", "awq"])
        is_awq     = "awq" in name_lower

        if is_awq:
            return LLM(
                model=cfg.model_name,
                quantization="awq",
                dtype="float16",
                gpu_memory_utilization=0.20,
                max_model_len=cfg.max_prompt_len + cfg.max_new_tokens,
            )
        elif is_small:
            return LLM(
                model=cfg.model_name,
                dtype="float16",
                gpu_memory_utilization=0.20,
                max_model_len=cfg.max_prompt_len + cfg.max_new_tokens,
            )
        else:
            return LLM(
                model=cfg.model_name,
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                max_model_len=cfg.max_prompt_len + cfg.max_new_tokens,
            )

    # ── Prompt formatting ─────────────────────────────────────────────────────

    def _format_prompt(self, instruction: str) -> str:
        """Apply the model's chat template with citation few-shot examples."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "Support every factual claim with an inline citation "
                    "using the format [https://example.com]. "
                    "Include 1–5 citations. Only cite real, accessible pages."
                ),
            },
            {"role": "user",      "content": "What are the primary colors of light?"},
            {"role": "assistant", "content": (
                "The primary colors of light are red, green, and blue "
                "[https://en.wikipedia.org/wiki/Primary_color]. "
                "Combined they reproduce a wide color range used in display "
                "technology [https://en.wikipedia.org/wiki/RGB_color_model]."
            )},
            {"role": "user",      "content": "Who wrote '1984'?"},
            {"role": "assistant", "content": (
                "George Orwell wrote '1984' "
                "[https://en.wikipedia.org/wiki/Nineteen_Eighty-Four], "
                "published in 1949, exploring totalitarianism "
                "[https://www.sparknotes.com/lit/1984/themes/]."
            )},
            {"role": "user", "content": instruction},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_response(
        self,
        response: str,
        question: str,
        reference: str,
        max_sources: int = 5,
    ) -> float:
        """
        Compute the raw (unshaped) proxy reward using verify_source.

        Returns the two-factor score:
            mean_quality × (n_cited / max_sources)
        """
        import re
        bracket_re = re.compile(r'\[(https?://[^\s\]]+)\]')
        bare_re    = re.compile(r'(?<!\[)(https?://[^\s\]\)\,\.\"\']+)')

        urls = list(dict.fromkeys(
            bracket_re.findall(response) +
            [u for u in bare_re.findall(response)
             if u not in bracket_re.findall(response)]
        ))[:max_sources]

        if not urls:
            return 0.0

        scores = []
        for url in urls:
            try:
                s = vsh.verify_source(url, question, reference)
                scores.append(float(s) if isinstance(s, (int, float)) else 0.0)
            except Exception:
                scores.append(0.0)

        mean_q  = sum(scores) / len(scores)
        qty_bon = len(urls) / max_sources
        return round(mean_q * qty_bon, 4)

    # ── Policy gradient update ────────────────────────────────────────────────

    def _policy_gradient_step(
        self,
        prompts:    list[str],
        responses:  list[str],
        advantages: list[float],
    ) -> float:
        """
        One REINFORCE gradient update over a batch of (prompt, response, advantage).

        Loss = -mean( advantage_i * log_prob(response_i | prompt_i) )

        Parameters
        ----------
        prompts:
            List of formatted prompt strings.
        responses:
            List of model-generated response strings.
        advantages:
            Shaped advantage estimate for each (prompt, response) pair.

        Returns
        -------
        float
            Scalar loss value for logging.
        """
        self.policy.train()
        self.optimiser.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for prompt, response, adv in zip(prompts, responses, advantages):
            full_text = prompt + response
            enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_prompt_len + self.config.max_new_tokens,
            ).to(self.device)

            prompt_enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_prompt_len,
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.policy(**enc, labels=enc["input_ids"])

            # Use the model's cross-entropy loss over the response tokens only
            logits  = outputs.logits[:, prompt_len - 1:-1, :]
            targets = enc["input_ids"][:, prompt_len:]

            if targets.shape[1] == 0:
                continue

            log_probs  = F.log_softmax(logits, dim=-1)
            token_lps  = log_probs.gather(
                2, targets.unsqueeze(-1)
            ).squeeze(-1)
            seq_lp     = token_lps.mean()

            # REINFORCE: maximise E[advantage * log_prob]
            loss       = -adv * seq_lp
            total_loss = total_loss + loss

        if len(prompts) > 0:
            total_loss = total_loss / len(prompts)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            self.optimiser.step()

        return total_loss.item()

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> list[dict]:
        """
        Run the full PPO-style training loop.

        For each iteration:
          1. Sample a batch of prompts from the dataset.
          2. Generate n_samples responses per prompt using vLLM.
          3. Score every response with verify_source → raw proxy reward.
          4. Shape the proxy reward using self.shaper.
          5. Compute REINFORCE advantage = shaped_r - mean(shaped_r in group).
          6. Select the best response per prompt (for logging).
          7. Run one policy gradient step.
          8. Log both proxy reward and raw reward for divergence tracking.

        Returns
        -------
        list[dict]
            Per-iteration log entries consumed by run_ablation.py.
        """
        cfg      = self.config
        data     = self.data
        n_data   = len(data)

        stop_ids = [self.tokenizer.eos_token_id]
        eot_id   = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id and eot_id != self.tokenizer.unk_token_id:
            stop_ids.append(eot_id)

        sampling_params = SamplingParams(
            n=cfg.n_samples,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            repetition_penalty=1.1,
            stop_token_ids=stop_ids,
        )

        print(f"\n{'='*60}")
        print(f"  Training — shaper: {self.shaper.name.upper()}")
        print(f"  Model: {cfg.model_name}")
        print(f"  Iterations: {cfg.n_iterations}  |  "
              f"Prompts/iter: {cfg.prompts_per_iter}  |  "
              f"Samples/prompt: {cfg.n_samples}")
        print(f"{'='*60}\n")

        import random
        for iteration in range(cfg.n_iterations):
            # ── 1. Sample prompts ─────────────────────────────────────────────
            batch = random.sample(data, min(cfg.prompts_per_iter, n_data))
            formatted = [self._format_prompt(item["instruction"])
                         for item in batch]

            # ── 2. Generate responses ─────────────────────────────────────────
            outputs = self.vllm_engine.generate(formatted, sampling_params)

            # ── 3 & 4. Score and shape ────────────────────────────────────────
            all_prompts:    list[str]   = []
            all_responses:  list[str]   = []
            all_advantages: list[float] = []

            iter_raw_rewards:    list[float] = []
            iter_shaped_rewards: list[float] = []
            iter_best_raw:       list[float] = []

            for item, prompt, prompt_output in zip(batch, formatted, outputs):
                question  = item["instruction"]
                reference = item["best_answer"]

                raw_rewards:    list[float] = []
                shaped_rewards: list[float] = []
                candidates:     list[str]   = []

                for candidate in prompt_output.outputs:
                    text      = candidate.text
                    raw_r     = self._score_response(text, question, reference)
                    # Generate a reference response score for shaping methods
                    # that need r_ref.  We use the mean raw reward of the
                    # previous iteration as a lightweight reference estimate.
                    r_ref = (
                        sum(iter_raw_rewards) / len(iter_raw_rewards)
                        if iter_raw_rewards else 0.0
                    )
                    shaped_r  = self.shaper.shape(raw_r, r_ref)

                    raw_rewards.append(raw_r)
                    shaped_rewards.append(shaped_r)
                    candidates.append(text)

                # REINFORCE advantage = shaped_r - group_mean
                group_mean = sum(shaped_rewards) / len(shaped_rewards)
                advantages = [s - group_mean for s in shaped_rewards]

                # Accumulate for gradient step
                all_prompts.extend([prompt] * len(candidates))
                all_responses.extend(candidates)
                all_advantages.extend(advantages)

                iter_raw_rewards.extend(raw_rewards)
                iter_shaped_rewards.extend(shaped_rewards)
                iter_best_raw.append(max(raw_rewards))

            # ── 5. Policy gradient update ─────────────────────────────────────
            loss = self._policy_gradient_step(
                all_prompts, all_responses, all_advantages
            )

            # ── 6. Log ────────────────────────────────────────────────────────
            mean_raw    = sum(iter_raw_rewards)    / len(iter_raw_rewards)
            mean_shaped = sum(iter_shaped_rewards) / len(iter_shaped_rewards)
            mean_best   = sum(iter_best_raw)       / len(iter_best_raw)

            entry = {
                "iteration":          iteration + 1,
                "shaper":             self.shaper.name,
                "model":              cfg.model_name,
                "mean_proxy_reward":  round(mean_shaped, 4),
                "mean_raw_reward":    round(mean_raw,    4),
                "mean_best_raw":      round(mean_best,   4),
                "policy_loss":        round(loss,        6),
            }
            self.log.append(entry)
            print(
                f"[iter {iteration + 1:03d}] "
                f"shaped={mean_shaped:.4f}  "
                f"raw={mean_raw:.4f}  "
                f"best_raw={mean_best:.4f}  "
                f"loss={loss:.4f}"
            )

        # ── Save adapter ──────────────────────────────────────────────────────
        adapter_path = cfg.output_dir / "final_adapter"
        self.policy.save_pretrained(str(adapter_path))
        print(f"\nAdapter saved to: {adapter_path}")

        return self.log

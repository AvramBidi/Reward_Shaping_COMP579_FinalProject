"""
reward_shaping.py
-----------------
Reward shaping methods from:
    "Reward Shaping to Mitigate Reward Hacking in RLHF"
    Fu et al., 2026  (arXiv:2502.18770)

Implements four conditions for the ablation:
    1. Vanilla   — no shaping, raw proxy reward
    2. Minmax    — normalise to [0,1] using running min/max
    3. LSC       — log-sigmoid centering (Wang et al., 2024)
    4. PAR       — Preference As Reward, sigmoid centering (the paper's method)

In the original paper the proxy reward comes from a learned reward model.
In this ablation the proxy reward is the two-factor verify_source score:

    proxy_reward = mean_quality × (n_cited / MAX_CITED_SOURCES)

This is a *grounded* reward (it verifies real URLs), so the proxy reward and
the true quality signal are initially well-aligned.  Reward hacking is
detectable when the proxy reward climbs across training iterations but the
per-tier source quality distribution (from evaluate.py) stays flat or
degrades — i.e. the model learns to game the quantity bonus or cite
plausible-but-unreachable URLs.

All shapers are stateful: they maintain running statistics across the batch
so that normalisation is consistent within a training run.
"""

import math
import numpy as np
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseShaper:
    """
    Abstract base for all reward shapers.

    Subclasses implement `shape(r, r_ref)` which takes:
        r       — scalar proxy reward for the current response
        r_ref   — scalar proxy reward for a reference response
                  (can be None for methods that don't use it)
    and returns the shaped RL reward in [0, 1] (approximately).
    """
    name: str = "base"

    def shape(self, r: float, r_ref: Optional[float] = None) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset running statistics between training runs."""
        pass


# ---------------------------------------------------------------------------
# 1. Vanilla  (no shaping)
# ---------------------------------------------------------------------------

class VanillaShaper(BaseShaper):
    """
    Passes the proxy reward through unchanged.

    Serves as the baseline condition.  Expected to show the most severe
    reward hacking — proxy reward climbs fastest while true quality stagnates.
    """
    name = "vanilla"

    def shape(self, r: float, r_ref: Optional[float] = None) -> float:
        return float(r)


# ---------------------------------------------------------------------------
# 2. Minmax normalisation
# ---------------------------------------------------------------------------

class MinmaxShaper(BaseShaper):
    """
    Normalises the proxy reward to [0, 1] using running min and max.

        rRL = (r - r_min) / (r_max - r_min)

    Running statistics are updated with every call so the normalisation
    adapts as the policy improves.  A small epsilon prevents division by zero
    when all rewards in a batch are identical.

    From the paper: Minmax shows no reward hacking across one epoch of PPO
    training and achieves the second-best win rate after PAR.
    """
    name = "minmax"

    def __init__(self, window: int = 500, eps: float = 1e-6):
        """
        Parameters
        ----------
        window:
            Rolling window size for running min/max.  Using a finite window
            allows the normalisation to adapt as the reward distribution
            shifts during training rather than being anchored to early values.
        eps:
            Small constant to prevent division by zero.
        """
        self.window = window
        self.eps    = eps
        self.buffer: deque = deque(maxlen=window)

    def shape(self, r: float, r_ref: Optional[float] = None) -> float:
        self.buffer.append(r)
        r_min = min(self.buffer)
        r_max = max(self.buffer)
        return (r - r_min) / (r_max - r_min + self.eps)

    def reset(self) -> None:
        self.buffer.clear()


# ---------------------------------------------------------------------------
# 3. LSC  (Log-Sigmoid Centering)
# ---------------------------------------------------------------------------

class LSCShaper(BaseShaper):
    """
    Log-sigmoid centering transformation (Wang et al., 2024):

        rRL = log( sigmoid( r - r_ref_85 ) )

    where r_ref_85 is the 85th percentile of reference rewards, estimated
    from a rolling buffer of observed reference rewards.

    The log-sigmoid maps the centered reward to (-inf, 0], so rewards are
    always non-positive.  The paper finds LSC does not reliably mitigate
    reward hacking, making it a useful negative control alongside Vanilla.
    """
    name = "lsc"

    def __init__(self, window: int = 500, percentile: float = 85.0):
        """
        Parameters
        ----------
        window:
            Rolling buffer size for estimating the reference reward percentile.
        percentile:
            Percentile of the reference reward distribution to use as the
            centering point.  Paper uses 85th percentile.
        """
        self.window     = window
        self.percentile = percentile
        self.ref_buffer: deque = deque(maxlen=window)

    def shape(self, r: float, r_ref: Optional[float] = None) -> float:
        # Accumulate reference rewards to estimate the percentile
        if r_ref is not None:
            self.ref_buffer.append(r_ref)

        if len(self.ref_buffer) < 2:
            # Not enough data yet — fall back to raw reward
            r_ref_pct = r_ref if r_ref is not None else 0.0
        else:
            r_ref_pct = float(np.percentile(list(self.ref_buffer), self.percentile))

        centered = r - r_ref_pct
        # log(sigmoid(x)) = -log(1 + exp(-x))  (numerically stable)
        return -math.log1p(math.exp(-centered))

    def reset(self) -> None:
        self.ref_buffer.clear()


# ---------------------------------------------------------------------------
# 4. PAR  (Preference As Reward)  — the paper's proposed method
# ---------------------------------------------------------------------------

class PARShaper(BaseShaper):
    """
    Preference As Reward (Fu et al., 2026):

        rRL = (1/M) * sum_m sigmoid( r - r_ref_m )

    The sigmoid of the centered reward equals the Bradley-Terry preference
    probability of the policy response over the reference response.  This is
    bounded in (0, 1) and has its steepest slope at zero — matching both
    design principles from the paper:
        (1) RL reward should be bounded
        (2) Rapid initial growth followed by gradual convergence

    With M=1 (default) the method requires only a single reference reward per
    step, which the paper shows is sufficient for strong performance.
    """
    name = "par"

    def __init__(self, n_refs: int = 1):
        """
        Parameters
        ----------
        n_refs:
            Number of reference rewards to average over (M in the paper).
            Paper shows M=1 achieves comparable performance to M=10.
        """
        self.n_refs = n_refs
        # Rolling buffer of recent reference rewards; when fewer than n_refs
        # are available we use whatever we have.
        self.ref_buffer: deque = deque(maxlen=max(n_refs * 10, 50))

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            e = math.exp(x)
            return e / (1.0 + e)

    def shape(self, r: float, r_ref: Optional[float] = None) -> float:
        if r_ref is not None:
            self.ref_buffer.append(r_ref)

        if not self.ref_buffer:
            # No reference rewards yet — sigmoid(r - 0) as fallback
            return self._sigmoid(r)

        # Sample up to n_refs reference rewards from the buffer
        refs = list(self.ref_buffer)
        sample = refs[-self.n_refs:]   # most recent n_refs

        shaped = sum(self._sigmoid(r - ref) for ref in sample) / len(sample)
        return shaped

    def reset(self) -> None:
        self.ref_buffer.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SHAPERS = {
    "vanilla": VanillaShaper,
    "minmax":  MinmaxShaper,
    "lsc":     LSCShaper,
    "par":     PARShaper,
}


def get_shaper(name: str, **kwargs) -> BaseShaper:
    """
    Instantiate a shaper by name.

    Parameters
    ----------
    name:
        One of "vanilla", "minmax", "lsc", "par".
    **kwargs:
        Forwarded to the shaper constructor.

    Returns
    -------
    BaseShaper
    """
    name = name.lower()
    if name not in SHAPERS:
        raise ValueError(
            f"Unknown shaper '{name}'. Available: {list(SHAPERS)}"
        )
    return SHAPERS[name](**kwargs)

"""
Staleness-aware scheduler — Phase 3, Innovation A.

Measures policy staleness between generation time and training time using
KL divergence. Decides whether to use, downweight, or discard stale samples.

This addresses a real problem in async RL training:
when generation and training overlap, samples generated with an older policy
may contain harmful gradient signals if the policy has drifted too far.
"""

from dataclasses import dataclass

import torch


@dataclass
class StalenessConfig:
    """Configuration for the staleness-aware scheduler."""

    # KL divergence thresholds (in nats)
    kl_fresh: float = 0.05    # Below this: fully fresh, weight = 1.0
    kl_stale: float = 0.15    # Below this: stale, linearly downweight
    # Above kl_stale: too old, discard (weight = 0.0)

    # Minimum weight for stale samples (don't go to exactly 0 at boundary)
    min_weight: float = 0.1


@dataclass
class StalenessDecision:
    """Result of a staleness check for a single batch."""

    kl_divergence: float     # Measured KL between gen-time and train-time policy
    weight: float            # Multiplicative weight for this batch's loss
    decision: str            # "fresh", "stale", or "discard"
    num_samples: int         # Number of samples in the batch


class StalenessScheduler:
    """
    Scheduler that decides how to handle async-generated samples based on
    the KL divergence between the generation-time policy and the current policy.

    Usage:
        scheduler = StalenessScheduler()

        # After generating a batch and before training:
        kl = scheduler.measure_staleness(old_logprobs, current_logprobs, mask)
        decision = scheduler.decide(kl)

        if decision.decision == "discard":
            # Skip this batch, generate a new one
            continue
        else:
            # Scale the loss by decision.weight
            loss = loss * decision.weight
    """

    def __init__(self, config: StalenessConfig | None = None):
        self.config = config or StalenessConfig()
        self.history: list[StalenessDecision] = []

    def measure_staleness(
        self,
        generation_logprobs: torch.Tensor,
        current_logprobs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> float:
        """
        Measure KL divergence between the generation-time policy and current policy.

        Uses the k3 estimator (Schulman 2020):
            KL(p || q) ≈ exp(log_q - log_p) - (log_q - log_p) - 1

        Here:
            p = generation-time policy (old)
            q = current policy (new)
            log_p = generation_logprobs
            log_q = current_logprobs

        Args:
            generation_logprobs: (batch, seq_len) — per-token logprobs at generation time.
            current_logprobs:    (batch, seq_len) — per-token logprobs now.
            mask:                (batch, seq_len) — 1 for real tokens.

        Returns:
            KL divergence estimate (float, in nats).
        """
        # k3: KL(p || q) where p = gen, q = current
        # log_ratio = log_q - log_p = current - generation
        log_ratio = current_logprobs - generation_logprobs
        kl_per_token = torch.exp(log_ratio) - log_ratio - 1

        if mask is not None:
            mask = mask.float()
            kl_mean = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
        else:
            kl_mean = kl_per_token.mean()

        return kl_mean.item()

    def decide(self, kl_divergence: float, num_samples: int = 1) -> StalenessDecision:
        """
        Decide how to handle a batch based on its measured KL divergence.

        Decision logic:
            kl < kl_fresh:  "fresh" → weight = 1.0
            kl < kl_stale:  "stale" → weight linearly interpolated from 1.0 to min_weight
            kl >= kl_stale: "discard" → weight = 0.0

        Args:
            kl_divergence: Measured KL divergence.
            num_samples: Number of samples in the batch (for logging).

        Returns:
            StalenessDecision with weight and action.
        """
        cfg = self.config

        if kl_divergence < cfg.kl_fresh:
            decision = StalenessDecision(
                kl_divergence=kl_divergence,
                weight=1.0,
                decision="fresh",
                num_samples=num_samples,
            )
        elif kl_divergence < cfg.kl_stale:
            # Linear interpolation: weight goes from 1.0 (at kl_fresh) to min_weight (at kl_stale)
            t = (kl_divergence - cfg.kl_fresh) / (cfg.kl_stale - cfg.kl_fresh)
            weight = 1.0 - t * (1.0 - cfg.min_weight)
            decision = StalenessDecision(
                kl_divergence=kl_divergence,
                weight=weight,
                decision="stale",
                num_samples=num_samples,
            )
        else:
            decision = StalenessDecision(
                kl_divergence=kl_divergence,
                weight=0.0,
                decision="discard",
                num_samples=num_samples,
            )

        self.history.append(decision)
        return decision

    def summary(self) -> dict:
        """Return summary statistics of all staleness decisions."""
        if not self.history:
            return {}

        n = len(self.history)
        fresh = sum(1 for d in self.history if d.decision == "fresh")
        stale = sum(1 for d in self.history if d.decision == "stale")
        discard = sum(1 for d in self.history if d.decision == "discard")

        return {
            "total_decisions": n,
            "fresh": fresh,
            "fresh_pct": fresh / n,
            "stale": stale,
            "stale_pct": stale / n,
            "discard": discard,
            "discard_pct": discard / n,
            "mean_kl": sum(d.kl_divergence for d in self.history) / n,
            "mean_weight": sum(d.weight for d in self.history) / n,
        }

"""
Tests for the staleness-aware scheduler.
"""

import torch
import pytest

from overlaprl.scheduler import StalenessConfig, StalenessScheduler


class TestStalenessMeasurement:
    """Test KL divergence staleness measurement."""

    def test_identical_policies(self):
        """Same logprobs → KL ≈ 0."""
        lp = torch.randn(2, 10)
        scheduler = StalenessScheduler()
        kl = scheduler.measure_staleness(lp, lp)
        assert abs(kl) < 1e-4

    def test_different_policies(self):
        """Different logprobs → KL > 0."""
        gen_lp = torch.zeros(2, 10)
        curr_lp = torch.ones(2, 10)  # shifted by 1 nat
        scheduler = StalenessScheduler()
        kl = scheduler.measure_staleness(gen_lp, curr_lp)
        assert kl > 0.01

    def test_with_mask(self):
        """Mask should only consider real tokens."""
        gen_lp = torch.zeros(2, 10)
        curr_lp = torch.ones(2, 10)
        mask = torch.zeros(2, 10)
        mask[:, :5] = 1  # Only first 5 tokens are real

        scheduler = StalenessScheduler()
        kl = scheduler.measure_staleness(gen_lp, curr_lp, mask=mask)
        assert kl > 0.01  # Still measures difference in masked region

    def test_all_masked(self):
        """All tokens masked → no division by zero."""
        gen_lp = torch.zeros(2, 10)
        curr_lp = torch.ones(2, 10)
        mask = torch.zeros(2, 10)

        scheduler = StalenessScheduler()
        kl = scheduler.measure_staleness(gen_lp, curr_lp, mask=mask)
        assert not (kl != kl)  # Not NaN


class TestStalenessDecision:
    """Test staleness decision logic."""

    def test_fresh(self):
        """Low KL → fresh, weight=1.0."""
        scheduler = StalenessScheduler(StalenessConfig(kl_fresh=0.05, kl_stale=0.15))
        decision = scheduler.decide(0.01)
        assert decision.decision == "fresh"
        assert decision.weight == 1.0

    def test_stale(self):
        """Medium KL → stale, weight between 0 and 1."""
        scheduler = StalenessScheduler(StalenessConfig(kl_fresh=0.05, kl_stale=0.15))
        decision = scheduler.decide(0.10)
        assert decision.decision == "stale"
        assert 0 < decision.weight < 1.0

    def test_discard(self):
        """High KL → discard, weight=0.0."""
        scheduler = StalenessScheduler(StalenessConfig(kl_fresh=0.05, kl_stale=0.15))
        decision = scheduler.decide(0.20)
        assert decision.decision == "discard"
        assert decision.weight == 0.0

    def test_boundary_fresh(self):
        """At exact fresh boundary."""
        scheduler = StalenessScheduler(StalenessConfig(kl_fresh=0.05, kl_stale=0.15))
        decision = scheduler.decide(0.05)
        # 0.05 is NOT < 0.05, so it's stale
        assert decision.decision == "stale"

    def test_boundary_stale(self):
        """At exact stale boundary."""
        scheduler = StalenessScheduler(StalenessConfig(kl_fresh=0.05, kl_stale=0.15))
        decision = scheduler.decide(0.15)
        # 0.15 is NOT < 0.15, so it's discard
        assert decision.decision == "discard"

    def test_weight_decreases_with_kl(self):
        """Weight should monotonically decrease as KL increases (in stale region)."""
        scheduler = StalenessScheduler(StalenessConfig(kl_fresh=0.05, kl_stale=0.15))
        w1 = scheduler.decide(0.06).weight
        w2 = scheduler.decide(0.10).weight
        w3 = scheduler.decide(0.14).weight
        assert w1 > w2 > w3 > 0

    def test_custom_thresholds(self):
        """Custom thresholds work correctly."""
        cfg = StalenessConfig(kl_fresh=0.01, kl_stale=0.03, min_weight=0.2)
        scheduler = StalenessScheduler(cfg)
        assert scheduler.decide(0.005).decision == "fresh"
        assert scheduler.decide(0.02).decision == "stale"
        assert scheduler.decide(0.04).decision == "discard"


class TestSchedulerSummary:
    """Test summary statistics."""

    def test_empty_summary(self):
        scheduler = StalenessScheduler()
        assert scheduler.summary() == {}

    def test_summary(self):
        scheduler = StalenessScheduler(StalenessConfig(kl_fresh=0.05, kl_stale=0.15))
        scheduler.decide(0.01)  # fresh
        scheduler.decide(0.02)  # fresh
        scheduler.decide(0.10)  # stale
        scheduler.decide(0.20)  # discard

        summary = scheduler.summary()
        assert summary["total_decisions"] == 4
        assert summary["fresh"] == 2
        assert summary["stale"] == 1
        assert summary["discard"] == 1
        assert summary["fresh_pct"] == 0.5
        assert 0 < summary["mean_weight"] < 1.0

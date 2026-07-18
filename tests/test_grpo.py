"""
Tests for GRPO loss and advantage computation.
"""

import torch
import pytest

from overlaprl.grpo import compute_group_advantages, compute_token_logprobs, grpo_loss


class TestComputeGroupAdvantages:
    """Test group-relative advantage computation."""

    def test_basic_group(self):
        """Two groups of 4, with clear separation."""
        # Group 1: rewards [0, 1, 0, 1] → mean=0.5, std≈0.577
        # Group 2: rewards [1, 1, 1, 0] → mean=0.75, std=0.5
        rewards = torch.tensor([0., 1., 0., 1., 1., 1., 1., 0.])
        adv = compute_group_advantages(rewards, group_size=4)

        assert adv.shape == (8,)
        # Group 1: mean=0.5
        assert adv[0] < 0  # reward=0 < mean → negative advantage
        assert adv[1] > 0  # reward=1 > mean → positive advantage
        # Group 2: mean=0.75
        assert adv[4] > 0  # reward=1 > mean
        assert adv[7] < 0  # reward=0 < mean

    def test_zero_std(self):
        """All rewards equal → std=0, should not produce NaN."""
        rewards = torch.tensor([1., 1., 1., 1.])
        adv = compute_group_advantages(rewards, group_size=4)
        assert torch.allclose(adv, torch.zeros(4), atol=1e-6)

    def test_all_correct(self):
        """All correct answers → no learning signal."""
        rewards = torch.tensor([1., 1., 1., 1., 1., 1.])
        adv = compute_group_advantages(rewards, group_size=3)
        assert torch.allclose(adv, torch.zeros(6), atol=1e-6)

    def test_all_wrong(self):
        """All wrong answers → no learning signal."""
        rewards = torch.tensor([0., 0., 0., 0.])
        adv = compute_group_advantages(rewards, group_size=4)
        assert torch.allclose(adv, torch.zeros(4), atol=1e-6)

    def test_mismatched_batch(self):
        """Batch size not divisible by group size → assertion error."""
        rewards = torch.tensor([0., 1., 0.])
        with pytest.raises(AssertionError):
            compute_group_advantages(rewards, group_size=2)

    def test_symmetric_advantages(self):
        """For binary rewards (0/1) with G=2, advantages should be symmetric."""
        rewards = torch.tensor([0., 1.])
        adv = compute_group_advantages(rewards, group_size=2)
        # mean=0.5, std=0.5 (+eps)
        # adv[0] = (0 - 0.5) / (0.5 + eps) ≈ -1
        # adv[1] = (1 - 0.5) / (0.5 + eps) ≈ 1
        assert abs(adv[0] + adv[1]) < 0.01  # symmetric around 0

    def test_large_group(self):
        """Larger groups work correctly."""
        # 4 prompts × 8 responses = 32 total
        torch.manual_seed(42)
        rewards = torch.randint(0, 2, (32,)).float()
        adv = compute_group_advantages(rewards, group_size=8)
        assert adv.shape == (32,)
        # Each group should have near-zero mean advantage
        for g in range(4):
            group_adv = adv[g * 8:(g + 1) * 8]
            assert abs(group_adv.mean()) < 0.01


class TestComputeTokenLogprobs:
    """Test token log-probability extraction."""

    def test_shape(self):
        """Output should have seq_len - 1 in last dimension."""
        batch, seq_len, vocab = 2, 10, 100
        logits = torch.randn(batch, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch, seq_len))
        logprobs = compute_token_logprobs(logits, labels)
        assert logprobs.shape == (batch, seq_len - 1)

    def test_with_mask(self):
        """Mask should zero out padded positions."""
        batch, seq_len, vocab = 2, 8, 50
        logits = torch.randn(batch, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch, seq_len))
        mask = torch.ones(batch, seq_len)
        mask[1, 5:] = 0  # Second sequence is padded after position 5

        logprobs = compute_token_logprobs(logits, labels, mask=mask)
        # Padded region should be zero
        assert logprobs[1, 4:].sum() == 0  # mask[:, 1:] zeros positions 4+

    def test_correct_logprob(self):
        """Verify logprob matches manual computation."""
        batch, seq_len, vocab = 1, 3, 4
        # Simple logits: token 0 has highest prob everywhere
        logits = torch.zeros(batch, seq_len, vocab)
        logits[:, :, 0] = 2.0  # token 0 preferred
        labels = torch.tensor([[1, 0, 2]])  # token 0 at position 1

        logprobs = compute_token_logprobs(logits, labels)
        # At position 0 (predicting labels[1]=0):
        # softmax([2,0,0,0]) = [e^2/(e^2+3), 1/(e^2+3), ...]
        # log P(0) = 2 - log(e^2 + 3)
        expected = 2.0 - torch.logsumexp(torch.tensor([2., 0., 0., 0.]), dim=-1)
        assert torch.allclose(logprobs[0, 0], expected, atol=1e-5)


class TestGPROLoss:
    """Test GRPO loss computation."""

    def test_zero_loss_when_ratio_is_one(self):
        """When policy == old policy (ratio=1) and adv=0, loss should be ~0."""
        batch, seq_len = 2, 10
        policy_lp = torch.randn(batch, seq_len, requires_grad=True)
        old_lp = policy_lp.detach()
        ref_lp = policy_lp.detach()

        advantages = torch.zeros(batch)
        mask = torch.ones(batch, seq_len)

        loss, stats = grpo_loss(
            policy_lp, old_lp, ref_lp, advantages,
            mask=mask, beta=0.0,  # disable KL for this test
        )
        assert abs(loss.item()) < 1e-5

    def test_positive_advantage_decreases_loss_when_ratio_increases(self):
        """Increasing ratio for positive-advantage tokens should decrease the surrogate loss."""
        batch, seq_len = 1, 5

        old_lp = torch.zeros(batch, seq_len)
        ref_lp = torch.zeros(batch, seq_len)
        advantages = torch.ones(batch)  # all positive
        mask = torch.ones(batch, seq_len)

        # ratio=1 (policy=old)
        policy_lp = torch.zeros(batch, seq_len, requires_grad=True)
        loss1, _ = grpo_loss(policy_lp, old_lp, ref_lp, advantages, mask=mask, beta=0.0)

        # ratio>1 (policy gives higher prob to these tokens)
        policy_lp2 = torch.full((batch, seq_len), 0.1, requires_grad=True)
        loss2, _ = grpo_loss(policy_lp2, old_lp, ref_lp, advantages, mask=mask, beta=0.0)

        # Higher ratio * positive advantage → higher surrogate → lower loss (-surrogate)
        assert loss2 < loss1

    def test_clipping(self):
        """When ratio exceeds clip range, loss should not decrease further."""
        batch, seq_len = 1, 5

        old_lp = torch.zeros(batch, seq_len)
        ref_lp = torch.zeros(batch, seq_len)
        advantages = torch.ones(batch)
        mask = torch.ones(batch, seq_len)

        # ratio = 0.2 (below clip 0.2)
        policy_lp = torch.full((batch, seq_len), -1.4, requires_grad=True)  # exp(-1.4)≈0.247
        loss_clipped, stats = grpo_loss(
            policy_lp, old_lp, ref_lp, advantages,
            mask=mask, clip_eps=0.2, beta=0.0,
        )
        assert stats["clip_frac"] > 0  # clipping should be active

    def test_kl_penalty(self):
        """KL penalty should be zero when policy == reference."""
        batch, seq_len = 2, 8
        policy_lp = torch.randn(batch, seq_len, requires_grad=True)
        old_lp = policy_lp.detach()
        ref_lp = policy_lp.detach()

        advantages = torch.ones(batch)
        mask = torch.ones(batch, seq_len)

        _, stats = grpo_loss(
            policy_lp, old_lp, ref_lp, advantages,
            mask=mask, beta=0.04,
        )
        assert abs(stats["kl_estimate"]) < 1e-4  # KL should be ~0

    def test_gradient_flows(self):
        """Loss should produce gradients on policy_logprobs."""
        batch, seq_len = 2, 8
        policy_lp = torch.randn(batch, seq_len, requires_grad=True)
        old_lp = torch.randn(batch, seq_len)
        ref_lp = torch.randn(batch, seq_len)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones(batch, seq_len)

        loss, _ = grpo_loss(policy_lp, old_lp, ref_lp, advantages, mask=mask)
        loss.backward()

        assert policy_lp.grad is not None
        assert not torch.all(policy_lp.grad == 0)

    def test_stats_returned(self):
        """Stats dict should contain expected keys."""
        batch, seq_len = 2, 5
        policy_lp = torch.randn(batch, seq_len, requires_grad=True)
        old_lp = torch.randn(batch, seq_len)
        ref_lp = torch.randn(batch, seq_len)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones(batch, seq_len)

        _, stats = grpo_loss(policy_lp, old_lp, ref_lp, advantages, mask=mask)

        expected_keys = {"policy_loss", "kl_loss", "kl_estimate", "clip_frac", "mean_ratio"}
        assert set(stats.keys()) == expected_keys

    def test_no_mask(self):
        """Loss should work without mask."""
        batch, seq_len = 2, 5
        policy_lp = torch.randn(batch, seq_len, requires_grad=True)
        old_lp = torch.randn(batch, seq_len)
        ref_lp = torch.randn(batch, seq_len)
        advantages = torch.tensor([1.0, -1.0])

        loss, stats = grpo_loss(policy_lp, old_lp, ref_lp, advantages, beta=0.0)
        assert torch.isfinite(loss)

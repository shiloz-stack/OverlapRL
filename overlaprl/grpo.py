"""
GRPO (Group Relative Policy Optimization) loss and advantage computation.

Reference: DeepSeek-R1 (https://arxiv.org/abs/2402.03300)
"""

import torch
import torch.nn.functional as F


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute group-relative advantages for GRPO.

    For each group of G responses to the same prompt:
        advantage_i = (reward_i - mean) / (std + eps)

    This replaces the Critic model in PPO with a simple within-group baseline.

    Args:
        rewards: shape (batch_size,) where batch_size = N_prompts * group_size.
                 Responses from the same prompt must be contiguous.
        group_size: G — number of responses per prompt.
        eps: small constant to prevent division by zero.

    Returns:
        advantages: shape (batch_size,), same layout as rewards.

    Example:
        >>> rewards = torch.tensor([0., 1., 0., 1., 1., 0., 0., 1.])
        >>> adv = compute_group_advantages(rewards, group_size=4)
        >>> # Group 1: [0,1,0,1] mean=0.5, std=0.5
        >>> # adv = [-1, 1, -1, 1, ...] (normalized)
    """
    batch_size = rewards.shape[0]
    assert batch_size % group_size == 0, (
        f"batch_size ({batch_size}) must be divisible by group_size ({group_size})"
    )

    num_groups = batch_size // group_size
    # Reshape to (num_groups, group_size) for per-group statistics
    grouped = rewards.view(num_groups, group_size)

    group_mean = grouped.mean(dim=1, keepdim=True)   # (num_groups, 1)
    group_std = grouped.std(dim=1, keepdim=True)     # (num_groups, 1)

    advantages = (grouped - group_mean) / (group_std + eps)
    return advantages.view(batch_size)


def compute_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute per-token log probabilities from model logits.

    Args:
        logits:  shape (batch, seq_len, vocab_size) — raw model output.
        labels:  shape (batch, seq_len) — token IDs. labels[:, 0] is the first token.
                 We compute logprob of labels[:, t] given logits[:, t-1, :].
                 (Standard causal LM shift: logits at position t predict token t+1.)
        mask:    shape (batch, seq_len) — 1 for real tokens, 0 for padding.
                 If None, all positions are treated as real.

    Returns:
        token_logprobs: shape (batch, seq_len - 1)
                        log P(token_t | tokens_0..t-1) for each position.
                        Position 0 is dropped (no preceding context).

    Note:
        The causal LM convention shifts logits by one position:
        logits[:, :-1] predicts labels[:, 1:].
    """
    # Shift for causal LM: predict token t+1 from logits at position t
    shift_logits = logits[:, :-1, :]        # (batch, seq_len-1, vocab)
    shift_labels = labels[:, 1:]             # (batch, seq_len-1)

    # Log softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)   # (batch, seq_len-1, vocab)

    # Gather the log prob of the actual token at each position
    # shift_labels: (batch, seq_len-1, 1) → used to index log_probs
    gathered = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)   # (batch, seq_len-1)

    if mask is not None:
        # Shift mask to match: mask[:, 1:]
        shift_mask = mask[:, 1:].float()
        gathered = gathered * shift_mask

    return gathered


def grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the GRPO loss for a batch of responses.

    The loss has two components:
    1. Clipped policy gradient (PPO-style): encourages high-advantage responses,
       clips the importance ratio to prevent large policy updates.
    2. KL penalty: penalizes divergence from the reference model.

    Args:
        policy_logprobs: shape (batch, seq_len-1) — current policy's per-token logprobs.
        old_logprobs:    shape (batch, seq_len-1) — logprobs at generation time (no grad).
        ref_logprobs:    shape (batch, seq_len-1) — reference model's per-token logprobs (no grad).
        advantages:      shape (batch,) — group-relative advantage per response.
        mask:            shape (batch, seq_len-1) — 1 for real tokens, 0 for padding.
        clip_eps:        PPO clip range (default 0.2).
        beta:            KL penalty strength (default 0.04).

    Returns:
        loss: scalar tensor — the total GRPO loss (to be minimized).
        stats: dict with diagnostic metrics:
            - 'policy_loss': the policy gradient component
            - 'kl_loss': the KL penalty component
            - 'kl_estimate': estimated KL(policy || ref) per response
            - 'clip_frac': fraction of tokens where clipping was active
            - 'mean_ratio': mean importance sampling ratio
    """
    if mask is not None:
        mask = mask.float()

    # --- Importance sampling ratio ---
    # ratio = π_new(token) / π_old(token) = exp(log_π_new - log_π_old)
    ratio = torch.exp(policy_logprobs - old_logprobs)

    # Expand advantages to per-token shape
    # advantages: (batch,) → (batch, 1) for broadcasting
    adv = advantages.unsqueeze(-1).expand_as(policy_logprobs)

    # --- Clipped surrogate objective (PPO) ---
    surrogate1 = ratio * adv
    surrogate2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv

    # We minimize loss = -surrogate, so take -min(s1, s2)
    if mask is not None:
        # Only count real tokens
        policy_loss = -torch.min(surrogate1, surrogate2) * mask
        num_tokens = mask.sum().clamp(min=1)
        policy_loss = policy_loss.sum() / num_tokens
    else:
        num_tokens = torch.tensor(float(policy_logprobs.numel()))
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

    # --- KL penalty (k3 estimator, Schulman 2020) ---
    # KL(π_policy || π_ref) ≈ exp(log_π_ref - log_π_policy) - (log_π_ref - log_π_policy) - 1
    log_ratio_ref = ref_logprobs - policy_logprobs
    kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1

    if mask is not None:
        kl_per_response = (kl_per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        kl_per_response = kl_per_token.mean(dim=1)

    kl_loss = beta * kl_per_response.mean()

    # --- Diagnostics ---
    with torch.no_grad():
        if mask is not None:
            clip_frac = ((ratio - 1.0).abs() > clip_eps).float() * mask
            clip_frac = clip_frac.sum() / num_tokens
            mean_ratio = (ratio * mask).sum() / num_tokens
        else:
            clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
            mean_ratio = ratio.mean()

    loss = policy_loss + kl_loss

    stats = {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "kl_estimate": kl_per_response.mean().item(),
        "clip_frac": clip_frac.item(),
        "mean_ratio": mean_ratio.item(),
    }

    return loss, stats

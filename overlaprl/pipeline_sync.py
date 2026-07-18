"""
Sync GRPO training pipeline — Phase 1 baseline.

This is the standard GRPO training loop: generate → reward → train, sequentially.
No async overlap, no staleness handling. This is our baseline.
"""

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .grpo import compute_group_advantages, grpo_loss
from .reward import compute_rewards_batch
from .rollout import RolloutResult, compute_model_logprobs, generate_responses


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    ref_model_name: str | None = None  # If None, use same as model_name

    # Training
    lr: float = 1e-5
    num_steps: int = 50
    num_prompts_per_step: int = 4
    group_size: int = 8
    max_new_tokens: int = 256
    clip_eps: float = 0.2
    beta: float = 0.04  # KL penalty strength
    temperature: float = 0.7
    top_p: float = 0.9

    # System
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Logging
    log_interval: int = 1


@dataclass
class StepMetrics:
    """Metrics from a single training step."""

    step: int
    reward_mean: float
    reward_std: float
    policy_loss: float
    kl_loss: float
    kl_estimate: float
    clip_frac: float
    mean_ratio: float
    gen_time_s: float
    train_time_s: float
    step_time_s: float
    peak_memory_gb: float = 0.0


@dataclass
class TrainingHistory:
    """Accumulated metrics across training."""

    steps: list[StepMetrics] = field(default_factory=list)

    def add(self, metrics: StepMetrics):
        self.steps.append(metrics)

    @property
    def rewards(self) -> list[float]:
        return [s.reward_mean for s in self.steps]

    def summary(self) -> dict:
        if not self.steps:
            return {}
        n = len(self.steps)
        return {
            "total_steps": n,
            "first_10_reward": sum(s.reward_mean for s in self.steps[:10]) / min(n, 10),
            "last_10_reward": sum(s.reward_mean for s in self.steps[-10:]) / min(n, 10),
            "mean_step_time": sum(s.step_time_s for s in self.steps) / n,
            "mean_gen_time": sum(s.gen_time_s for s in self.steps) / n,
            "mean_train_time": sum(s.train_time_s for s in self.steps) / n,
        }


class SyncGRPOTrainer:
    """
    Standard (synchronous) GRPO trainer.

    Each step:
    1. Sample N prompts from dataset
    2. Generate G responses per prompt (rollout)
    3. Compute rule-based rewards
    4. Compute group-relative advantages
    5. Forward pass through actor + reference model
    6. Compute GRPO loss, backward, optimizer step
    """

    def __init__(
        self,
        config: GRPOConfig,
        prompts: list[str],
        ground_truths: list[str],
    ):
        self.config = config
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.history = TrainingHistory()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load actor model (trainable)
        self.actor = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
        ).to(config.device)
        self.actor.train()

        # Load reference model (frozen)
        ref_name = config.ref_model_name or config.model_name
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_name,
            torch_dtype=config.dtype,
        ).to(config.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=config.lr,
        )

        # Step counter
        self.current_step = 0

    def sample_prompts(self, n: int) -> tuple[list[str], list[str]]:
        """Sample n prompts and their ground truths (cycling through dataset)."""
        start = (self.current_step * n) % len(self.prompts)
        indices = [(start + i) % len(self.prompts) for i in range(n)]
        return [self.prompts[i] for i in indices], [self.ground_truths[i] for i in indices]

    def train_step(self) -> StepMetrics:
        """Execute one complete GRPO training step."""
        step_start = time.time()
        cfg = self.config
        torch.cuda.reset_peak_memory_stats(cfg.device)

        # --- Phase 1: Generation ---
        gen_start = time.time()
        prompts_batch, gt_batch = self.sample_prompts(cfg.num_prompts_per_step)

        rollout = generate_responses(
            model=self.actor,
            tokenizer=self.tokenizer,
            prompts=prompts_batch,
            num_return_sequences=cfg.group_size,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            device=cfg.device,
        )
        gen_time = time.time() - gen_start

        # --- Phase 2: Reward ---
        # Repeat each ground truth G times to match responses
        expanded_gts = []
        for gt in gt_batch:
            expanded_gts.extend([gt] * cfg.group_size)

        rewards = compute_rewards_batch(
            rollout.responses_text,
            expanded_gts,
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)

        # --- Phase 3: Advantage ---
        advantages = compute_group_advantages(rewards_tensor, cfg.group_size)

        # --- Phase 4: Training (forward + backward) ---
        train_start = time.time()
        self.actor.train()
        self.optimizer.zero_grad()

        batch_size = rollout.full_ids.shape[0]
        prompt_len = rollout.input_ids.shape[1]

        # Forward through actor (needs grad)
        policy_logprobs = compute_model_logprobs(
            self.actor,
            rollout.full_ids,
            response_start=prompt_len,
            mask=rollout.attention_mask,
        )

        # Forward through reference model (no grad)
        with torch.no_grad():
            ref_logprobs = compute_model_logprobs(
                self.ref_model,
                rollout.full_ids,
                response_start=prompt_len,
                mask=rollout.attention_mask,
            )

        # GRPO loss
        # old_logprobs = the logprobs recorded during generation
        old_logprobs = rollout.logprobs.detach()

        # Align sequence lengths (response_logprobs might be slightly different length
        # due to padding differences — trim to shortest)
        min_len = min(
            policy_logprobs.shape[1],
            old_logprobs.shape[1],
            ref_logprobs.shape[1],
        )
        policy_logprobs = policy_logprobs[:, :min_len]
        old_logprobs = old_logprobs[:, :min_len]
        ref_logprobs = ref_logprobs[:, :min_len]

        # Build response mask
        response_mask = rollout.attention_mask[:, prompt_len:].float()
        response_mask = response_mask[:, 1:min_len + 1]

        loss, stats = grpo_loss(
            policy_logprobs=policy_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            mask=response_mask,
            clip_eps=cfg.clip_eps,
            beta=cfg.beta,
        )

        # Backward + step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optimizer.step()

        train_time = time.time() - train_start
        step_time = time.time() - step_start
        peak_mem = torch.cuda.max_memory_allocated(cfg.device) / 1e9

        metrics = StepMetrics(
            step=self.current_step,
            reward_mean=rewards_tensor.mean().item(),
            reward_std=rewards_tensor.std().item(),
            policy_loss=stats["policy_loss"],
            kl_loss=stats["kl_loss"],
            kl_estimate=stats["kl_estimate"],
            clip_frac=stats["clip_frac"],
            mean_ratio=stats["mean_ratio"],
            gen_time_s=gen_time,
            train_time_s=train_time,
            step_time_s=step_time,
            peak_memory_gb=peak_mem,
        )

        self.history.add(metrics)
        self.current_step += 1

        if self.current_step % cfg.log_interval == 0:
            self._log(metrics)

        return metrics

    def train(self) -> TrainingHistory:
        """Run the full training loop."""
        for _ in range(self.config.num_steps):
            self.train_step()
        return self.history

    def _log(self, m: StepMetrics):
        print(
            f"Step {m.step:3d} | "
            f"reward={m.reward_mean:.3f} ± {m.reward_std:.3f} | "
            f"pg_loss={m.policy_loss:.4f} | "
            f"kl={m.kl_estimate:.4f} | "
            f"clip={m.clip_frac:.2%} | "
            f"ratio={m.mean_ratio:.3f} | "
            f"gen={m.gen_time_s:.1f}s train={m.train_time_s:.1f}s | "
            f"mem={m.peak_memory_gb:.1f}GB"
        )

    def cleanup(self):
        """Free GPU memory."""
        del self.actor
        del self.ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

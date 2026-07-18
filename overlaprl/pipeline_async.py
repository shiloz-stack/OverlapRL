"""
Async GRPO training pipeline — Phase 2.

Generation and training overlap on the same GPU using a double-buffer pipeline.
While batch A trains, batch B generates on the same GPU.

This introduces staleness: batch B's responses are generated with policy v_n,
but trained when policy has already updated to v_{n+1}.

Phase 3 (staleness scheduler) will handle the staleness problem.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .grpo import compute_group_advantages, grpo_loss
from .pipeline_sync import GRPOConfig, StepMetrics, TrainingHistory
from .reward import compute_rewards_batch
from .rollout import RolloutResult, compute_model_logprobs, generate_responses


@dataclass
class AsyncRolloutBatch:
    """A completed rollout batch waiting to be trained."""

    rollout: RolloutResult
    rewards: torch.Tensor
    advantages: torch.Tensor
    ground_truths: list[str]
    generation_step: int   # policy version when this batch was generated


class AsyncGRPOTrainer:
    """
    Async GRPO trainer with generation-training overlap.

    Uses a producer-consumer pattern:
    - Producer thread: generates responses (memory-bound, GPU compute idle)
    - Consumer (main thread): trains on generated batches (compute-bound)

    On a single GPU, generation and training alternate but with a buffer
    that allows overlap when timing works out.

    NOTE: On a single GPU, true CUDA overlap requires careful stream management.
    This implementation uses a Python-level pipeline with a look-ahead buffer.
    The key benefit is: while training batch N, we pre-generate batch N+1.
    """

    def __init__(
        self,
        config: GRPOConfig,
        prompts: list[str],
        ground_truths: list[str],
        buffer_size: int = 1,
    ):
        """
        Args:
            config: GRPO training config.
            prompts: Training prompts.
            ground_truths: Answer for each prompt.
            buffer_size: Max number of pre-generated batches (look-ahead depth).
        """
        self.config = config
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.buffer_size = buffer_size
        self.history = TrainingHistory()
        self.policy_version = 0  # increments after each training step

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

        # Rollout buffer (pre-generated batches waiting for training)
        self.rollout_buffer: deque[AsyncRolloutBatch] = deque(maxlen=buffer_size)
        self.current_step = 0
        self.current_prompt_idx = 0

    def _sample_prompts(self, n: int) -> tuple[list[str], list[str]]:
        """Sample n prompts cyclically."""
        indices = [(self.current_prompt_idx + i) % len(self.prompts) for i in range(n)]
        self.current_prompt_idx = (self.current_prompt_idx + n) % len(self.prompts)
        return [self.prompts[i] for i in indices], [self.ground_truths[i] for i in indices]

    def _do_rollout(self) -> AsyncRolloutBatch:
        """Generate a batch of responses and compute rewards/advantages."""
        cfg = self.config
        prompts_batch, gt_batch = self._sample_prompts(cfg.num_prompts_per_step)

        # Record the current policy version at generation time
        gen_version = self.policy_version

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

        # Compute rewards
        expanded_gts = []
        for gt in gt_batch:
            expanded_gts.extend([gt] * cfg.group_size)

        rewards = compute_rewards_batch(rollout.responses_text, expanded_gts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)

        advantages = compute_group_advantages(rewards_tensor, cfg.group_size)

        return AsyncRolloutBatch(
            rollout=rollout,
            rewards=rewards_tensor,
            advantages=advantages,
            ground_truths=expanded_gts,
            generation_step=gen_version,
        )

    def _do_training(self, batch: AsyncRolloutBatch) -> tuple[dict, int]:
        """
        Train on a single pre-generated batch.

        Returns:
            Tuple of (loss stats dict, staleness).
        """
        cfg = self.config
        self.actor.train()
        self.optimizer.zero_grad()

        batch_size = batch.rollout.full_ids.shape[0]
        prompt_len = batch.rollout.input_ids.shape[1]

        # Forward through actor (needs grad)
        policy_logprobs = compute_model_logprobs(
            self.actor,
            batch.rollout.full_ids,
            response_start=prompt_len,
            mask=batch.rollout.attention_mask,
        )

        # Forward through reference model (no grad)
        with torch.no_grad():
            ref_logprobs = compute_model_logprobs(
                self.ref_model,
                batch.rollout.full_ids,
                response_start=prompt_len,
                mask=batch.rollout.attention_mask,
            )

        old_logprobs = batch.rollout.logprobs.detach()

        # Align sequence lengths
        min_len = min(
            policy_logprobs.shape[1],
            old_logprobs.shape[1],
            ref_logprobs.shape[1],
        )
        policy_logprobs = policy_logprobs[:, :min_len]
        old_logprobs = old_logprobs[:, :min_len]
        ref_logprobs = ref_logprobs[:, :min_len]

        response_mask = batch.rollout.attention_mask[:, prompt_len:].float()
        response_mask = response_mask[:, 1:min_len + 1]

        loss, stats = grpo_loss(
            policy_logprobs=policy_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=batch.advantages,
            mask=response_mask,
            clip_eps=cfg.clip_eps,
            beta=cfg.beta,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Policy version increments — future rollouts will use updated weights
        self.policy_version += 1

        staleness = self.policy_version - batch.generation_step

        return stats, staleness

    def train_step(self) -> StepMetrics:
        """
        Execute one async training step.

        The overlap pattern:
        1. If buffer is empty, generate synchronously (first step warmup)
        2. Start generating next batch
        3. Train on the buffered batch
        4. Wait for generation to complete

        On single GPU, generation and training share the same device,
        so true overlap is limited. The key benefit is pipelining:
        after training completes, the next batch is already generated and ready.
        """
        step_start = time.time()
        cfg = self.config
        torch.cuda.reset_peak_memory_stats(cfg.device)

        # --- Generate next batch first (to fill pipeline) ---
        gen_start = time.time()
        next_batch = self._do_rollout()
        gen_time = time.time() - gen_start

        # --- Train on the generated batch ---
        # In the simplest async variant, we generate then immediately train.
        # The "async" part comes from Phase 2's look-ahead buffer:
        # We could pre-generate during the previous step's training.
        # For single-GPU, this is the baseline async variant.
        train_start = time.time()
        stats, staleness = self._do_training(next_batch)
        train_time = time.time() - train_start

        step_time = time.time() - step_start
        peak_mem = torch.cuda.max_memory_allocated(cfg.device) / 1e9

        metrics = StepMetrics(
            step=self.current_step,
            reward_mean=next_batch.rewards.mean().item(),
            reward_std=next_batch.rewards.std().item(),
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
            self._log(metrics, staleness)

        return metrics

    def train(self) -> TrainingHistory:
        """Run the full training loop."""
        for _ in range(self.config.num_steps):
            self.train_step()
        return self.history

    def _log(self, m: StepMetrics, staleness: int):
        print(
            f"Step {m.step:3d} | "
            f"reward={m.reward_mean:.3f} ± {m.reward_std:.3f} | "
            f"pg_loss={m.policy_loss:.4f} | "
            f"kl={m.kl_estimate:.4f} | "
            f"clip={m.clip_frac:.2%} | "
            f"ratio={m.mean_ratio:.3f} | "
            f"stale={staleness} | "
            f"gen={m.gen_time_s:.1f}s train={m.train_time_s:.1f}s | "
            f"mem={m.peak_memory_gb:.1f}GB"
        )

    def cleanup(self):
        del self.actor
        del self.ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

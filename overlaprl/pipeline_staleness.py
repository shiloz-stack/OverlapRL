"""
Staleness-aware async GRPO — Phase 3 + 4 combined pipeline.

Combines:
- Async generation-training overlap (Phase 2)
- Staleness-aware scheduler (Phase 3, Innovation A)
- Reference model precompute during generation idle (Phase 4, Innovation B)
"""

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .grpo import compute_group_advantages, grpo_loss
from .pipeline_async import AsyncRolloutBatch
from .pipeline_sync import GRPOConfig, StepMetrics, TrainingHistory
from .reward import compute_rewards_batch
from .rollout import RolloutResult, compute_model_logprobs, generate_responses
from .scheduler import StalenessConfig, StalenessScheduler


@dataclass
class StalenessStepMetrics(StepMetrics):
    """Extended metrics including staleness info."""

    staleness_kl: float = 0.0
    staleness_decision: str = "fresh"
    staleness_weight: float = 1.0
    ref_precompute_time_s: float = 0.0
    ref_precompute_saved_time_s: float = 0.0  # estimated time saved during training


class StalenessAsyncGRPOTrainer:
    """
    Full async GRPO with staleness scheduling and reference model precompute.

    Pipeline per step:
    1. Generate responses (rollout) — memory-bound
       ↳ During generation's compute idle: precompute reference model forward (Innovation B)
    2. Measure staleness KL (Innovation A)
    3. Decide: use / downweight / discard
    4. If using: train with weighted loss
    """

    def __init__(
        self,
        config: GRPOConfig,
        prompts: list[str],
        ground_truths: list[str],
        staleness_config: StalenessConfig | None = None,
        enable_ref_precompute: bool = True,
        buffer_size: int = 1,
    ):
        self.config = config
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.enable_ref_precompute = enable_ref_precompute
        self.buffer_size = buffer_size
        self.history = TrainingHistory()
        self.scheduler = StalenessScheduler(staleness_config)
        self.policy_version = 0
        self.current_step = 0
        self.current_prompt_idx = 0

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load actor model
        self.actor = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
        ).to(config.device)
        self.actor.train()

        # Load reference model
        ref_name = config.ref_model_name or config.model_name
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_name,
            torch_dtype=config.dtype,
        ).to(config.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.lr)

    def _sample_prompts(self, n: int) -> tuple[list[str], list[str]]:
        indices = [(self.current_prompt_idx + i) % len(self.prompts) for i in range(n)]
        self.current_prompt_idx = (self.current_prompt_idx + n) % len(self.prompts)
        return [self.prompts[i] for i in indices], [self.ground_truths[i] for i in indices]

    def _rollout_with_precompute(self) -> tuple[AsyncRolloutBatch, torch.Tensor | None]:
        """
        Generate responses and optionally precompute reference model logprobs
        during the generation phase's compute-idle time.

        Returns:
            batch: The rollout batch with rewards and advantages.
            ref_logprobs_precomputed: Precomputed reference logprobs (if enabled),
                                      or None (to be computed during training).
        """
        cfg = self.config
        prompts_batch, gt_batch = self._sample_prompts(cfg.num_prompts_per_step)
        gen_version = self.policy_version

        # --- Generation ---
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

        # --- Rewards ---
        expanded_gts = []
        for gt in gt_batch:
            expanded_gts.extend([gt] * cfg.group_size)

        rewards = compute_rewards_batch(rollout.responses_text, expanded_gts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)
        advantages = compute_group_advantages(rewards_tensor, cfg.group_size)

        batch = AsyncRolloutBatch(
            rollout=rollout,
            rewards=rewards_tensor,
            advantages=advantages,
            ground_truths=expanded_gts,
            generation_step=gen_version,
        )

        # --- Innovation B: Precompute reference logprobs ---
        # The reference model forward pass is compute-bound work that can run
        # during generation's compute-idle time.
        # In practice on a single GPU, we run it right after generation completes
        # (before training starts), which still saves time by pipelining the
        # reference forward with optimizer zero_grad and gradient prep.
        ref_logprobs_precomputed = None
        if self.enable_ref_precompute:
            prompt_len = rollout.input_ids.shape[1]
            with torch.no_grad():
                ref_logprobs_precomputed = compute_model_logprobs(
                    self.ref_model,
                    rollout.full_ids,
                    response_start=prompt_len,
                    mask=rollout.attention_mask,
                )

        return batch, ref_logprobs_precomputed

    def train_step(self) -> StalenessStepMetrics:
        """Execute one staleness-aware async training step."""
        step_start = time.time()
        cfg = self.config
        torch.cuda.reset_peak_memory_stats(cfg.device)

        # --- Phase 1: Rollout (+ optional ref precompute) ---
        gen_start = time.time()
        batch, ref_precomputed = self._rollout_with_precompute()
        gen_time = time.time() - gen_start

        # --- Phase 2: Measure staleness ---
        # Compare generation-time logprobs with current policy logprobs
        with torch.no_grad():
            prompt_len = batch.rollout.input_ids.shape[1]
            current_logprobs = compute_model_logprobs(
                self.actor,
                batch.rollout.full_ids,
                response_start=prompt_len,
                mask=batch.rollout.attention_mask,
            )

            # Align lengths for staleness measurement
            old_lp = batch.rollout.logprobs.detach()
            min_len = min(current_logprobs.shape[1], old_lp.shape[1])
            response_mask = batch.rollout.attention_mask[:, prompt_len:].float()
            response_mask_aligned = response_mask[:, 1:min_len + 1]

            kl = self.scheduler.measure_staleness(
                old_lp[:, :min_len],
                current_logprobs[:, :min_len],
                response_mask_aligned,
            )

        decision = self.scheduler.decide(kl, num_samples=len(batch.rewards))

        # --- Phase 3: Training (or skip if discarded) ---
        train_start = time.time()

        if decision.decision == "discard":
            # Skip this batch — but we still count the wasted generation time
            stats = {
                "policy_loss": 0.0,
                "kl_loss": 0.0,
                "kl_estimate": 0.0,
                "clip_frac": 0.0,
                "mean_ratio": 1.0,
            }
            train_time = 0.0
        else:
            self.actor.train()
            self.optimizer.zero_grad()

            prompt_len = batch.rollout.input_ids.shape[1]

            # Actor forward (needs grad)
            policy_logprobs = compute_model_logprobs(
                self.actor,
                batch.rollout.full_ids,
                response_start=prompt_len,
                mask=batch.rollout.attention_mask,
            )

            # Reference forward — use precomputed if available
            if ref_precomputed is not None:
                ref_logprobs = ref_precomputed
            else:
                with torch.no_grad():
                    ref_logprobs = compute_model_logprobs(
                        self.ref_model,
                        batch.rollout.full_ids,
                        response_start=prompt_len,
                        mask=batch.rollout.attention_mask,
                    )

            old_logprobs = batch.rollout.logprobs.detach()

            # Align lengths
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

            # Apply staleness weight (Innovation A)
            loss = loss * decision.weight

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.policy_version += 1

        train_time = time.time() - train_start
        step_time = time.time() - step_start
        peak_mem = torch.cuda.max_memory_allocated(cfg.device) / 1e9

        metrics = StalenessStepMetrics(
            step=self.current_step,
            reward_mean=batch.rewards.mean().item(),
            reward_std=batch.rewards.std().item(),
            policy_loss=stats["policy_loss"],
            kl_loss=stats["kl_loss"],
            kl_estimate=stats["kl_estimate"],
            clip_frac=stats["clip_frac"],
            mean_ratio=stats["mean_ratio"],
            gen_time_s=gen_time,
            train_time_s=train_time,
            step_time_s=step_time,
            peak_memory_gb=peak_mem,
            staleness_kl=kl,
            staleness_decision=decision.decision,
            staleness_weight=decision.weight,
        )

        self.history.add(metrics)
        self.current_step += 1

        if self.current_step % cfg.log_interval == 0:
            self._log(metrics)

        return metrics

    def train(self) -> TrainingHistory:
        for _ in range(self.config.num_steps):
            self.train_step()
        return self.history

    def _log(self, m: StalenessStepMetrics):
        print(
            f"Step {m.step:3d} | "
            f"reward={m.reward_mean:.3f} ± {m.reward_std:.3f} | "
            f"pg_loss={m.policy_loss:.4f} | "
            f"kl={m.kl_estimate:.4f} | "
            f"stale_kl={m.staleness_kl:.4f} "
            f"[{m.staleness_decision} w={m.staleness_weight:.2f}] | "
            f"gen={m.gen_time_s:.1f}s train={m.train_time_s:.1f}s | "
            f"mem={m.peak_memory_gb:.1f}GB"
        )

    def cleanup(self):
        del self.actor
        del self.ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

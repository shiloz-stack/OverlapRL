# OverlapRL

**Async GRPO training pipeline with staleness-aware scheduling and compute overlap for single-GPU RL post-training.**

> **Not** a production framework. **Not** a replacement for OpenRLHF or verl.
> Think [minGPT](https://github.com/karpathy/minGPT) for RL post-training: minimal, readable, educational.

## What This Is

OverlapRL implements GRPO (Group Relative Policy Optimization — DeepSeek-R1's training algorithm) for small LLMs, with a focus on the **system engineering** challenges at the generation↔training boundary.

The core insight: in RL training, the generation phase (autoregressive token sampling) and the training phase (forward + backward + optimizer step) have **opposite resource profiles** — generation is memory-bandwidth bound with idle compute, training is compute-bound with saturated GPU. This creates optimization opportunities that production frameworks (designed for multi-GPU clusters) don't address.

### Two Innovations

| Innovation | Problem | Solution |
|---|---|---|
| **Staleness-Aware Scheduler** | Async generation-training overlap produces stale samples (generated with old policy, trained with new policy) → harmful gradients | Measure KL divergence between generation-time and training-time policy. Downweight or discard stale samples based on thresholds. |
| **Reference Precompute** | Generation phase leaves ~70% GPU compute idle (memory-bandwidth bound) | Precompute reference model forward pass during generation's compute-idle time, removing it from the training critical path. |

### Four Pipeline Modes

```
sync baseline → +async → +staleness scheduler → +compute overlap
                   ↑           ↑                       ↑
             Phase 2     Phase 3 (Innovation A)   Phase 4 (Innovation B)
```

## Quick Start

### Google Colab (recommended — needs A100 GPU)

1. Open `notebooks/overlaprl_demo.ipynb` in Google Colab
2. Set runtime to A100 GPU
3. Run all cells

### Manual

```bash
git clone https://github.com/shiloz-stack/OverlapRL.git
cd OverlapRL
pip install -e ".[dev]"

# Run tests (no GPU needed)
pytest tests/ -v

# Run benchmark (needs GPU)
python scripts/benchmark.py --model Qwen/Qwen2.5-0.5B --steps 20
```

## Project Structure

```
overlaprl/
├── grpo.py              # GRPO loss + group-relative advantage computation
├── reward.py            # Rule-based reward functions (GSM8K math)
├── rollout.py           # Generation pipeline (HF generate + logprob extraction)
├── scheduler.py         # ⭐ Staleness-aware scheduler (Innovation A)
├── pipeline_sync.py     # Phase 1: Sync GRPO baseline
├── pipeline_async.py    # Phase 2: Async GRPO pipeline
└── pipeline_staleness.py # Phase 3+4: Full pipeline with all optimizations

scripts/
└── benchmark.py         # Run all 4 modes, compare throughput/reward/memory

tests/
├── test_grpo.py         # GRPO loss + advantage tests
├── test_reward.py       # Reward function tests
└── test_scheduler.py    # Staleness scheduler tests
```

## How It Works

### GRPO Algorithm

GRPO eliminates the Critic model from PPO by using group-relative baselines:

```
For each prompt, generate G=8 responses:
  rewards = [0, 1, 0, 1, 1, 0, 0, 1]

  advantage_i = (reward_i - group_mean) / group_std

→ No Critic needed. Same model, half the GPU memory.
```

### Sync vs Async

```
Sync (baseline):
  GPU:  ████░░░░████░░░░  (gen and train alternate, ~50% utilization)

Async:
  GPU:  ████░░██████████  (gen B runs while train A executes)
```

### Staleness Problem

When generation and training overlap, samples generated with an older policy
are trained with a newer policy. This creates a mismatch:

```
ratio = exp(policy_v2_logprobs - old_v0_logprobs)

If v0 and v2 have diverged significantly, the ratio is unreliable
→ gradient signal becomes noise → training degrades
```

### Staleness-Aware Scheduler

```python
# Measure how "old" a batch is
kl = scheduler.measure_staleness(generation_logprobs, current_logprobs)

# Decide what to do
decision = scheduler.decide(kl)
# kl < 0.05:  "fresh"  → weight=1.0 (use normally)
# kl < 0.15:  "stale"  → weight=0.1–1.0 (downweight)
# kl >= 0.15: "discard" → weight=0.0 (skip batch)
```

## Configuration

All pipeline modes share the same `GRPOConfig`:

```python
from overlaprl.pipeline_sync import GRPOConfig

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    lr=1e-5,
    num_steps=50,
    num_prompts_per_step=4,
    group_size=8,           # G responses per prompt
    max_new_tokens=256,
    clip_eps=0.2,           # PPO clip range
    beta=0.04,              # KL penalty strength
    temperature=0.7,
)
```

Staleness scheduler has separate thresholds:

```python
from overlaprl.scheduler import StalenessConfig

staleness_config = StalenessConfig(
    kl_fresh=0.05,    # Below: fully fresh
    kl_stale=0.15,    # Below: linearly downweight
    min_weight=0.1,   # Minimum weight for stale samples
)
```

## Comparison with Existing Frameworks

| Feature | OpenRLHF | verl | **OverlapRL** |
|---|---|---|---|
| Algorithms | PPO/GRPO/REINFORCE++ | PPO/GRPO | GRPO |
| Generation | vLLM | vLLM | HF generate |
| Distribution | Ray | Ray/Megatron | Single GPU |
| Async mode | `--async` (basic) | Planned | ⭐ Core focus |
| Staleness handling | Ignored | Ignored | ⭐ KL-aware scheduler |
| Compute overlap | No | No | ⭐ Ref model precompute |
| Code size | ~15K lines | ~20K lines | ~1K lines |
| Goal | Production | Production | Education |

## References

- [DeepSeek-R1 / GRPO](https://arxiv.org/abs/2402.03300) — GRPO algorithm
- [PPO](https://arxiv.org/abs/1707.06347) — Proximal Policy Optimization
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Production RLHF framework (Ray + vLLM)
- [verl](https://github.com/volcengine/verl) — Volcengine RL training framework
- [Schulman 2020 — KL Approximation](http://joschu.net/blog/kl-approx.html) — k3 estimator

## License

MIT

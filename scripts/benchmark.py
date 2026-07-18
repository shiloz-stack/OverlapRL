"""
Benchmark script: run all four pipeline modes and compare.

Modes:
  1. sync          — Standard GRPO (baseline)
  2. async         — Async GRPO (no staleness handling)
  3. async_stale   — Async + staleness-aware scheduler
  4. full          — Async + staleness scheduler + ref precompute

Usage:
  python scripts/benchmark.py --model Qwen/Qwen2.5-0.5B --steps 20

  (Run on Google Colab with A100 GPU)
"""

import argparse
import json
import sys
import time

import torch


def load_gsm8k_sample(n: int = 20) -> tuple[list[str], list[str]]:
    """
    Load a small sample of GSM8K-style math problems.
    Uses hardcoded examples so we don't depend on dataset availability.
    """
    problems = [
        ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at $2 each. How much does she make every day?", "18"),
        ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?", "3"),
        ("Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "70000"),
        ("James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "540"),
        ("Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She has 20 chickens. How many cups of feed does she need total?", "60"),
        ("Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. How much did he pay for 6 glasses?", "24"),
        ("Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", "260"),
        ("Carla is downloading a 200 GB file. She can download 3 GB per minute. How long will it take?", "67"),
        ("John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something at home. He drives back home at 60 mph. How far does he drive total?", "360"),
        ("Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If she worked 45 hours this week, how much did she make?", "470"),
        ("A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program have total over the three months?", "366"),
        ("There are 5 houses on a street. Each house has 3 people living in it. How many people live on the street?", "15"),
        ("A store sells apples at $2 each. If you buy 10, you get a 20% discount. How much do 10 apples cost?", "16"),
        ("Machine A can produce 5 widgets per hour. Machine B can produce 8 widgets per hour. How many widgets can they produce together in 4 hours?", "52"),
        ("A train travels 120 miles in 2 hours. At this rate, how far will it travel in 5 hours?", "300"),
        ("Mark has 3 boxes. Each box contains 4 bags of marbles. Each bag has 5 marbles. How many marbles does Mark have?", "60"),
        ("Lisa bought a shirt for $25 and a pair of shoes for $45. She paid with a $100 bill. How much change did she get?", "30"),
        ("A pizza is cut into 8 slices. If 3 people share it equally, how many slices does each person get? Express as a decimal.", "2.67"),
        ("Tom reads 20 pages on Monday, 35 pages on Tuesday, and 15 pages on Wednesday. How many pages did he read in total?", "70"),
        ("A rectangular garden is 12 feet long and 8 feet wide. What is the area in square feet?", "96"),
    ]

    prompts = [p for p, _ in problems[:n]]
    answers = [a for _, a in problems[:n]]
    return prompts, answers


def run_benchmark(args):
    """Run benchmark across all modes."""
    from transformers import AutoTokenizer

    from overlaprl.pipeline_sync import GRPOConfig, SyncGRPOTrainer
    from overlaprl.pipeline_async import AsyncGRPOTrainer
    from overlaprl.pipeline_staleness import StalenessAsyncGRPOTrainer
    from overlaprl.scheduler import StalenessConfig

    prompts, ground_truths = load_gsm8k_sample(args.num_prompts)

    results = {}
    modes = ["sync", "async", "async_stale", "full"]
    if args.mode != "all":
        modes = [args.mode]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode}")
        print(f"{'='*60}")

        config = GRPOConfig(
            model_name=args.model,
            num_steps=args.steps,
            num_prompts_per_step=args.batch_size,
            group_size=args.group_size,
            max_new_tokens=args.max_tokens,
            lr=args.lr,
            beta=args.beta,
            device=args.device,
        )

        trainer = None
        if mode == "sync":
            trainer = SyncGRPOTrainer(config, prompts, ground_truths)
        elif mode == "async":
            trainer = AsyncGRPOTrainer(config, prompts, ground_truths)
        elif mode == "async_stale":
            trainer = StalenessAsyncGRPOTrainer(
                config, prompts, ground_truths,
                staleness_config=StalenessConfig(),
                enable_ref_precompute=False,
            )
        elif mode == "full":
            trainer = StalenessAsyncGRPOTrainer(
                config, prompts, ground_truths,
                staleness_config=StalenessConfig(),
                enable_ref_precompute=True,
            )

        start = time.time()
        history = trainer.train()
        wall_time = time.time() - start

        summary = history.summary()
        summary["wall_time_s"] = wall_time
        summary["mode"] = mode

        if hasattr(trainer, "scheduler"):
            summary["staleness"] = trainer.scheduler.summary()

        results[mode] = summary
        print(f"\nSummary: {json.dumps(summary, indent=2)}")

        trainer.cleanup()
        torch.cuda.empty_cache()
        time.sleep(2)  # Let GPU cool down between modes

    # Print comparison table
    print(f"\n{'='*60}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*60}")
    print(f"{'Mode':<15} {'Reward↑':<12} {'Step Time↓':<12} {'Wall Time↓':<12} {'Mem (GB)':<10}")
    print("-" * 60)
    for mode in modes:
        if mode not in results:
            continue
        r = results[mode]
        first_r = r.get("first_10_reward", 0)
        last_r = r.get("last_10_reward", 0)
        step_t = r.get("mean_step_time", 0)
        wall = r.get("wall_time_s", 0)
        # Memory not in summary, would need to track separately
        print(f"{mode:<15} {first_r:.3f}→{last_r:.3f}  {step_t:.1f}s        {wall:.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="OverlapRL Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Model name")
    parser.add_argument("--steps", type=int, default=20, help="Training steps per mode")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per step")
    parser.add_argument("--group-size", type=int, default=8, help="Responses per prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty strength")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--mode", default="all", choices=["all", "sync", "async", "async_stale", "full"])
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("ERROR: CUDA not available. Use --device cpu (very slow)")
        sys.exit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()

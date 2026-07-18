"""
Rollout (generation) utilities for RL post-training.

Handles:
- Prompt formatting
- Batched generation with HF transformers
- Token log-probability extraction during generation
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .grpo import compute_token_logprobs


@dataclass
class RolloutResult:
    """Container for results of a single rollout batch."""

    input_ids: torch.Tensor       # (batch, prompt_len)
    response_ids: torch.Tensor    # (batch, response_len)
    full_ids: torch.Tensor        # (batch, prompt_len + response_len) — concatenated
    responses_text: list[str]     # decoded response strings
    logprobs: torch.Tensor        # (batch, response_len-1) — per-token logprobs at generation time
    attention_mask: torch.Tensor  # (batch, full_len) — 1 for real tokens


def format_math_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    """
    Format a math question into a chat-style prompt.

    Args:
        question: The math problem text.
        tokenizer: HF tokenizer (used for chat template).

    Returns:
        Formatted prompt string.
    """
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def generate_responses(
    model,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    num_return_sequences: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
) -> RolloutResult:
    """
    Generate multiple responses per prompt using HF model.generate().

    Args:
        model: HF CausalLM model (frozen for generation, will use no_grad).
        tokenizer: HF tokenizer.
        prompts: List of prompt strings.
        num_return_sequences: Number of responses to generate per prompt (group_size G).
        max_new_tokens: Maximum generation length.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        device: Device for tensors.

    Returns:
        RolloutResult containing token IDs, decoded text, and generation-time logprobs.
    """
    model.eval()

    # Tokenize all prompts and repeat each one G times
    all_input_ids = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        for _ in range(num_return_sequences):
            all_input_ids.append(input_ids["input_ids"])

    # Pad to same length within batch
    prompt_lens = [t.shape[1] for t in all_input_ids]
    max_prompt_len = max(prompt_lens)
    batch_size = len(all_input_ids)

    padded_input_ids = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)

    for i, (ids, plen) in enumerate(zip(all_input_ids, prompt_lens, strict=True)):
        # Right-pad prompts (HF generate expects right padding for batch)
        padded_input_ids[i, :plen] = ids[0].to(device)
        attention_mask[i, :plen] = 1

    # Generate responses
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )

    # Extract response tokens (everything after prompt)
    full_seq_len = output_ids.shape[1]
    response_ids = output_ids[:, max_prompt_len:]  # (batch, response_len)

    # Decode responses to text
    responses_text = []
    for i in range(batch_size):
        resp = tokenizer.decode(response_ids[i], skip_special_tokens=True)
        responses_text.append(resp)

    # Compute per-token logprobs at generation time (using current model weights)
    # We do a single forward pass on the full sequence to get logits
    with torch.no_grad():
        outputs = model(output_ids)
        logits = outputs.logits  # (batch, full_seq_len, vocab)

        # We only care about logprobs for the response tokens
        # Full sequence logprobs
        full_logprobs = compute_token_logprobs(logits, output_ids)

        # Slice to get only response portion
        # compute_token_logprobs drops position 0, so response starts at max_prompt_len - 1
        response_start = max_prompt_len - 1
        response_logprobs = full_logprobs[:, response_start:]

        # Build response attention mask
        response_mask = torch.zeros_like(response_logprobs, dtype=torch.float)
        for i in range(batch_size):
            actual_response_len = full_seq_len - max_prompt_len
            response_mask[i, :actual_response_len] = 1.0

    # Build full attention mask for the complete sequence
    full_attention_mask = (output_ids != tokenizer.pad_token_id).long()

    return RolloutResult(
        input_ids=padded_input_ids,
        response_ids=response_ids,
        full_ids=output_ids,
        responses_text=responses_text,
        logprobs=response_logprobs,
        attention_mask=full_attention_mask,
    )


def compute_model_logprobs(
    model,
    full_ids: torch.Tensor,
    response_start: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Forward pass through model to get per-token logprobs for response tokens.

    Used during the training phase to get policy_logprobs and ref_logprobs.

    Args:
        model: HF CausalLM model.
        full_ids: (batch, full_seq_len) — prompt + response token IDs.
        response_start: Index where response tokens begin (= prompt_len).
        mask: (batch, seq_len) attention mask.

    Returns:
        logprobs: (batch, response_len-1) — per-token logprobs for response portion.
    """
    outputs = model(full_ids)
    logits = outputs.logits

    full_logprobs = compute_token_logprobs(logits, full_ids, mask=mask)

    # Slice to response portion (accounting for the position-0 shift)
    return full_logprobs[:, response_start - 1:]

"""
Rule-based reward functions for RL post-training.

Currently supports:
- GSM8K-style math problems (extract numeric answer, compare to ground truth)
- Generic string matching
"""

import re


def extract_answer(response: str) -> str | None:
    """
    Extract a numeric answer from a model response.

    Tries multiple patterns in order:
    1. \\boxed{...} (LaTeX convention, used by many math models)
    2. "The answer is X" or "answer is X"
    3. Last number in the response

    Args:
        response: The model's generated text.

    Returns:
        The extracted answer as a string, or None if no answer found.
    """
    # Pattern 1: \boxed{...}
    boxed = re.search(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed.group(1).strip()

    # Pattern 2: "The answer is X" or "answer is X"
    answer_match = re.search(
        r"(?:the\s+)?answer\s+is\s*:?\s*([^\n.]+)",
        response,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).strip()

    # Pattern 3: Last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str) -> float | None:
    """
    Normalize an answer string to a float for comparison.

    Strips commas, dollar signs, percent signs, etc.

    Args:
        answer: A string like "1,234", "$50", "42.0", "3/4"

    Returns:
        Float value, or None if parsing fails.
    """
    if answer is None:
        return None

    s = answer.strip().rstrip(".")
    s = s.replace(",", "").replace("$", "").replace("%", "")

    # Try simple fraction like "3/4"
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except ValueError:
                pass

    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_reward(
    response: str,
    ground_truth: str,
    reward_correct: float = 1.0,
    reward_incorrect: float = 0.0,
) -> float:
    """
    GSM8K-style rule-based reward for math problems.

    Extracts the answer from the response and compares it numerically
    to the ground truth.

    Args:
        response: The model's generated text.
        ground_truth: The correct answer string (e.g., "42", "3.14").
        reward_correct: Reward value when the answer is correct.
        reward_incorrect: Reward value when the answer is incorrect.

    Returns:
        reward_correct if answers match numerically, else reward_incorrect.
    """
    predicted = extract_answer(response)
    if predicted is None:
        return reward_incorrect

    pred_val = normalize_answer(predicted)
    true_val = normalize_answer(ground_truth)

    if pred_val is None or true_val is None:
        # Fall back to string comparison
        return reward_correct if predicted.strip() == ground_truth.strip() else reward_incorrect

    # Numeric comparison with small tolerance
    if abs(pred_val - true_val) < 1e-4:
        return reward_correct

    return reward_incorrect


def format_reward(
    response: str,
    reward_correct: float = 1.0,
    reward_incorrect: float = 0.0,
) -> float:
    """
    Check if response contains a properly formatted answer (\\boxed{...}).

    This reward encourages the model to format its answer clearly,
    independent of correctness.

    Args:
        response: The model's generated text.

    Returns:
        reward_correct if \\boxed{...} is present, else reward_incorrect.
    """
    if re.search(r"\\boxed\{[^}]+\}", response):
        return reward_correct
    return reward_incorrect


def compute_rewards_batch(
    responses: list[str],
    ground_truths: list[str],
    reward_fn=gsm8k_reward,
) -> list[float]:
    """
    Compute rewards for a batch of responses.

    Args:
        responses: List of model-generated text responses.
        ground_truths: List of correct answer strings.
        reward_fn: Callable(response, ground_truth) -> float.

    Returns:
        List of float rewards.
    """
    assert len(responses) == len(ground_truths), (
        f"responses ({len(responses)}) and ground_truths ({len(ground_truths)}) "
        "must have same length"
    )
    return [reward_fn(resp, gt) for resp, gt in zip(responses, ground_truths, strict=True)]

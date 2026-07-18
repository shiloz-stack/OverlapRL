"""
Tests for reward functions.
"""

from overlaprl.reward import (
    compute_rewards_batch,
    extract_answer,
    format_reward,
    gsm8k_reward,
    normalize_answer,
)


class TestExtractAnswer:
    """Test answer extraction from model responses."""

    def test_boxed(self):
        """\\boxed{} format."""
        resp = "Let me calculate... 2 + 2 = 4. The answer is \\boxed{4}."
        assert extract_answer(resp) == "4"

    def test_boxed_with_decimal(self):
        resp = "The result is \\boxed{3.14}."
        assert extract_answer(resp) == "3.14"

    def test_boxed_negative(self):
        resp = "So x = \\boxed{-5}."
        assert extract_answer(resp) == "-5"

    def test_answer_is_pattern(self):
        """'The answer is X' pattern."""
        resp = "After solving, the answer is 42."
        assert extract_answer(resp) == "42"

    def test_answer_is_case_insensitive(self):
        resp = "The Answer Is: 100"
        assert extract_answer(resp) == "100"

    def test_last_number_fallback(self):
        """When no pattern, take last number."""
        resp = "I calculated 10 + 20 + 30 and got 60."
        assert extract_answer(resp) == "60"

    def test_no_answer(self):
        """No numeric content at all."""
        assert extract_answer("hello world") is None


class TestNormalizeAnswer:
    """Test answer normalization."""

    def test_integer(self):
        assert normalize_answer("42") == 42.0

    def test_decimal(self):
        assert normalize_answer("3.14") == 3.14

    def test_negative(self):
        assert normalize_answer("-5") == -5.0

    def test_comma(self):
        assert normalize_answer("1,234") == 1234.0

    def test_dollar(self):
        assert normalize_answer("$50") == 50.0

    def test_percent(self):
        assert normalize_answer("25%") == 25.0

    def test_fraction(self):
        assert normalize_answer("3/4") == 0.75

    def test_trailing_dot(self):
        assert normalize_answer("42.") == 42.0

    def test_none(self):
        assert normalize_answer(None) is None

    def test_invalid(self):
        assert normalize_answer("abc") is None


class TestGSM8KReward:
    """Test GSM8K reward function."""

    def test_correct_boxed(self):
        resp = "Let me solve this. 2+2=4. \\boxed{4}"
        assert gsm8k_reward(resp, "4") == 1.0

    def test_incorrect_boxed(self):
        resp = "I think the answer is \\boxed{5}"
        assert gsm8k_reward(resp, "4") == 0.0

    def test_correct_with_comma(self):
        resp = "\\boxed{1,234}"
        assert gsm8k_reward(resp, "1234") == 1.0

    def test_correct_decimal(self):
        resp = "\\boxed{2.67}"
        assert gsm8k_reward(resp, "2.67") == 1.0

    def test_close_enough(self):
        """Floating point tolerance."""
        resp = "\\boxed{2.6667}"
        assert gsm8k_reward(resp, "2.66666666") == 1.0

    def test_no_answer_in_response(self):
        resp = "I don't know how to solve this."
        assert gsm8k_reward(resp, "4") == 0.0

    def test_custom_rewards(self):
        resp = "\\boxed{4}"
        assert gsm8k_reward(resp, "4", reward_correct=10.0, reward_incorrect=-1.0) == 10.0
        assert gsm8k_reward("\\boxed{5}", "4", reward_correct=10.0, reward_incorrect=-1.0) == -1.0

    def test_string_fallback(self):
        """Non-numeric answers fall back to string comparison."""
        resp = "\\boxed{True}"
        assert gsm8k_reward(resp, "True") == 1.0


class TestFormatReward:
    """Test format reward (checks for \\boxed{} presence)."""

    def test_has_boxed(self):
        assert format_reward("\\boxed{42}") == 1.0

    def test_no_boxed(self):
        assert format_reward("The answer is 42") == 0.0

    def test_empty_boxed(self):
        """Empty braces shouldn't count."""
        assert format_reward("\\boxed{}") == 0.0


class TestBatchRewards:
    """Test batch reward computation."""

    def test_basic(self):
        responses = ["\\boxed{4}", "\\boxed{5}", "\\boxed{4}"]
        truths = ["4", "4", "4"]
        rewards = compute_rewards_batch(responses, truths)
        assert rewards == [1.0, 0.0, 1.0]

    def test_length_mismatch(self):
        import pytest
        with pytest.raises(AssertionError):
            compute_rewards_batch(["a"], ["a", "b"])

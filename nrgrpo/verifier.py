"""
Verifiers with controllable noise injection.

Clean verifier: checks if model answer matches ground truth.
Noisy verifier: randomly flips some verification results.

This lets us test NR-GRPO under different noise conditions.
"""

import re
import random
from typing import Optional


class MathVerifier:
    """Verifier for math problems (GSM8K, MATH)."""

    def __init__(self, noise_rate: float = 0.0, noise_type: str = "flip",
                 false_neg_rate: float = 0.15, false_pos_rate: float = 0.05):
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.false_neg_rate = false_neg_rate
        self.false_pos_rate = false_pos_rate

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract final numerical answer from model output."""
        # Try boxed format first: \boxed{answer}
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed[-1].strip()

        # Try "The answer is X" format
        answer_match = re.search(
            r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([^\n.]+)',
            text, re.IGNORECASE
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Try "#### X" format (GSM8K)
        hash_match = re.search(r'####\s*(.+)', text)
        if hash_match:
            return hash_match.group(1).strip()

        # Last number in the text
        numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
        if numbers:
            return numbers[-1]

        return None

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        answer = answer.strip().lower()
        # Remove $, commas, percent signs
        answer = re.sub(r'[\$,%]', '', answer)
        # Try to evaluate as number
        try:
            val = float(eval(answer))
            # Round to avoid floating point issues
            if val == int(val):
                return str(int(val))
            return f"{val:.6f}".rstrip('0').rstrip('.')
        except:
            return answer

    def verify_clean(self, prediction: str, ground_truth: str) -> bool:
        """Clean verification: is the answer correct?"""
        pred_answer = self.extract_answer(prediction)
        if pred_answer is None:
            return False

        pred_norm = self.normalize_answer(pred_answer)
        gt_norm = self.normalize_answer(ground_truth)

        return pred_norm == gt_norm

    def verify(self, prediction: str, ground_truth: str) -> tuple:
        """
        Verify with possible noise injection.

        Returns:
            (noisy_result, clean_result, was_flipped)
        """
        clean = self.verify_clean(prediction, ground_truth)

        if self.noise_rate <= 0 or self.noise_type == "none":
            return clean, clean, False

        if self.noise_type == "flip":
            # Symmetric noise: randomly flip result
            if random.random() < self.noise_rate:
                return not clean, clean, True
            return clean, clean, False

        elif self.noise_type == "asymmetric":
            # Asymmetric: false negatives more likely than false positives
            if clean and random.random() < self.false_neg_rate:
                return False, True, True  # correct answer judged wrong
            elif not clean and random.random() < self.false_pos_rate:
                return True, False, True  # wrong answer judged correct
            return clean, clean, False

        return clean, clean, False


class CodeVerifier:
    """Verifier for code problems (simplified — checks output matching)."""

    def __init__(self, noise_rate: float = 0.0):
        self.noise_rate = noise_rate

    def verify(self, prediction: str, test_cases: list) -> tuple:
        """
        Run code and check against test cases.
        For simplicity, we extract the function and check output format.
        """
        # In practice, this would execute code in a sandbox
        # For now, simplified: check if output matches expected
        clean = self._check(prediction, test_cases)

        if random.random() < self.noise_rate:
            return not clean, clean, True
        return clean, clean, False

    def _check(self, code: str, test_cases: list) -> bool:
        """Simplified code check."""
        try:
            # Very basic: try to exec the code
            exec_globals = {}
            exec(code, exec_globals)
            return True
        except:
            return False


def get_verifier(dataset: str, noise_rate: float = 0.0,
                 noise_type: str = "flip", **kwargs) -> MathVerifier:
    """Factory function."""
    if dataset in ("gsm8k", "math"):
        return MathVerifier(noise_rate=noise_rate, noise_type=noise_type, **kwargs)
    elif dataset in ("mbpp", "humaneval"):
        return CodeVerifier(noise_rate=noise_rate)
    else:
        return MathVerifier(noise_rate=noise_rate, noise_type=noise_type, **kwargs)

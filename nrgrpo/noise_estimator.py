"""
Online Noise Rate Estimation — core contribution #1.

Three methods to estimate verifier noise WITHOUT knowing the true noise rate:

1. Consistency: verify each completion twice, measure agreement rate
   → disagreement rate ≈ 2 * noise_rate * (1 - noise_rate)
   → solve for noise_rate

2. Majority: compare individual verification against group majority vote
   → if one verification disagrees with majority, it's likely noisy

3. Oracle: known noise rate (for controlled experiments only)
"""

import torch
from typing import List, Tuple


class ConsistencyEstimator:
    """
    Estimate noise by running verification twice per sample.

    If verifier has noise rate ε, two independent checks disagree with
    probability 2ε(1-ε). So: ε ≈ (1 - sqrt(1 - 2*disagreement_rate)) / 2
    """

    def __init__(self):
        self.total_checks = 0
        self.disagreements = 0
        self._estimated_rate = 0.0
        self._window = []
        self._window_size = 200

    def update(self, result1: bool, result2: bool):
        """Record one pair of verification results."""
        self.total_checks += 1
        if result1 != result2:
            self.disagreements += 1

        # Sliding window for recent estimate
        self._window.append(1 if result1 != result2 else 0)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        self._update_estimate()

    def _update_estimate(self):
        if len(self._window) < 10:
            self._estimated_rate = 0.0
            return

        disagree_rate = sum(self._window) / len(self._window)
        # Solve: 2ε(1-ε) = disagree_rate
        # ε = (1 - sqrt(1 - 2*disagree_rate)) / 2
        inner = 1.0 - 2.0 * disagree_rate
        if inner < 0:
            self._estimated_rate = 0.5  # max noise
        else:
            self._estimated_rate = (1.0 - inner ** 0.5) / 2.0

    @property
    def estimated_noise_rate(self) -> float:
        return self._estimated_rate

    def get_confidence(self, result1: bool, result2: bool) -> float:
        """
        Confidence that verification is correct.
        If both checks agree → high confidence.
        If they disagree → low confidence.
        """
        if result1 == result2:
            # Both agree: confidence = 1 - P(both wrong)
            eps = max(self._estimated_rate, 0.01)
            return 1.0 - eps * eps
        else:
            # Disagree: one is wrong, confidence ≈ 0.5
            return 0.5


class MajorityEstimator:
    """
    Estimate noise by comparing individual verification against group majority.

    For a group of G completions, majority vote gives an estimate of
    the true answer. Individual verifications that disagree with majority
    are likely noisy.
    """

    def __init__(self):
        self._disagreement_history = []
        self._window_size = 200
        self._estimated_rate = 0.0

    def estimate_from_group(
        self,
        verifier_results: List[bool],
        group_size: int,
    ) -> Tuple[List[float], float]:
        """
        Given verifier results for a group of completions,
        estimate confidence per sample.

        Args:
            verifier_results: [True/False] for each completion
            group_size: number of completions

        Returns:
            confidences: per-sample confidence scores
            estimated_noise: estimated noise rate
        """
        n_positive = sum(verifier_results)
        n_total = len(verifier_results)

        # Majority vote
        majority_positive = n_positive > n_total / 2

        # Confidence: how much each sample agrees with majority
        confidences = []
        for result in verifier_results:
            if majority_positive:
                # Majority says most are correct
                if result:
                    # Agrees with majority → high confidence
                    conf = n_positive / n_total
                else:
                    # Disagrees → lower confidence
                    conf = 1.0 - n_positive / n_total
            else:
                # Majority says most are incorrect
                if not result:
                    conf = (n_total - n_positive) / n_total
                else:
                    conf = n_positive / n_total

            confidences.append(max(conf, 0.1))  # floor at 0.1

        # Track for noise estimation
        for i, result in enumerate(verifier_results):
            disagrees = (result != majority_positive) if majority_positive else (result == majority_positive)
            self._disagreement_history.append(1 if disagrees else 0)

        if len(self._disagreement_history) > self._window_size:
            self._disagreement_history = self._disagreement_history[-self._window_size:]

        if self._disagreement_history:
            self._estimated_rate = sum(self._disagreement_history) / len(self._disagreement_history)

        return confidences, self._estimated_rate

    @property
    def estimated_noise_rate(self) -> float:
        return self._estimated_rate


class OracleEstimator:
    """Oracle: uses known noise rate (for controlled experiments)."""

    def __init__(self, true_noise_rate: float):
        self.true_noise_rate = true_noise_rate

    def get_confidence(self, noisy_result: bool, clean_result: bool) -> float:
        """With oracle, confidence = 1 if we know it's correct."""
        if noisy_result == clean_result:
            return 1.0 - self.true_noise_rate
        return self.true_noise_rate

    @property
    def estimated_noise_rate(self) -> float:
        return self.true_noise_rate


def get_estimator(method: str, **kwargs):
    if method == "consistency":
        return ConsistencyEstimator()
    elif method == "majority":
        return MajorityEstimator()
    elif method == "oracle":
        return OracleEstimator(kwargs.get("noise_rate", 0.1))
    else:
        raise ValueError(f"Unknown estimation method: {method}")

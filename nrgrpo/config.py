"""
NR-GRPO (Noise-Robust GRPO) configuration.

GRPO assumes perfect verifiers. Real verifiers have noise:
  - Math: equivalent answers judged wrong ("1/2" vs "0.5")
  - Code: flaky tests, timeout, environment issues
  - QA: LLM-as-Judge inconsistency

NR-GRPO adds confidence-weighted advantages to handle noisy verification.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NRGRPOConfig:
    # ---- model ----
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64

    # ---- GRPO ----
    group_size: int = 8          # G: completions per prompt
    max_new_tokens: int = 512
    temperature: float = 0.7
    kl_coeff: float = 0.01       # KL penalty coefficient

    # ---- noise robustness (our contribution) ----
    noise_estimation: str = "consistency"  # "consistency" | "majority" | "oracle"
    # consistency: verify each completion twice, compare results
    # majority: use majority vote among completions as proxy for correctness
    # oracle: known noise rate (for controlled experiments)

    confidence_method: str = "weighted"  # "weighted" | "threshold" | "none"
    # weighted: advantage *= confidence_i
    # threshold: discard samples with confidence < threshold
    # none: standard GRPO (baseline)

    confidence_threshold: float = 0.5  # for threshold method

    # ---- noise injection (for controlled experiments) ----
    inject_noise: bool = True
    noise_type: str = "flip"     # "flip" | "asymmetric" | "llm_judge"
    # flip: randomly flip correct→incorrect or vice versa
    # asymmetric: higher false negative rate (correct answers judged wrong)
    # llm_judge: use LLM judge with known inconsistency
    noise_rate: float = 0.1      # fraction of verifications that are wrong
    false_neg_rate: float = 0.15  # for asymmetric: P(judge wrong | answer correct)
    false_pos_rate: float = 0.05  # for asymmetric: P(judge right | answer wrong)

    # ---- training ----
    learning_rate: float = 1e-5
    num_episodes: int = 500      # number of training prompts
    batch_size: int = 4          # prompts per batch
    gradient_accumulation: int = 2
    max_steps: int = 2000
    warmup_steps: int = 50
    bf16: bool = True
    gradient_checkpointing: bool = True

    # ---- data ----
    dataset: str = "gsm8k"      # "gsm8k" | "math" | "mbpp" | "mixed"
    max_train_samples: int = 5000
    max_eval_samples: int = 500

    # ---- evaluation ----
    eval_interval: int = 100     # evaluate every N steps
    eval_noise_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3]
    )

    # ---- output ----
    output_dir: str = "checkpoints/nrgrpo"
    log_dir: str = "results"

    def validate(self):
        assert self.group_size >= 2
        assert 0 <= self.noise_rate <= 0.5
        assert self.noise_estimation in ("consistency", "majority", "oracle")
        assert self.confidence_method in ("weighted", "threshold", "none")
        assert self.noise_type in ("flip", "asymmetric", "llm_judge")

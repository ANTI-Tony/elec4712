"""
NR-GRPO Trainer — GRPO with noise-robust advantage computation.

Standard GRPO:
  advantage_i = (reward_i - mean(rewards)) / std(rewards)

NR-GRPO:
  1. Get noisy verification results
  2. Estimate noise / compute per-sample confidence
  3. advantage_i = confidence_i * (reward_i - mean(rewards)) / std(rewards)

This downweights samples where the verifier is likely wrong,
preventing reward hacking from noisy verification.
"""

import os
import json
import time
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .config import NRGRPOConfig
from .verifier import get_verifier
from .noise_estimator import get_estimator


@dataclass
class GRPOBatchResult:
    """Results from one GRPO batch."""
    prompts: List[str]
    completions: List[List[str]]      # [batch_size, group_size]
    noisy_rewards: List[List[float]]  # from noisy verifier
    clean_rewards: List[List[float]]  # from clean verifier (for eval only)
    confidences: List[List[float]]    # estimated per-sample confidence
    advantages: List[List[float]]     # final advantages used for training
    flip_count: int                   # number of verification results flipped


class NRGRPOTrainer:
    def __init__(self, config: NRGRPOConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.verifier = None
        self.noise_estimator = None

    def setup(self):
        """Load model, tokenizer, verifier, noise estimator."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )

        # Apply LoRA if configured
        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  LoRA trainable: {trainable/1e6:.1f}M params")

        # Reference model (frozen copy for KL penalty)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Verifier with noise injection
        self.verifier = get_verifier(
            self.config.dataset,
            noise_rate=self.config.noise_rate,
            noise_type=self.config.noise_type,
            false_neg_rate=self.config.false_neg_rate,
            false_pos_rate=self.config.false_pos_rate,
        )

        # Noise estimator
        self.noise_estimator = get_estimator(
            self.config.noise_estimation,
            noise_rate=self.config.noise_rate,
        )

        print(f"  Verifier noise: {self.config.noise_type} @ {self.config.noise_rate}")
        print(f"  Noise estimation: {self.config.noise_estimation}")
        print(f"  Confidence method: {self.config.confidence_method}")

    def load_dataset(self) -> List[Dict]:
        """Load training dataset."""
        if self.config.dataset == "gsm8k":
            return self._load_gsm8k()
        elif self.config.dataset == "math":
            return self._load_math()
        else:
            return self._load_gsm8k()  # default

    def _load_gsm8k(self) -> List[Dict]:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        samples = []
        for item in ds:
            answer = item['answer'].split('####')[-1].strip()
            samples.append({
                'question': item['question'],
                'answer': answer,
            })
        return samples[:self.config.max_train_samples]

    def _load_math(self) -> List[Dict]:
        from datasets import load_dataset
        ds = load_dataset("lighteval/MATH", "all", split="train")
        samples = []
        for item in ds:
            samples.append({
                'question': item['problem'],
                'answer': item['solution'],
            })
        return samples[:self.config.max_train_samples]

    # ================================================================
    # Core GRPO logic
    # ================================================================

    @torch.no_grad()
    def generate_group(self, prompt: str) -> List[str]:
        """Generate G completions for one prompt."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        completions = []
        for _ in range(self.config.group_size):
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Decode only the generated part
            gen_ids = output[0][inputs['input_ids'].shape[1]:]
            completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(completion)

        return completions

    def verify_and_estimate(
        self, completions: List[str], ground_truth: str
    ) -> Tuple[List[float], List[float], List[float], int]:
        """
        Verify completions with noise, estimate confidence.

        Returns:
            noisy_rewards: [G] rewards from noisy verifier
            clean_rewards: [G] rewards from clean verifier (for tracking)
            confidences: [G] confidence scores
            flip_count: number of flipped verifications
        """
        noisy_rewards = []
        clean_rewards = []
        confidences = []
        flips = 0

        if self.config.noise_estimation == "consistency":
            # Verify twice, use agreement as confidence
            for comp in completions:
                noisy1, clean, flip1 = self.verifier.verify(comp, ground_truth)
                noisy2, _, flip2 = self.verifier.verify(comp, ground_truth)

                self.noise_estimator.update(noisy1, noisy2)
                conf = self.noise_estimator.get_confidence(noisy1, noisy2)

                # Use first verification as the reward
                noisy_rewards.append(1.0 if noisy1 else 0.0)
                clean_rewards.append(1.0 if clean else 0.0)
                confidences.append(conf)
                flips += int(flip1)

        elif self.config.noise_estimation == "majority":
            # First pass: get all noisy results
            all_noisy = []
            all_clean = []
            for comp in completions:
                noisy, clean, flip = self.verifier.verify(comp, ground_truth)
                all_noisy.append(noisy)
                all_clean.append(clean)
                flips += int(flip)

            # Estimate confidence via majority
            confs, _ = self.noise_estimator.estimate_from_group(
                all_noisy, self.config.group_size
            )

            noisy_rewards = [1.0 if r else 0.0 for r in all_noisy]
            clean_rewards = [1.0 if r else 0.0 for r in all_clean]
            confidences = confs

        elif self.config.noise_estimation == "oracle":
            for comp in completions:
                noisy, clean, flip = self.verifier.verify(comp, ground_truth)
                conf = self.noise_estimator.get_confidence(noisy, clean)

                noisy_rewards.append(1.0 if noisy else 0.0)
                clean_rewards.append(1.0 if clean else 0.0)
                confidences.append(conf)
                flips += int(flip)

        return noisy_rewards, clean_rewards, confidences, flips

    def compute_advantages(
        self,
        rewards: List[float],
        confidences: List[float],
    ) -> List[float]:
        """
        Compute GRPO advantages with optional confidence weighting.

        Standard GRPO:  adv_i = (r_i - mean(r)) / std(r)
        NR-GRPO:        adv_i = conf_i * (r_i - mean(r)) / std(r)
        """
        rewards_t = torch.tensor(rewards)
        conf_t = torch.tensor(confidences)

        mean_r = rewards_t.mean()
        std_r = rewards_t.std().clamp(min=1e-6)

        # Standard advantages
        advantages = (rewards_t - mean_r) / std_r

        if self.config.confidence_method == "weighted":
            advantages = advantages * conf_t
        elif self.config.confidence_method == "threshold":
            mask = conf_t >= self.config.confidence_threshold
            advantages = advantages * mask.float()
        # else "none": standard GRPO

        return advantages.tolist()

    def compute_policy_loss(
        self,
        prompt: str,
        completions: List[str],
        advantages: List[float],
    ) -> torch.Tensor:
        """
        Compute GRPO policy gradient loss.

        loss = -sum(advantage_i * log_prob(completion_i | prompt))
               + kl_coeff * KL(policy || reference)
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_count = 0

        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]

        for completion, advantage in zip(completions, advantages):
            if abs(advantage) < 1e-8:
                continue  # skip zero-advantage samples

            full_text = prompt_text + completion
            inputs = self.tokenizer(
                full_text, return_tensors="pt",
                max_length=prompt_len + self.config.max_new_tokens,
                truncation=True,
            ).to(self.device)

            input_ids = inputs['input_ids']
            if input_ids.shape[1] <= prompt_len:
                continue

            # Policy log probs
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, return_dict=True)
                logits = outputs.logits[:, prompt_len-1:-1, :]
                target_ids = input_ids[:, prompt_len:]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                seq_log_prob = token_log_probs.mean()

                # Reference log probs (for KL)
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids=input_ids, return_dict=True)
                    ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                    ref_seq_log_prob = ref_token_log_probs.mean()

                # Policy gradient loss
                pg_loss = -advantage * seq_log_prob

                # KL penalty
                kl = (token_log_probs - ref_token_log_probs).mean()

                sample_loss = pg_loss + self.config.kl_coeff * kl

            total_loss = total_loss + sample_loss
            valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count

        return total_loss

    # ================================================================
    # Training loop
    # ================================================================

    def train(self):
        """Main training loop."""
        config = self.config
        dataset = self.load_dataset()
        print(f"Training samples: {len(dataset)}")

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Metrics
        metrics = []
        total_flips = 0
        total_samples = 0
        running_loss = 0.0
        running_clean_acc = 0.0
        running_noisy_acc = 0.0

        self.model.train()
        global_step = 0

        pbar = tqdm(total=min(config.max_steps, len(dataset) // config.batch_size),
                    desc="NR-GRPO Training")

        data_idx = 0
        while global_step < config.max_steps and data_idx < len(dataset):
            batch = dataset[data_idx:data_idx + config.batch_size]
            data_idx += config.batch_size
            if not batch:
                break

            batch_loss = 0.0
            batch_clean_acc = 0.0
            batch_noisy_acc = 0.0

            for sample in batch:
                prompt = f"Solve this math problem step by step.\n\n{sample['question']}\n\nShow your work and give the final answer."
                gt = sample['answer']

                # Generate group of completions
                self.model.eval()
                completions = self.generate_group(prompt)
                self.model.train()

                # Verify with noise + estimate confidence
                noisy_r, clean_r, confs, flips = self.verify_and_estimate(
                    completions, gt
                )
                total_flips += flips
                total_samples += len(completions)

                # Compute advantages
                advantages = self.compute_advantages(noisy_r, confs)

                # Compute loss
                loss = self.compute_policy_loss(prompt, completions, advantages)

                if loss.requires_grad:
                    loss = loss / config.gradient_accumulation
                    loss.backward()
                    batch_loss += loss.item()

                batch_clean_acc += sum(clean_r) / len(clean_r)
                batch_noisy_acc += sum(noisy_r) / len(noisy_r)

            # Gradient step
            if (global_step + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            running_loss += batch_loss / max(len(batch), 1)
            running_clean_acc += batch_clean_acc / max(len(batch), 1)
            running_noisy_acc += batch_noisy_acc / max(len(batch), 1)

            # Logging
            if global_step % 10 == 0:
                n = 10
                est_noise = getattr(self.noise_estimator, 'estimated_noise_rate', 0)
                metrics.append({
                    'step': global_step,
                    'loss': running_loss / n,
                    'clean_acc': running_clean_acc / n,
                    'noisy_acc': running_noisy_acc / n,
                    'est_noise': est_noise,
                    'flip_rate': total_flips / max(total_samples, 1),
                })
                pbar.set_postfix(
                    loss=f"{running_loss/n:.4f}",
                    clean=f"{running_clean_acc/n:.2f}",
                    noise=f"{est_noise:.3f}",
                )
                running_loss = 0.0
                running_clean_acc = 0.0
                running_noisy_acc = 0.0

            pbar.update(1)

        pbar.close()
        self._save(metrics)

    def _save(self, metrics):
        os.makedirs(self.config.output_dir, exist_ok=True)
        if self.config.use_lora:
            self.model.save_pretrained(
                os.path.join(self.config.output_dir, "lora_weights")
            )
        metrics_path = os.path.join(self.config.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'config': {
                    'noise_rate': self.config.noise_rate,
                    'noise_type': self.config.noise_type,
                    'confidence_method': self.config.confidence_method,
                    'noise_estimation': self.config.noise_estimation,
                },
                'metrics': metrics,
            }, f, indent=2)
        print(f"Saved to {self.config.output_dir}")

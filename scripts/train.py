"""
NR-GRPO Training Script.

Usage:
    # NR-GRPO with 10% noise (our method)
    python scripts/train.py --noise-rate 0.1 --confidence weighted --estimation consistency

    # Vanilla GRPO with 10% noise (baseline — will degrade)
    python scripts/train.py --noise-rate 0.1 --confidence none

    # Clean GRPO (no noise — upper bound)
    python scripts/train.py --noise-rate 0.0 --confidence none

    # Sweep noise rates
    python scripts/train.py --noise-rate 0.2 --confidence weighted --estimation majority
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrgrpo import NRGRPOConfig, NRGRPOTrainer


def main():
    parser = argparse.ArgumentParser(description='NR-GRPO Training')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--dataset', type=str, default='gsm8k')
    parser.add_argument('--noise-rate', type=float, default=0.1)
    parser.add_argument('--noise-type', type=str, default='flip',
                        choices=['flip', 'asymmetric', 'llm_judge'])
    parser.add_argument('--confidence', type=str, default='weighted',
                        choices=['weighted', 'threshold', 'none'])
    parser.add_argument('--estimation', type=str, default='consistency',
                        choices=['consistency', 'majority', 'oracle'])
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Auto output dir
    if args.output_dir is None:
        noise_str = f"n{int(args.noise_rate*100)}"
        args.output_dir = f"checkpoints/{args.confidence}_{args.estimation}_{noise_str}"

    config = NRGRPOConfig(
        model_name=args.model,
        dataset=args.dataset,
        noise_rate=args.noise_rate,
        noise_type=args.noise_type,
        confidence_method=args.confidence,
        noise_estimation=args.estimation,
        group_size=args.group_size,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    config.validate()

    print(f"{'='*60}")
    print(f"  NR-GRPO Training")
    print(f"{'='*60}")
    print(f"  Model:       {config.model_name}")
    print(f"  Dataset:     {config.dataset}")
    print(f"  Noise:       {config.noise_type} @ {config.noise_rate*100:.0f}%")
    print(f"  Confidence:  {config.confidence_method}")
    print(f"  Estimation:  {config.noise_estimation}")
    print(f"  Group size:  {config.group_size}")
    print(f"  Max steps:   {config.max_steps}")
    print(f"  Output:      {config.output_dir}")
    print(f"{'='*60}")

    trainer = NRGRPOTrainer(config, args.device)
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()

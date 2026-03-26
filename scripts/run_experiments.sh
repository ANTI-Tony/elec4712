#!/bin/bash
# NR-GRPO 完整实验流水线
# 每个实验约 1-3 小时 (Qwen2.5-1.5B on A100)

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
STEPS=500

echo "========================================"
echo "  NR-GRPO Experiments"
echo "========================================"

# ============================================================
# Exp 1: Clean GRPO (no noise) — upper bound
# ============================================================
echo "[1/8] Clean GRPO (no noise)"
python scripts/train.py --model $MODEL --noise-rate 0.0 --confidence none --max-steps $STEPS --output-dir checkpoints/clean_grpo

# ============================================================
# Exp 2-4: Vanilla GRPO under noise (baseline — should degrade)
# ============================================================
for NOISE in 0.1 0.2 0.3; do
    N=$(echo "$NOISE * 100" | bc | cut -d. -f1)
    echo "[Vanilla] noise=$NOISE"
    python scripts/train.py --model $MODEL --noise-rate $NOISE --confidence none --max-steps $STEPS --output-dir checkpoints/vanilla_n${N}
done

# ============================================================
# Exp 5-7: NR-GRPO under noise (ours — should be robust)
# ============================================================
for NOISE in 0.1 0.2 0.3; do
    N=$(echo "$NOISE * 100" | bc | cut -d. -f1)
    echo "[NR-GRPO] noise=$NOISE, consistency estimation"
    python scripts/train.py --model $MODEL --noise-rate $NOISE --confidence weighted --estimation consistency --max-steps $STEPS --output-dir checkpoints/nrgrpo_consistency_n${N}
done

# ============================================================
# Exp 8: NR-GRPO with majority estimation (ablation)
# ============================================================
echo "[NR-GRPO] noise=0.2, majority estimation"
python scripts/train.py --model $MODEL --noise-rate 0.2 --confidence weighted --estimation majority --max-steps $STEPS --output-dir checkpoints/nrgrpo_majority_n20

echo "========================================"
echo "  All experiments complete!"
echo "========================================"

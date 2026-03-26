#!/bin/bash
set -e
echo "============================================"
echo "  NR-GRPO 环境安装 (RunPod A100)"
echo "============================================"

pip install --upgrade pip
pip install torch transformers accelerate peft datasets trl
pip install sentencepiece protobuf tqdm

# Verify
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')

import transformers
print(f'Transformers: {transformers.__version__}')

from peft import LoraConfig
print(f'PEFT: OK')

from datasets import load_dataset
print(f'Datasets: OK')
"

echo "============================================"
echo "  安装完成!"
echo "  python scripts/train.py --noise-rate 0.1"
echo "============================================"

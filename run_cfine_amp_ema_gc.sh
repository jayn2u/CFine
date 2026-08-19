#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
uv run python train.py \
    --amp \
    --amp_dtype fp16 \
    --gradient_checkpointing \
    --ema \
    --ema_decay 0.999 \
    --wandb

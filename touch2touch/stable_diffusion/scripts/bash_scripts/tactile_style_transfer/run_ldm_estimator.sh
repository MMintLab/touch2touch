#!/usr/bin/env bash
set -euo pipefail

diffusion_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
repo_dir="$(cd -- "$diffusion_dir/../.." && pwd)"

export PYTHONPATH="$repo_dir:$diffusion_dir${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$diffusion_dir"

exec python scripts/tactile_style_transfer_estimator.py \
    --indir "$repo_dir/data/test/test_only/gelslims" \
    --outdir outputs/tactile_style_transfer_estimator_trial/ \
    --ddim_steps 200 \
    --config configs/tactile_style_transfer/gelslim2bubble.yaml \
    --ckpt "$repo_dir/checkpoints/diffusion/rot_flip/checkpoint.ckpt" \
    --n_samples 1 \
    --scale 7.5 \
    "$@"

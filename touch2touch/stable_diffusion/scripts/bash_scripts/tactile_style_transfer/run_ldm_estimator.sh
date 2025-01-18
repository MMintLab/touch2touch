export CUDA_VISIBLE_DEVICES=0
python scripts/tactile_style_transfer_estimator.py \
    --indir /home/samanta/touch2touch/data/test/test_only/gelslims \
    --outdir outputs/tactile_style_transfer_estimator_trial/ \
    --ddim_steps 200 \
    --config configs/tactile_style_transfer/gelslim2bubble.yaml \
    --ckpt /home/samanta/touch2touch/checkpoints/diffusion/rot_flip/checkpoint.ckpt\
    --max_sample -1 \
    --n_samples 1 \
    --scale 7.5

# --ckpt logs/2024-01-18T16-19-17_gelslim2bubble/checkpoints/epoch=000019.ckpt \
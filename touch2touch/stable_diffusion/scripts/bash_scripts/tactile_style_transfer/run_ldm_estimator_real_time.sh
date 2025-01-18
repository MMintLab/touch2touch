export CUDA_VISIBLE_DEVICES=0
value=`cat /home/samanta/save_temps/model_path.txt`
model_path_ckpt="$value/checkpoint.ckpt"
python scripts/tactile_style_transfer_estimator_real_time.py \
    --inpath /home/samanta/save_temps/save_temp_gelslim/ \
    --outdir outputs/tactile_style_transfer_estimator_debug/ \
    --ddim_steps 200 \
    --config configs/tactile_style_transfer/gelslim2bubble.yaml \
    --ckpt "$model_path_ckpt" \
    --max_sample -1 \
    --n_samples 1 \
    --scale 7.5
mode=$1
if [ $mode = "train" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python main.py --base configs/tactile_style_transfer/gelslim2bubble.yaml -t --gpus 0,1,2,3

elif [ $mode = "eval" ]; then
    python /home/samanta/stable-diffusion/scripts/gelslim2bubble.py \
        --outdir outputs_nobackup/gelslim2bubble_trial/ \
        --ddim_steps 200 \
        --config /home/samanta/stable-diffusion/configs/tactile_style_transfer/gelslim2bubble.yaml\
        --ckpt /home/samanta/stable-diffusion/logs/2024-01-18T16-19-17_gelslim2bubble/checkpoints/epoch=000029.ckpt \
        --max_sample -1 \
        --n_samples 10 \
        --scale 4
else
    echo "Invalid mode"
fi
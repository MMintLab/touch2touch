# touch2touch

This repository shows the main files for training a cross-modal tactile generation model using the VQ-VAE architecture and the main files to evaluate its performance.

# Dataset
We use Touch2Touch Dataset.

Paper: [https://www.arxiv.org/abs/2409.08269](https://www.arxiv.org/abs/2409.08269)

Dataset: https://drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=sharing

# Conda Environment Setup
Before running the code, please setup the right conda environment. You can download the .yml file from: https://drive.google.com/file/d/1vEvKdE5AxCES3c5P4aMf-l-FlOj6UBUd/view?usp=drive_link

```
conda env create -f touch2touch.yml
conda activate haptics_bl
```

# Checkpoints
Touch2Touch VQ-VAE Model Checkpoint: https://drive.google.com/file/d/10_HR54aKSUuF3hQPTY1zLOgYuBHMITch/view?usp=drive_link

# Train VQVAE Model
```
cd scripts
python train_vq_vae.py --model_type VQ-VAE-small --device cuda:0 --data cross_GB --dataset new_partial --mod 4 --random_sensor --color_jitter --rotation --flipping
```

# Evaluate VQVAE Model
```
python testing.py
```

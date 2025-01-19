# touch2touch
Today's touch sensors come in many shapes and sizes. This has made it challenging to develop general-purpose touch processing methods since models are generally tied to one specific sensor design. We address this problem by performing cross-modal prediction between touch sensors: given the tactile signal from one sensor, we use a generative model to estimate how the same physical contact would be perceived by another sensor. This allows us to apply sensor-specific methods to the generated signal. We implement this idea by training a diffusion model to translate between the popular GelSlim and Soft Bubble sensors. As a downstream task, we perform in-hand object pose estimation using GelSlim sensors while using an algorithm that operates only on Soft Bubble signals. 

This repository shows the main files for:
 * Training a cross-modal tactile generation model using the Stable Diffusion architecture (Coming soon).
 * Inference the tactile generation of Soft Bubble images from GelSlim images using our stable diffusion checkpoint.
 * Evaluating cross-modal tactile generation performance with diffusion model architectures.
 * Training a cross-modal tactile generation model using the VQ-VAE architecture.
 * Evaluating cross-modal tactile generation performance with VQ-VAE model architectures.

Project Webpage: https://www.mmintlab.com/research/touch2touch/

Paper: [https://www.arxiv.org/abs/2409.08269](https://www.arxiv.org/abs/2409.08269)

# Get Touch2Touch Dataset
* Download data folder from https:[//drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=sharing](https://drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=drive_link)
* Add to the repository folder recommended location:
```
<PATH_TO_REPO>
|---data
```

# Training using the Stable Diffusion architecture
Coming soon

# Inference using the Stable Diffusion architecture
## Conda Environment Setup
Before running the code, please setup the right conda environment. You can download the .yml file from: https://drive.google.com/file/d/1vEvKdE5AxCES3c5P4aMf-l-FlOj6UBUd/view?usp=drive_link

```
conda env create -f touch2touch.yml
conda activate haptics_bl
```

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

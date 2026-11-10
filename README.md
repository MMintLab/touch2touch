# touch2touch
Today's touch sensors come in many shapes and sizes. This has made it challenging to develop general-purpose touch processing methods since models are generally tied to one specific sensor design. We address this problem by performing cross-modal prediction between touch sensors: given the tactile signal from one sensor, we use a generative model to estimate how the same physical contact would be perceived by another sensor. This allows us to apply sensor-specific methods to the generated signal. We implement this idea by training a diffusion model to translate between the popular GelSlim and Soft Bubble sensors. As a downstream task, we perform in-hand object pose estimation using GelSlim sensors while using an algorithm that operates only on Soft Bubble signals. 

This repository includes the main files for:
* Training a cross-modal tactile generation model with Stable Diffusion (coming soon).
* Generating Soft Bubble images from GelSlim images using our Stable Diffusion checkpoint.
* Evaluating cross-modal tactile generation with diffusion models.
* Training a cross-modal tactile generation model with VQ-VAE.
* Evaluating cross-modal tactile generation with VQ-VAE.

Project Webpage: https://www.mmintlab.com/research/touch2touch/

Paper: [https://www.arxiv.org/abs/2409.08269](https://www.arxiv.org/abs/2409.08269)

# Get Touch2Touch Dataset
* Download data folder from https:[//drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=sharing](https://drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=drive_link)
* Add to the repository folder recommended location:
```
<PATH_TO_REPO>
|---data
```
# Get Stable Diffusion and VQ-VAE Checkpoints
* Download checkpoints folder from https:[//drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=sharing](https://drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=drive_link)
* Add to the repository folder recommended location:
```
<PATH_TO_REPO>
|---checkpoints
```

# Training using the Stable Diffusion architecture
Coming soon

# Inference using the Stable Diffusion architecture
## Conda Environment Setup
Before running the code, please setup the right conda environment. You can download the ldm.yml file from: [//drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=sharing](https://drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=drive_link)
```
conda env create -f ldm.yml
conda activate ldm
```

## Get all necessary checkpoints for inference using Stable Diffusion
* Download models folder from [stable_diffusion_add_ckpts](https://drive.google.com/drive/folders/1Dx9v6N3RFlPp12NNVr-bBvDKaY64HvQd?usp=drive_link) and add it to ```<PATH_TO_REPO>/touch2touch/stable_diffusion/models```
* Download ckpt folder from [stable_diffusion_add_ckpts](https://drive.google.com/drive/folders/1Dx9v6N3RFlPp12NNVr-bBvDKaY64HvQd?usp=drive_link) and add it to ```<PATH_TO_REPO>/touch2touch/stable_diffusion/ldm/modules/encoders/ckpt```

## Running inference

From the repository root, with the model environment active:

```bash
bash touch2touch/stable_diffusion/scripts/bash_scripts/tactile_style_transfer/run_ldm_estimator.sh
```

The launcher runs from `touch2touch/stable_diffusion` so the autoencoder path
in the YAML resolves correctly. The `data`, `checkpoints`, and
`touch2touch/stable_diffusion/models` directories can be symlinks to files
stored outside the repository.

For a smoke test, place one GelSlim `.pt` file in `data/test/smoke/gelslims`
and override the input directory and number of sampling steps:

```bash
bash touch2touch/stable_diffusion/scripts/bash_scripts/tactile_style_transfer/run_ldm_estimator.sh \
  --indir "$PWD/data/test/smoke/gelslims" --ddim_steps 10
```

# Conda Environment Setup
Before running the following scripts, please setup the right conda environment. You can download the touch2touch.yml file from: https://drive.google.com/file/d/1vEvKdE5AxCES3c5P4aMf-l-FlOj6UBUd/view?usp=drive_link
```
conda env create -f touch2touch.yml
conda activate haptics_bl
```
# Evaluating Stable Diffusion Model
```
python testing.py --model diffusion_norm --name rot_flip
```

# Training using VQVAE Model
```
cd scripts
python train_vq_vae.py --model_type VQ-VAE-small --device cuda:0 --data cross_GB --dataset new_partial --mod 4 --random_sensor --color_jitter --rotation --flipping
```

# Evaluating VQVAE Model
```
python testing.py --model vqvae --name rot_flip
```

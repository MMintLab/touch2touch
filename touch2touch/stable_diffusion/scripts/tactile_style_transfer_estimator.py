import argparse
import os
import os.path as osp
import sys
import glob
from statistics import mode
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from touch2touch.stable_diffusion.ldm.util import instantiate_from_config
from touch2touch.stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from touch2touch.stable_diffusion.ldm.models.diffusion.plms import PLMSSampler

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from ipdb import set_trace as st

# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

# f = open('/home/samanta/save_temps/model_path.txt', 'r')
# model_path = f.readline().strip()
# f.close()
# STATS = torch.load(os.path.join(model_path, 'training_stats.pt'))
# GELSLIM_MEAN = STATS['gelslim_mean']
# GELSLIM_STD = STATS['gelslim_std']
# GELSLIM_MEAN = torch.tensor([-0.0082, -0.0059, -0.0066])
# GELSLIM_STD = torch.tensor([0.0989, 0.0746, 0.0731])


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open(
            "assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept

def get_input(batch):
    x = batch
    if len(x.shape) == 3:
        x = x[..., None]
    # x = rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    return x


def map_back(image):
    # image = image * GELSLIM_STD[None, :, None, None].to(
        # image.device) + GELSLIM_MEAN[None, :, None, None].to(image.device)
    image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
    return image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=""
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2touch-ycb/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=3,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        # default=7.5,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/fredyang/fredyang/stable-diffusion/logs/2023-10-13T11-56-07_img2touch_cmc_ae/configs/2023-10-13T11-56-07-project.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/2023-10-13T11-56-07_img2touch_cmc_ae/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--max_sample",
        type=float,
        default=10000,
        help="maximum number of sample",
    )

    opt = parser.parse_args()
    return opt


class TouchEstimator:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.load_model()
        self.init_preprocessing()

    def init_preprocessing(self):
        model_path = self.opt.ckpt.split('checkpoint.ckpt')[0]
        STATS = torch.load(os.path.join(model_path, 'training_stats.pt'))
        GELSLIM_MEAN = STATS['gelslim_mean']
        GELSLIM_STD = STATS['gelslim_std']

        self.gelslim_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (256, 256), interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.Normalize(GELSLIM_MEAN, GELSLIM_STD),
        ])

    def load_model(self):
        config = OmegaConf.load(f"{self.opt.config}")
        model = load_model_from_config(config, f"{self.opt.ckpt}")
        model = model.to(self.device)
        if self.opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)
        self.model = model
        self.sampler = sampler

    def load_gelslim(self, gelslim_path, finger_index=0):
        # load gelslim
        gelslim = torch.load(gelslim_path)
        # import pdb; pdb.set_trace()
        gelslim = gelslim['gelslim'][finger_index].float() - \
            gelslim['gelslim_ref'][finger_index].float()
        gelslim = self.gelslim_transform(gelslim)
        gelslim = gelslim[None, ...]

        return gelslim

    def estimate(self, gelslim_dir, finger_index=0):
        gelslim_paths = sorted(os.listdir(gelslim_dir))

        os.makedirs(self.opt.outdir, exist_ok=True)
        outpath = self.opt.outdir
        full_images_path = os.path.join(outpath, "full_images")
        os.makedirs(full_images_path, exist_ok=True)

        start_code = None
        if self.opt.fixed_code:
            start_code = torch.randn(
                [self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f], device=self.device)

        precision_scope = autocast if self.opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for gelslim_path in tqdm(gelslim_paths, total=len(gelslim_paths)):
                        uc = None
                        gelslim = self.load_gelslim(
                            osp.join(gelslim_dir, gelslim_path), finger_index).to(self.device)
                        estimate_index = int(
                            gelslim_path.split('/')[-1].split('.')[0].split('_')[-1])
                        prompts = torch.cat(
                            [gelslim for _ in range(self.opt.n_samples)])
                        c = self.model.get_learned_conditioning(prompts)
                        shape = (self.model.channels,
                                 self.model.image_size, self.model.image_size)
                        samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                              conditioning=c,
                                                              batch_size=self.opt.n_samples,
                                                              shape=shape,
                                                              verbose=False,
                                                              unconditional_guidance_scale=self.opt.scale,
                                                              unconditional_conditioning=uc,
                                                              eta=self.opt.ddim_eta,
                                                              x_T=start_code)
                        x_samples_ddim = self.model.decode_first_stage(
                            samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim

                        x_checked_image_torch = torch.from_numpy(
                            x_checked_image).permute(0, 3, 1, 2)

                        if not self.opt.skip_save:
                            sample_path = os.path.join(
                                full_images_path, f"{estimate_index:04}")
                            os.makedirs(sample_path, exist_ok=True)
                            origin_images = [map_back(gelslim[:, :3])]
                            origin_name = ['gelslim']

                            for index, x_sample in enumerate(origin_images):
                                x_sample = 255. * \
                                    rearrange(
                                        x_sample.squeeze().cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(
                                    x_sample.astype(np.uint8))

                                # img.save(os.path.join(
                                #     sample_path, "{}.png".format(origin_name[index])))
                                
                                # img.save(os.path.join('/home/samanta/save_temps/save_temp_result_images','gelslim.png'))

                            model_path = self.opt.ckpt.split('checkpoint.ckpt')[0]
                            STATS = torch.load(os.path.join(model_path, 'training_stats.pt'))
                            BUBBLES_MIN, BUBBLES_MAX = STATS['bubbles_min'], STATS['bubbles_max']
                            
                            # save_path = "/home/samanta/save_temps/save_temp_bubbles/bubble_generated_data_0.pt"
                            root_folder = gelslim_dir.split("/test/")[0]
                            dataset_name = gelslim_dir.split("/test/")[1].split("/gelslims")[0]
                            diffusion_model = self.opt.ckpt.split('/')[-2]
                            if finger_index == 0:
                                finger_folder = "results_samples_right"
                            else:
                                finger_folder = "results_samples_left"
                            save_path = os.path.join(root_folder, "test", "diffusion_results", diffusion_model, finger_folder, dataset_name, gelslim_path)
                            
                            if not os.path.exists(os.path.dirname(save_path)):
                                os.makedirs(os.path.dirname(save_path))
                            
                            results_img_pt = {'prediction': (x_checked_image_torch + 1.) / 2. * (BUBBLES_MAX - BUBBLES_MIN) + BUBBLES_MIN}
                            torch.save(results_img_pt, save_path)
                            generated_image_count = 0
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * \
                                    rearrange(x_sample.cpu().numpy(),
                                              'c h w -> h w c')
                                img = Image.fromarray(
                                    x_sample.astype(np.uint8))
                                
                                # img.save(os.path.join(
                                #     sample_path, f"{generated_image_count:02}.png"))
                                # img.save(os.path.join('/home/samanta/save_temps/save_temp_result_images','bubbles.png'))
                                generated_image_count += 1


def main():
    opt = parse_args()
    seed_everything(opt.seed)
    touch_estimator = TouchEstimator(opt)

    gelslim_dir = opt.indir

    touch_estimator.estimate(gelslim_dir, 0)
    touch_estimator.estimate(gelslim_dir, 1)


if __name__ == "__main__":
    main()

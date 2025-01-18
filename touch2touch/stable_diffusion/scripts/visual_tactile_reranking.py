import os
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.regnet import regnet_y_800mf, regnet_y_400mf, regnet_y_1_6gf
import numpy as np
import PIL
from PIL import Image
import cv2
from typing import List
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import shutil
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pytorch_msssim import ssim

def calc_psnr(img1, img2):
    if len(img1.shape) == 3:
        mse = torch.mean((img1 - img2) ** 2)
        return 20. * torch.log10(1. / torch.sqrt(mse))
    elif len(img1.shape) == 4:
        mse = torch.mean((img1 - img2) ** 2,dim=[1,2,3])
        psnr = 20. * torch.log10(1. / torch.sqrt(mse))
        return torch.mean(psnr)
    
def calc_ssim(img1, img2):
    # img1: (N,3,H,W) a batch of RGB images
    # img2: (N,3,H,W)
    ssim_val = ssim(img1, img2, data_range=1.)
    return ssim_val

preprocess = {
    'to_tensor': T.Compose([
        T.ToTensor(),
    ]),
    'rgb': T.Compose([
        # T.CenterCrop(480),
        T.Resize(128),
    ]),
    'tac': T.Compose([
        # T.CenterCrop(480),
        T.Resize(128),
    ]), }


class Encoder(nn.Module):
    '''
    an image encoder which uses a resnet18 backbone
    '''

    def __init__(self, feature_dim=32,
                 model_type='resnet18',
                 **kwargs):
        super().__init__()
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Linear(512, feature_dim)
        elif model_type == 'regnet_400':
            self.resnet = regnet_y_400mf(pretrained=True)
            self.resnet.fc = nn.Linear(440, feature_dim)
        elif model_type == 'regnet_800':
            self.resnet = regnet_y_800mf(pretrained=True)
            self.resnet.fc = nn.Linear(784, feature_dim)
        elif model_type == 'regnet_1600':
            self.resnet = regnet_y_1_6gf(pretrained=True)
            self.resnet.fc = nn.Linear(888, feature_dim)

    def forward(self, batch: torch.tensor):
        '''
        takes in an image and returns the resnet18 features
        '''
        features = self.resnet(batch)
        feat_norm = torch.norm(features, dim=1)
        return features/feat_norm.view(features.shape[0], 1)

    def encode(self, im: np.ndarray):
        '''
        takes in an image and returns the resnet18 features
        '''
        im = im.unsqueeze(0)
        with torch.no_grad():
            features = self(im)
        return features

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(
            save_name, map_location=torch.device('cpu')))


rgb_encoder = Encoder(feature_dim=16, model_type='resnet18')
tac_encoder = Encoder(feature_dim=16, model_type='resnet18')
rgb_encoder.eval()
tac_encoder.eval()
rgb_ckpt_path = '/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/logs/ckpts/ResnetEncoder/rgb_enc.pth'
tac_ckpt_path = '/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/logs/ckpts/ResnetEncoder/tac_enc.pth'

rgb_encoder.load(rgb_ckpt_path)
tac_encoder.load(tac_ckpt_path)

rgb_encoder = rgb_encoder.cuda()
tac_encoder = tac_encoder.cuda()

output_dir = '/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_multiscale/full_images'
eval_dir = '/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_multiscale/eval'
os.makedirs(os.path.join(eval_dir,'gt'), exist_ok=True)
os.makedirs(os.path.join(eval_dir,'pred'), exist_ok=True)
os.makedirs(os.path.join(eval_dir,'input'), exist_ok=True)

n_samples = 16
reranking = True
# reranking = False

def process_sample(sample):
    sample_id = str(sample).zfill(5)
    sample_path = os.path.join(output_dir, instance, sample_id + '.png')
    sample_tac = Image.open(sample_path)
    sample_tac = preprocess['to_tensor'](sample_tac).cuda()
    sample_tac = preprocess['tac'](sample_tac)
    return sample_tac

gt_batch, pred_batch = [], []
instances = os.listdir(output_dir)
# instances.remove('eval')

# def save_pairs(instance):
#     if not os.path.exists(os.path.join(eval_dir,'gt',instance+'.png')):
#         rgb_path = os.path.join(output_dir, instance, 'input.png')
#         gt_path = os.path.join(output_dir, instance, 'reference_image.png')
#         max_sample_id = str(0).zfill(2)
#         # save
#         max_sample_path = os.path.join(output_dir, instance, max_sample_id+'.png')
#         gt_to_save = Image.open(gt_path).resize((128,128))
#         pred_to_save = Image.open(max_sample_path).resize((128,128))
#         input_to_save = Image.open(rgb_path).resize((128,128))
        
#         gt_to_save.save(os.path.join(eval_dir,'gt',instance+'.png'))
#         pred_to_save.save(os.path.join(eval_dir,'pred',instance+'.png'))
#         input_to_save.save(os.path.join(eval_dir,'input',instance+'.png'))
    
# with ThreadPoolExecutor() as executor:
#     results = list(tqdm(executor.map(save_pairs, instances), total=len(instances)))
CVTP = []
for instance in tqdm(instances):
    rgb_path = os.path.join(output_dir, instance, 'input.png')
    gt_path = os.path.join(output_dir, instance, 'reference_image.png')
    rgb = Image.open(rgb_path)
    rgb = preprocess['to_tensor'](rgb).cuda()
    rgb = preprocess['rgb'](rgb)
    rgb = rgb.unsqueeze(0)
    sample_tacs = []
    with ThreadPoolExecutor() as executor:
        sample_tacs = list(executor.map(process_sample, range(n_samples)))
    sample_tacs = torch.stack(sample_tacs)
    with torch.no_grad():
        rgb_feats = rgb_encoder(rgb)
        tac_feats = tac_encoder(sample_tacs)
    dot_prods = rgb_feats.mm(tac_feats.t()).cpu()  # [B x B]
    CVTP.append(torch.max(dot_prods).item())
    if reranking:
        max_sample_id = int(torch.argmax(dot_prods).cpu())
    else:
        # max_sample_id = random.randint(0, n_samples-1)
        max_sample_id = int(torch.argmin(dot_prods).cpu())
    max_sample_id = str(max_sample_id).zfill(5)
    # save
    max_sample_path = os.path.join(output_dir, instance, max_sample_id+'.png')
    gt_to_save = Image.open(gt_path).resize((128,128))
    pred_to_save = Image.open(max_sample_path).resize((128,128))
    input_to_save = Image.open(rgb_path).resize((128,128))
    
    gt_to_save.save(os.path.join(eval_dir,'gt',instance+'.png'))
    pred_to_save.save(os.path.join(eval_dir,'pred',instance+'.png'))
    input_to_save.save(os.path.join(eval_dir,'input',instance+'.png'))
    
    gt_batch.append(preprocess['to_tensor'](gt_to_save))
    pred_batch.append(preprocess['to_tensor'](pred_to_save))
    print('CVTP: ', np.mean(CVTP))
# import ipdb; ipdb.set_trace()
print('CVTP: ', np.mean(CVTP))
gt_batch = torch.stack(gt_batch)
pred_batch = torch.stack(pred_batch)
print('PSNR: ', calc_psnr(gt_batch, pred_batch), 'SSIM: ', calc_ssim(gt_batch, pred_batch))
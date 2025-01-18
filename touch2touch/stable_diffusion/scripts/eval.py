'''
Evaluation script. Evaluates generated images from running test.py
using various metrics
'''
import os
import sys
import argparse
from pathlib import Path
from shutil import copyfile

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

from util.util import yaml_config, get_most_free_gpu
import lpips
from skimage.metrics import structural_similarity as ssim
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default=None, 
                         help='Path to config file. Overrides all options')        
parser.add_argument('--name', type=str, default='', 
                         help='name of experiment')
parser.add_argument('--eval_fn', type=str, default='lpips', 
                         help='One of [lpips, mse, l1, ssim]')
parser.add_argument('--results_dir', type=str, default='./results/', 
                         help='Path to results dir')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                         help='Path to checkpoints dir')
parser.add_argument('--which_epoch', type=str, default='latest', 
                         help='Which epoch to evaluate')
parser.add_argument('--no_save', action='store_true', help='Save results?')
                                 
# Parse args
opt = parser.parse_args()
if opt.config is not None:
    opt = yaml_config(opt, opt.config)

# Autoset gpu
gpu_id = get_most_free_gpu()
torch.cuda.set_device(int(gpu_id))
print(f'\n========== Automatically using GPU {gpu_id} ==========\n')

# Get paths
results_dir = Path(opt.results_dir) / opt.name / f'test_{opt.which_epoch}' / 'images'
opt_path = Path(opt.checkpoints_dir) / opt.name / 'opt.txt'
loss_path = Path(opt.checkpoints_dir) / opt.name / 'loss_log.txt'

# If we couldn't find a results dir, ask if user wants to
# make a dummy record (useful for failed experiments)
if not results_dir.exists():
    x = input('Could not find results dir, make a dummy record? (y/n) ')
    
    if x == 'y':
        summary_dir = Path(opt.results_dir) / '_summary' / 'experiments' / opt.name
        summary_dir.mkdir(parents=True, exist_ok=True)
        copyfile(opt_path, summary_dir / 'opt.txt')
        copyfile(loss_path, summary_dir / 'loss_log.txt')
        
        print('\n========== Made dummy record ==========\n')
    else:
        print('\n========== Did NOT make dummy record ==========\n')
        
    sys.exit()
        

fnames = os.listdir(results_dir)

# Get predicted and target paths
generated_fnames = [f for f in fnames if 'generated' in f]
target_fnames = [f for f in fnames if 'target' in f]

# Sort to get canonical order
generated_fnames = sorted(generated_fnames)
target_fnames = sorted(target_fnames)

def pil2ten(im):
    # Converts PIL to pytorch tensor
    im = np.array(im) / 255.
    im = (im * 2) - 1
    return torch.tensor(im).permute(2,0,1).unsqueeze(0).float().cuda()

if opt.eval_fn == 'lpips':
    eval_fn = lpips.LPIPS(net='alex')
    eval_fn = eval_fn.cuda()
elif opt.eval_fn == 'mse':
    eval_fn = F.mse_loss
elif opt.eval_fn == 'l1':
    eval_fn = F.l1_loss
elif opt.eval_fn == 'ssim':
    eval_fn = ssim
else:
    assert True, "`eval_fn` must be one of [lpips, mse, l1, ssim]"

# Calculate scores for every generated image
scores = []
for f_g, f_t in tqdm(zip(generated_fnames, target_fnames), \
    total=len(target_fnames)):
        
    pred = Image.open(results_dir / f_g)
    target = Image.open(results_dir / f_t)
    
    pred = pil2ten(pred)
    target = pil2ten(target)

    with torch.no_grad():
        if opt.eval_fn == 'ssim':
            pred = pred.cpu().numpy()[0].transpose(1,2,0)
            target = target.cpu().numpy()[0].transpose(1,2,0)
        
            score = eval_fn(pred, target, multichannel=True)
        else:
            score = eval_fn(pred, target)

    scores.append(score.item())
    
score = np.mean(scores)
err = np.std(scores) * 1.96 / np.sqrt(len(scores))
print(f'Mean {opt.eval_fn} score: {score:.5f}')
print(f'Err {opt.eval_fn} score: {err:.5f}')

#####################
# Save summary info #
#####################

if not opt.no_save:
    summary_dir = Path(opt.results_dir) / '_summary' / 'experiments' / opt.name
    summary_dir.mkdir(parents=True, exist_ok=True)
    copyfile(opt_path, summary_dir / 'opt.txt')
    copyfile(loss_path, summary_dir / 'loss_log.txt')
    np.save(summary_dir / f'{opt.eval_fn}.npy', score)
    np.save(summary_dir / f'{opt.eval_fn}_err.npy', err)
    
    # write epoch
    result = [str(opt.which_epoch), '\t', str(score), '\n']
    with open(str(summary_dir)+'/epoch_score.txt','a') as f:
        f.writelines(result)
else:
    print('!!! Not Saving !!!')

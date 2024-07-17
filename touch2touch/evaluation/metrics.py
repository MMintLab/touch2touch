import torch
from touch2touch.evaluation.ssim import ssim, SSIM
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import data, img_as_float
import cv2
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import shutil
from torchvision import transforms

def psnr_metric(batch_gt, batch_pred):
    mse = torch.mean((batch_gt - batch_pred) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return 100
    max_pixel = torch.tensor(0.0225)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def ssim_metric(batch_gt, batch_pred, device='cpu'):
    ssim_result = SSIM(L=batch_pred.max() - batch_pred.min()).to(device)(batch_gt, batch_pred)
    return ssim_result

def fid_metric(batch_gt, batch_pred, batch_size=50, device='cpu', dims=2048, num_workers=0):
    # Save the images for FID calculation from the batch
    folder_path_gt = './fid_images/gt/'
    folder_path_pred = './fid_images/pred/'
    
    if not os.path.exists(folder_path_gt):
        os.makedirs(folder_path_gt)
    if not os.path.exists(folder_path_pred):
        os.makedirs(folder_path_pred)
    for i in range(batch_gt.shape[0]):
        # import pdb; pdb.set_trace()
        gt_img = batch_gt[i].cpu().detach().numpy()
        pred_img = batch_pred[i].cpu().detach().numpy()
        gt_img = cv2.normalize(gt_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        pred_img = cv2.normalize(pred_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gt_path = os.path.join(folder_path_gt, 'img_' + str(i) + '.jpg')
        pred_path = os.path.join(folder_path_pred, 'img_' + str(i) + '.jpg')
        cv2.imwrite(gt_path, gt_img)
        cv2.imwrite(pred_path, pred_img) #str(i) +
    fid_value = calculate_fid_given_paths([folder_path_gt, folder_path_pred], batch_size, device, dims, num_workers)
    
    if os.path.exists(os.path.dirname(os.path.dirname(folder_path_gt))):
        shutil.rmtree(os.path.dirname(os.path.dirname(folder_path_gt)))
    return fid_value

if __name__ == '__main__':
    data_path = '/home/samanta/Documents/UMICH/UMICH_Research_Codes/tactile_style_transfer/processed_data/bubbles/bubbles_testing_data_processed_flipped_2/bubble_style_transfer_dataset_bubbles_test_obj_hex_small_peg_seen/data_0.pt'
    bubble_data = torch.load(data_path)
    img = bubble_data['bubble_imprint'][0][0].numpy()
    # depth_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imwrite('depth.jpg', depth_normalized)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig("results.jpg")
    
    # import pdb; pdb.set_trace()
    resize = transforms.Resize((128, 128))
    # img = img_as_float(data.camera())
    rows, cols = img.shape

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    rng = np.random.default_rng()
    noise[rng.random(size=noise.shape) > 0.5] *= -1

    img_noise = img + noise
    img_const = img + abs(noise)

    img_tensor = resize(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
    img_noise_tensor = resize(torch.from_numpy(img_noise).unsqueeze(0).unsqueeze(0).float())
    img_const_tensor = resize(torch.from_numpy(img_const).unsqueeze(0).unsqueeze(0).float())

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8), sharex=True, sharey=True)
    ax = axes.ravel()

    # mse_none = F.mse_loss(img_tensor, img_tensor, reduction="mean")
    mse_none = psnr_metric(img_tensor, img_tensor)
    # ssim_none = ssim(img_tensor, img_tensor, L=img_tensor.max() - img_tensor.min())
    ssim_none = ssim_metric(img_tensor, img_tensor)
    fid_none = fid_metric(img_tensor.unsqueeze(0), img_tensor.unsqueeze(0))
    # import pdb; pdb.set_trace()

    # mse_noise = F.mse_loss(img_tensor, img_noise_tensor, reduction="mean")
    mse_noise = psnr_metric(img_tensor, img_noise_tensor)
    # ssim_noise = ssim(img_tensor, img_noise_tensor, L=img_noise_tensor.max() - img_noise_tensor.min())
    ssim_noise = ssim_metric(img_tensor, img_noise_tensor)
    fid_noise = fid_metric(img_tensor.unsqueeze(0), img_noise_tensor.unsqueeze(0))

    # mse_const = F.mse_loss(img_tensor, img_const_tensor, reduction="mean")
    mse_const = psnr_metric(img_tensor, img_const_tensor)
    # ssim_const = ssim(img_tensor, img_const_tensor, L=img_const_tensor.max() - img_const_tensor.min())
    ssim_const = ssim_metric(img_tensor, img_const_tensor)
    fid_const = fid_metric(img_tensor.unsqueeze(0), img_const_tensor.unsqueeze(0))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_xlabel(f"MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}, FID: {fid_none:.2f}")
    ax[0].set_title("Original image")

    ax[1].imshow(img_noise, cmap=plt.cm.gray)
    ax[1].set_xlabel(f"MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}, FID: {fid_noise:.2f}")
    ax[1].set_title("Image with noise")

    ax[2].imshow(img_const, cmap=plt.cm.gray)
    ax[2].set_xlabel(f"MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}, FID: {fid_const:.2f}")
    ax[2].set_title("Image plus constant")

    mse_none = F.mse_loss(img_tensor, img_tensor, reduction="mean")
    ssim_none = SSIM(L=img_tensor.max() - img_tensor.min())(img_tensor, img_tensor)

    mse_noise = F.mse_loss(img_tensor, img_noise_tensor, reduction="mean")
    ssim_noise = SSIM(L=img_noise_tensor.max() - img_noise_tensor.min())(img_tensor, img_noise_tensor)

    mse_const = F.mse_loss(img_tensor, img_const_tensor, reduction="mean")
    ssim_const = SSIM(L=img_const_tensor.max() - img_const_tensor.min())(img_tensor, img_const_tensor)

    ax[3].imshow(img, cmap=plt.cm.gray)
    ax[3].set_xlabel(f"MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}")
    ax[3].set_title("Original image")

    ax[4].imshow(img_noise, cmap=plt.cm.gray)
    ax[4].set_xlabel(f"MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}")
    ax[4].set_title("Image with noise")

    ax[5].imshow(img_const, cmap=plt.cm.gray)
    ax[5].set_xlabel(f"MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}")
    ax[5].set_title("Image plus constant")

    [ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[]) for i in range(len(axes))]

    plt.tight_layout()
    plt.savefig("results.png")

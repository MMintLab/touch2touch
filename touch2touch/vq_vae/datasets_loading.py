import torch
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset, Dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np
import glob as glob
import cv2
import os
import re
import matplotlib.pyplot as plt
# from ICP_dataset_cal import visualize_bubbles_icp_results, transformation_3D_from_2D, from_bubbles_to_pcd, tr_pointcloud, RANSAC_place_fit, transformation_from_quat, from_bubbles_to_pcd_create_mask, from_pcd_to_bubbles, project_masked_depth_image

import random
import open3d as o3d
import copy
# from bubbles_gelslim_spatial_relation import compensate_distortion

GELSLIM_MEAN = torch.tensor([-0.0082, -0.0059, -0.0066])
GELSLIM_STD = torch.tensor([0.0989, 0.0746, 0.0731])
BUBBLES_MEAN = torch.tensor([0.00382])
BUBBLES_STD = torch.tensor([0.00424])

DEFAULT_IMG_H = 320
DEFAULT_IMG_W = 427
DEFAULT_FX = 6500 # focal length
DEFAULT_CAMERA_MATRIX = np.array([[DEFAULT_FX, 0, DEFAULT_IMG_W//2],[0, DEFAULT_FX, DEFAULT_IMG_H//2],[0,0,1]])
DEFAULT_DISTORTION = np.array([[ 4.74972785e+02, -2.42689656e+05,  2.64865296e-02, 1.92364950e-02, -3.49364916e+02]])

def compensate_distortion(img, distortion=DEFAULT_DISTORTION, matrix=DEFAULT_CAMERA_MATRIX, newcameramtx=None, roi=None):
    h, w = img.shape[:2]
    if newcameramtx is None:
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))
    # correct distortion:
    corrected_img = cv2.undistort(img, matrix, distortion, None, newcameramtx)
    if roi is not None:
        # crop the image to get just the region of interest
        corrected_img = corrected_img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    return corrected_img

class RandomFlip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num = random.choice([0, 1, 2])

    def __call__(self, bubbles, gelslim):
        # no change if zero

        # do either horizontal or vertical
        if self.num == 1:
            choose = random.choice([0, 1])
            if choose:
                bubbles = transforms.functional.hflip(bubbles)
                gelslim = transforms.functional.hflip(gelslim)
            
            else:
                bubbles = transforms.functional.vflip(bubbles)
                gelslim = transforms.functional.vflip(gelslim)
                
        # do both vertical and horizontal in different order
        elif self.num == 2:
            choose = random.choice([0, 1])

            if choose:
                bubbles = transforms.functional.hflip(bubbles)
                gelslim = transforms.functional.hflip(gelslim)
                bubbles = transforms.functional.vflip(bubbles)
                gelslim = transforms.functional.vflip(gelslim)
            
            else:
                bubbles = transforms.functional.vflip(bubbles)
                gelslim = transforms.functional.vflip(gelslim)
                bubbles = transforms.functional.hflip(bubbles)
                gelslim = transforms.functional.hflip(gelslim)
        
        return bubbles, gelslim

class RandomAffine(torch.nn.Module):
    def __init__(self, degrees, minT, maxT):
        angleList = range(-degrees, degrees, 5)
        translationList = list(np.arange(minT, maxT, 0.1))
        scaleList = list(np.arange(0.5, 1.5, 0.1))
        self.angleChosen = random.choice(angleList)
        self.translationChosen = random.choice(translationList)
        self.scale = random.choice(scaleList)

    def __call__(self, bubbles, gelslim):
        t_bubbles = transforms.functional.affine(bubbles, self.angleChosen, [self.translationChosen / 4, self.translationChosen / 4], self.scale, 0, fill=0.5)
        t_gelslim = transforms.functional.affine(gelslim, self.angleChosen, [self.translationChosen, self.translationChosen], self.scale, 0, fill=0.5)
        return [t_bubbles, t_gelslim]

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x^i and x^j, which we consider as a positive pair.
    """

    def __init__(self, size=128):                
        gelslim_color_jitter = transforms.ColorJitter(
           0.8, 0.8, 0.8, 0.2
        )
        
        bubble_color_jitter = transforms.ColorJitter(
           0.0, 0.0, 0.0, 0.3
        )
        self.gelslim_transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(size=size),
                transforms.RandomApply([gelslim_color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.3),
            ]
        )

        self.bubble_transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(size=size),
                transforms.RandomApply([bubble_color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

    def __call__(self, sensor_img, is_gelslim=1):
        if is_gelslim:
            return self.gelslim_transform(sensor_img)
        else:
            return self.bubble_transform(sensor_img)        


class filterHighFreq(object):
    """Filter high frequency features from images"""
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, gelslim_diff):
        # Remove High Frecuencies
        device = gelslim_diff.device
        gelslim_diff_r = gelslim_diff[0]
        gelslim_diff_l = gelslim_diff[1]
        f_l = np.fft.fft2(gelslim_diff_l.detach().cpu().numpy())
        f_r = np.fft.fft2(gelslim_diff_r.detach().cpu().numpy())
        fshift_l = np.fft.fftshift(f_l) ## shift for centering 0.0 (x,y)
        fshift_r = np.fft.fftshift(f_r) ## shift for centering 0.0 (x,y)

        rows = gelslim_diff_l.shape[1] #taking the size of the image
        cols = gelslim_diff_l.shape[2]
        crow, ccol = rows//2, cols//2
        shift = self.shift
        original_l = np.copy(fshift_l)
        original_r = np.copy(fshift_r)
        fshift_l[:, crow-shift:crow+shift, ccol-shift:ccol+shift] = 0
        fshift_r[:, crow-shift:crow+shift, ccol-shift:ccol+shift] = 0
        f_ishift_l= np.fft.ifftshift(original_l - fshift_l)
        f_ishift_r= np.fft.ifftshift(original_r - fshift_r)

        gelslim_diff_l = torch.from_numpy(np.fft.ifft2(f_ishift_l))
        gelslim_diff_r = torch.from_numpy(np.fft.ifft2(f_ishift_r))
        gelslim_diff_l = gelslim_diff_l.to(torch.float32)
        gelslim_diff_r = gelslim_diff_r.to(torch.float32)

        gelslim_diff = torch.cat((gelslim_diff_r.unsqueeze(0), gelslim_diff_l.unsqueeze(0)), dim=0).to(device = device)
        return gelslim_diff

class threshZeroed(object):
    """Zeros out part of image with close values to zero"""
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, gelslim_diff):
        gelslim_diff[gelslim_diff < self.thresh] = 0
        return gelslim_diff
    
def logging_image_grid(images, captions, ncol=7, normalize = True):
    if not normalize:
        norm_text = "_not_normalized"
    else:
        norm_text = ""

    grids = [make_grid(img, nrow=ncol,padding=1, normalize=normalize, scale_each=True, pad_value=1) for img in images]
    for grid, caption in zip(grids, captions):
        plt.imshow(np.asarray(grid.permute((1,2,0))))
        plt.title(caption)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
    return

def sort_order(filename):
    return int(re.findall(r'\d+', filename)[-1])

class unnormalize(object):
    """Zeros out part of image with close values to zero"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, normalized_image):
        self.mean = self.mean.to(normalized_image.device)
        self.std = self.std.to(normalized_image.device)
        image = normalized_image*self.std + self.mean
        return image
    
def datasets_definiton(name):
    if name == '1-tool':
        train_tools = ['pattern_01_2_lines_angle_1']
        test_tools = ['pattern_31_rod']
    elif name == 'train-tools':
        train_tools = [ 'pattern_01_2_lines_angle_1',
                        'pattern_03_2_lines_angle_3',
                        'pattern_04_3_lines_angle_1',
                        'pattern_06_5_lines_angle_1',
                        'pattern_07_curves_degree_30_radios_10',
                        'pattern_09_curves_degree_120_radios_10',
                        'pattern_10_curves_degree_150_radios_10',
                        'pattern_11_curves_degree_30_radios_20',
                        'pattern_12_curves_degree_45_radios_20',
                        'pattern_14_curves_degree_150_radios_20',
                        'pattern_15_circle',
                        'pattern_17_ellipse_2',
                        'pattern_18_hex_1',
                        'pattern_20_hex_3',
                        'pattern_31_rod'
                     ]

        test_tools = ['pattern_02_2_lines_angle_2',
                      'pattern_05_3_lines_angle_2',
                      'pattern_08_curves_degree_45_radios_10',
                    #   'pattern_13_curves_degree_120_radios_20',
                    #   'pattern_16_ellipse_1',
                    #   'pattern_19_hex_2',
                    #   'test_obj_hex_small_peg_seen',
                    #   'test_obj_square_small_peg_seen',
                    #   'test_obj_tilted_square_small_peg_seen'
                    ]
                      
    elif name == 'train-test-tools':
        train_tools = [ 'pattern_01_2_lines_angle_1',
                        'pattern_02_2_lines_angle_2',
                        'pattern_03_2_lines_angle_3',
                        'pattern_04_3_lines_angle_1',
                        'pattern_05_3_lines_angle_2',
                        'pattern_06_5_lines_angle_1',
                        'pattern_07_curves_degree_30_radios_10',
                        'pattern_08_curves_degree_45_radios_10',
                        'pattern_09_curves_degree_120_radios_10',
                        'pattern_10_curves_degree_150_radios_10',
                        'pattern_11_curves_degree_30_radios_20',
                        'pattern_12_curves_degree_45_radios_20',
                        'pattern_13_curves_degree_120_radios_20',
                        'pattern_14_curves_degree_150_radios_20',
                        'pattern_15_circle',
                        'pattern_16_ellipse_1',
                        'pattern_17_ellipse_2',
                        'pattern_18_hex_1',
                        'pattern_19_hex_2',
                        'pattern_20_hex_3',
                        'pattern_31_rod']
        test_tools = ['test_obj_hex_small_peg_seen',
                      'test_obj_square_small_peg_seen',
                      'test_obj_tilted_square_small_peg_seen']
        
    elif name == 'new-all-data':
        train_tools = ['pattern_01_2_lines_angle_1',
               'pattern_02_2_lines_angle_2',
               'pattern_03_2_lines_angle_3',
               'pattern_04_3_lines_angle_1',
               'pattern_05_3_lines_angle_2',
               'pattern_06_5_lines_angle_1',
               'pattern_31_rod',
               'pattern_32',
               'pattern_33',
               'pattern_35',
               'pattern_36',
               'pattern_37']
        test_tools = []
    
    elif name == 'new-partial-data':
        train_tools = ['pattern_01_2_lines_angle_1',
               'pattern_02_2_lines_angle_2',
               'pattern_03_2_lines_angle_3',
               'pattern_04_3_lines_angle_1',
               'pattern_06_5_lines_angle_1',
               'pattern_31_rod',
               'pattern_32',
               'pattern_33',
               'pattern_37',]
        test_tools = ['pattern_05_3_lines_angle_2','pattern_35', 'pattern_36']

    elif name == 'debug':
        train_tools = ['pattern_01_2_lines_angle_1']
        test_tools = ['test_obj_hex_small_peg_seen',
                      'test_obj_square_small_peg_seen',
                      'test_obj_tilted_square_small_peg_seen']
    else:
        raise ValueError(f'Unknown dataset: {name})')
    
    return train_tools, test_tools

def data_symmetry_train(images, labels, mod = '1'):
    '''
    Inputs:
        images: tensor of shape (batch_size, 2, 3, 128, 128) or (batch_size, 2, 1, 128, 128)
        labels: tensor of shape (batch_size, 2, 3, 128, 128) or (batch_size, 2, 1, 128, 128)
        mod: '0' for only left sensor, '1' for both sensors, '2' for both sensors in grayscale, '6' for both sensors in grayscale and repeated 3 times, 'combined' for both sensors in grayscale and repeated 3 times and concatenated
    '''
    if mod == '0':
        images = images[:,1]
        labels = labels[:,1]
        '''
        Output: images and labels from only the left sensor (batch_size, 3, 128, 128) and (batch_size, 1, 128, 128)
        '''
    elif mod == '1':
        images = torch.cat((images[:,0], images[:,1]), dim=0)
        labels = torch.cat((labels[:,0], labels[:,1]), dim=0)
        '''
        Output: images and labels from both sensors concatenated as extended batch (2*batch_size, 3, 128, 128) and (2*batch_size, 1, 128, 128)
        '''
    elif mod == '2':
        to_grayscale = transforms.Grayscale()
        if images.shape[2] == 3:
            images = to_grayscale(images)
        if labels.shape[2] == 3:
            labels = to_grayscale(labels)
        images = torch.cat((images[:,0], images[:,1]), dim=1)
        labels = torch.cat((labels[:,0], labels[:,1]), dim=1)
        '''
        Output: images and labels from both sensors concatenated along the channels dimension in gray scale (2*batch_size, 2, 128, 128) and (2*batch_size, 2, 128, 128)
        '''
    elif mod == 'combined':
        if images.shape[2] == 1:
            images = images.repeat(1, 1, 3, 1, 1)
        if labels.shape[2] == 1:
            labels = labels.repeat(1, 1, 3, 1, 1)
        images = torch.cat((images[:,0], labels[:,0], images[:,1], labels[:,1]), dim=0)
        labels = images
        '''
        Output: images and labels from both sensors concatenated along the batch dimension (4*batch_size, 3, 128, 128) and (4*batch_size, 3, 128, 128)
        '''

    return images, labels

def data_symmetry_viz(input, mod = '1'):
    # TODO: Check left and right indices
    '''
    Inputs:
        input: tensor of shape corresponding to data_symmetry_train output
        mod: '0' for only left sensor, '1' for both sensors, '2' for both sensors in grayscale, '6' for both sensors repeated 3 times i necessary, 'combined' for both sensors in grayscale and repeated 3 times and concatenated
    '''
    if mod == '0':
        return input
    elif mod == '1':
        input_l, input_r= torch.split(input, int(input.shape[0]/2), dim=0)
        input = torch.cat([input_l, input_r], dim = 2)
    elif mod == '2':
        input = torch.cat([input[:,0].unsqueeze(dim=1), input[:,1].unsqueeze(dim=1)], dim = 2)
    elif mod == '6':
        input = torch.cat([input[:,:3], input[:,3:]], dim = 2)
    elif mod == 'combined':
        images_l, labels_l, images_r, labels_r= torch.split(input, int(input.shape[0]/4), dim=0)
        images = torch.cat([images_l, images_r], dim = 2)
        labels = torch.cat([labels_l, labels_r], dim = 2)
        input = torch.cat([images, labels], dim = 0)

    return input

# DATASET CREATION
class TactileTransferAllInfoAugment(Dataset):
    def __init__(
                    self, 
                    root_dir_bubbles, 
                    root_dir_gelslim, 
                    device, 
                    bubbles_transform=None, 
                    gelslim_transform=None, 
                    data = 'cross_GB', 
                    grayscale = False, 
                    single = True, 
                    cropped = False, 
                    distortion = True, 
                    random_sensor = False, 
                    color_jitter = False, 
                    rotation = False, 
                    flipping = False
                    ):
        """
        Args:
            root_dir: the directory of the dataset
            transform: pytorch transformations.
        """

        self.bubbles_transform = bubbles_transform
        self.gelslim_transform = gelslim_transform
        self.bubbles_files = sorted(glob.glob(os.path.join(root_dir_bubbles, '*.pt')), key=sort_order)

        if not distortion:
            root_dir_gelslim = root_dir_gelslim.replace('gelslims/', 'gelslims_no_distortion/')
        self.gelslim_files = sorted(glob.glob(os.path.join(root_dir_gelslim, '*.pt')), key=sort_order)

        self.device = device
        self.grayscale = grayscale
        self.single = single
        self.data = data
        self.cropped = cropped
        self.bubbles_rotate = transforms.RandomRotation((180,180))

        self.color_jitter = transforms.Compose(
                                                [transforms.RandomApply(
                                                                        [transforms.ColorJitter(
                                                                            brightness=0.02,
                                                                            contrast=0.02,
                                                                            saturation=0.02)], p=0.5),
                                                transforms.RandomApply(
                                                    [transforms.GaussianBlur(5, sigma=(0.5, 1))], p=0.5)])
        
        self.random_sensor = random_sensor
        self.rotation = rotation
        self.flipping = flipping
        self.color_jittering = color_jitter

    def __len__(self):
        return len(self.bubbles_files)

    def __getitem__(self, idx):
        # BUBBLES
        bubbles_data = torch.load(self.bubbles_files[idx], map_location=self.device) 
        bubbles_img = bubbles_data['bubble_imprint']
        bubbles_img = self.bubbles_rotate(bubbles_img)
        bubbles_ref = bubbles_data['bubble_depth_ref']

        bubbles_min = bubbles_img.min()
        bubbles_max = bubbles_img.max()
        bubbles_mean = bubbles_img.mean()
        bubbles_std = bubbles_img.std()

        if self.cropped:
            BC, BH, BW = bubbles_img[0].shape
            GH = 53
            GW = 71
            offset_H = -4
            offset_W = 10
            # bubbles_img = self.bubbles_rotate(bubbles_img)
            bubbles_img = bubbles_img[:,:,int(BH/2) - int(GH/2) - 1 + offset_H:int(BH/2) + int(GH/2) + offset_H, int(BW/2) - int(GW/2) - 1 + offset_W :int(BW/2) + int(GW/2) + offset_W]
        # bubbles_diff = (bubbles_data['bubble_depth_ref'] - bubbles_data['bubble_imprint'])

        if self.bubbles_transform:
            bubbles_img = self.bubbles_transform(bubbles_img)

        # GELSLIM
        gelslim_data = torch.load(self.gelslim_files[idx], map_location=self.device)
        gelslim_img = gelslim_data['gelslim']
        gelslim_ref = gelslim_data['gelslim_ref']

        gelslim_diff = gelslim_img - gelslim_ref

        gelslim_img = gelslim_img / 255
        gelslim_ref = gelslim_ref / 255
        gelslim_diff = gelslim_diff / 255

        if self.gelslim_transform:
            gelslim_img = self.gelslim_transform(gelslim_img)
            gelslim_ref = self.gelslim_transform(gelslim_ref)
            gelslim_diff = self.gelslim_transform(gelslim_diff)

        gelslim_mean = gelslim_diff.mean(dim=[0,2,3])
        gelslim_std = gelslim_diff.std(dim=[0,2,3])
        
        # GRAYSCALE
        if self.grayscale:
            # Grayscale gelslim diff image if needed
            gelslim_diff = transforms.functional.rgb_to_grayscale(gelslim_diff, num_output_channels=1)
            repeat = 1
        else:
            repeat = 3

        # RANDOM SELECTION OF RIGHT AND LEFT SENSOR
        if self.random_sensor:
            finger_index = random.randint(0, 1)
            gelslim_diff = gelslim_diff[finger_index]
            bubbles_img = bubbles_img[finger_index]
            self.single = True

        # ADDITIONAL AUGMENTATIONS
        if self.color_jittering:
            if self.data == 'cross_BG':
                bubbles_img = self.color_jitter(bubbles_img)
            elif self.data == 'cross_GB':
                gelslim_diff = self.color_jitter(gelslim_diff)
            else:
                bubbles_img = self.color_jitter(bubbles_img)
                gelslim_diff = self.color_jitter(gelslim_diff)
        
        if self.rotation:
            rotation = random.choice([0, 90, 180, 270])
            gelslim_diff = transforms.functional.rotate(gelslim_diff, rotation)
            bubbles_img = transforms.functional.rotate(bubbles_img, rotation)
        
        if self.flipping:
            if random.random() > 0.5:
                gelslim_diff = transforms.functional.hflip(gelslim_diff)
                bubbles_img = transforms.functional.hflip(bubbles_img)
            if random.random() > 0.5:
                gelslim_diff = transforms.functional.vflip(gelslim_diff)
                bubbles_img = transforms.functional.vflip(bubbles_img)

        # ADDITIONAL INFORMATION
        if self.single:
            if not self.color_jittering:
                gelslim_diff = gelslim_diff[1]
                bubbles_img = bubbles_img[1]

            info = {
                'bubbles_data': {
                                'bubble_depth_ref': bubbles_ref[1],
                                'theta': bubbles_data['theta'],
                                'K': bubbles_data['K'][1],
                                'bubbles_tr_quat': bubbles_data['bubbles_tr_quat'][1],
                                'tool_tr_quat': bubbles_data['tool_tr_quat'],
                                'finger_tr_quat': bubbles_data['finger_tr_quat'][1],
                                'grasp_frame_quat': bubbles_data['grasp_frame_quat'],
                                'x': bubbles_data['x'],
                                'y': bubbles_data['y'],
                                'img_mean': bubbles_mean,
                                'img_std': bubbles_std,
                                'min': bubbles_min,
                                'max': bubbles_max,
                                },
                'gelslim_data': {
                                'gelslim_ref': gelslim_ref[1],
                                'theta': gelslim_data['theta'],
                                'gelslim_tr_quat': gelslim_data['gelslim_tr_quat'][1],
                                'tool_tr_quat': gelslim_data['tool_tr_quat'],
                                'finger_tr_quat': gelslim_data['finger_tr_quat'][1],
                                'grasp_frame_quat': gelslim_data['grasp_frame_quat'],
                                'x': gelslim_data['x'],
                                'y': gelslim_data['y'],
                                'img_mean': gelslim_mean,
                                'img_std': gelslim_std,
                                }}
        else:
            info = {
                    'bubbles_data': {
                                    'bubble_depth_ref': bubbles_ref,
                                    'theta': bubbles_data['theta'],
                                    'K': bubbles_data['K'],
                                    'bubbles_tr_quat': bubbles_data['bubbles_tr_quat'],
                                    'tool_tr_quat': bubbles_data['tool_tr_quat'],
                                    'finger_tr_quat': bubbles_data['finger_tr_quat'],
                                    'grasp_frame_quat': bubbles_data['grasp_frame_quat'],
                                    'x': bubbles_data['x'],
                                    'y': bubbles_data['y'],
                                    'img_mean': bubbles_mean,
                                    'img_std': bubbles_std,
                                    'min': bubbles_min,
                                    'max': bubbles_max,
                                    },
                    'gelslim_data': {
                                    'gelslim_ref': gelslim_ref,
                                    'theta': gelslim_data['theta'],
                                    'gelslim_tr_quat': gelslim_data['gelslim_tr_quat'],
                                    'tool_tr_quat': gelslim_data['tool_tr_quat'],
                                    'finger_tr_quat': gelslim_data['finger_tr_quat'],
                                    'grasp_frame_quat': gelslim_data['grasp_frame_quat'],
                                    'x': gelslim_data['x'],
                                    'y': gelslim_data['y'],
                                    'img_mean': gelslim_mean,
                                    'img_std': gelslim_std,
                                    }}
        
        if self.data == 'cross_GB':
            return gelslim_diff, bubbles_img, info
        elif self.data == 'bubbles':
            return bubbles_img, bubbles_img, info
        elif self.data == 'gelslim':
            return gelslim_diff, gelslim_diff, info
        elif self.data == 'cross_BG':
            return bubbles_img, gelslim_diff, info
        else:
            raise ValueError('data must be either cross_GB, bubbles, gelslim, or cross_BG')
        return
        
class TactileTransferDiffusion(Dataset):
    def __init__(self, tool_name,  root_dir_bubbles, root_dir_gelslim, diffusion_results_path, device, diffusion_idx = 0, bubbles_transform=None, gelslim_transform=None, dataset_norm = False):
        """
        Args:
            root_dir: the directory of the dataset
            transform: pytorch transformations.
        """
        self.bubbles_transform = bubbles_transform
        self.gelslim_transform = gelslim_transform
        self.bubbles_files = sorted(glob.glob(os.path.join(root_dir_bubbles, '*.pt')), key=sort_order)
        self.gelslim_files = sorted(glob.glob(os.path.join(root_dir_gelslim, '*.pt')), key=sort_order)
        self.device = device
        self.diffusion_idx = diffusion_idx

        # diffusion_results_files = []
        # for filename in os.listdir(diffusion_results_path):
        #     # Check if the file is a .pt file
        #     if filename.endswith('.pt'):
        #         if tool_name in filename:
        #             diffusion_results_files.append(os.path.join(diffusion_results_path, filename))

        self.diffusion_results_path = diffusion_results_path
        self.diffusion_results_files = sorted(os.listdir(diffusion_results_path), key=sort_order)
        # self.diffusion_results_files = sorted(diffusion_results_files, key=sort_order)
        # import pdb; pdb.set_trace()
        print(len(self.diffusion_results_files))
        self.bubbles_rotate = transforms.RandomRotation((180,180))
        
        self.dataset_norm = dataset_norm
        gt_mean = torch.tensor([0.0126])
        gt_std = torch.tensor([0.0015])
        train_mean = torch.tensor([0.0027])
        train_std =  torch.tensor([0.0033])
        self.diffusion_pred_norm = transforms.Compose([transforms.Normalize(gt_mean, gt_std),
                                                       unnormalize(train_mean, train_std)
                                                      ])
        

    def __len__(self):
        return len(self.bubbles_files)

    def __getitem__(self, idx):
        # BUBBLES
        bubbles_data = torch.load(self.bubbles_files[idx], map_location=self.device) 
        bubbles_img = bubbles_data['bubble_imprint']
        bubbles_img = self.bubbles_rotate(bubbles_img)
        bubbles_ref = bubbles_data['bubble_depth_ref']
        # bubbles_img_diffusion = transforms.functional.rgb_to_grayscale(torch.load(os.path.join(self.diffusion_results_path, self.diffusion_results_files[idx]), map_location=self.device)['gt'])
        bubbles_img_prediction = transforms.functional.rgb_to_grayscale(torch.load(os.path.join(self.diffusion_results_path, self.diffusion_results_files[idx]), map_location=self.device)['prediction'][self.diffusion_idx].float()).unsqueeze(0)
        # bubbles_img_diffusion_prev = torch.load(self.diffusion_results_files[idx], map_location=self.device)['gt']
        # bubbles_img_diffusion = ((bubbles_img_diffusion_prev[:,0] + bubbles_img_diffusion_prev[:,1] + bubbles_img_diffusion_prev[:,2])/3).unsqueeze(0)
        # bubbles_img_pred_prev = torch.load(self.diffusion_results_files[idx], map_location=self.device)['prediction'][self.diffusion_idx].float().unsqueeze(0)
        # bubbles_img_prediction = ((bubbles_img_pred_prev[:,0] + bubbles_img_pred_prev[:,1] + bubbles_img_pred_prev[:,2])/3).unsqueeze(0)
        # import pdb; pdb.set_trace()
        
        if self.dataset_norm:
            bubbles_img_prediction = self.diffusion_pred_norm(bubbles_img_prediction)

        bubbles_gt_min = bubbles_img.min()
        bubbles_gt_max = bubbles_img.max()
        bubbles_gt_mean = bubbles_img.mean()
        bubbles_gt_std = bubbles_img.std()
        # bubbles_gt_diff_min = bubbles_img.min()
        # bubbles_gt_diff_max = bubbles_img.max()
        # bubbles_gt_diff_mean = bubbles_img_diffusion.mean()
        # bubbles_gt_diff_std = bubbles_img_diffusion.std()
        bubbles_pred_min = bubbles_img_prediction.min()
        bubbles_pred_max = bubbles_img_prediction.max()
        bubbles_pred_mean = bubbles_img_prediction.mean()
        bubbles_pred_std = bubbles_img_prediction.std()

        if self.bubbles_transform:
            bubbles_img = self.bubbles_transform(bubbles_img)
            # bubbles_img_diffusion = self.bubbles_transform(bubbles_img_diffusion)
            bubbles_img_prediction = self.bubbles_transform(bubbles_img_prediction)
        
        # bubbles_img_diffusion = torch.cat([bubbles_img_diffusion, torch.flip(bubbles_img_diffusion, [2])], dim = 0)
        bubbles_img_prediction = torch.cat([bubbles_img_prediction, torch.flip(bubbles_img_prediction, [2])], dim = 0)

        # GELSLIM
        gelslim_data = torch.load(self.gelslim_files[idx], map_location=self.device)
        gelslim_img = gelslim_data['gelslim']
        gelslim_ref = gelslim_data['gelslim_ref']
        gelslim_diff = gelslim_img - gelslim_ref

        gelslim_img = gelslim_img / 255
        gelslim_ref = gelslim_ref / 255
        gelslim_diff = gelslim_diff / 255

        # gelslim_diff_diffusion = torch.load(self.diffusion_results_files[idx], map_location=self.device)['input']

        if self.gelslim_transform:
            gelslim_img = self.gelslim_transform(gelslim_img)
            gelslim_ref = self.gelslim_transform(gelslim_ref)
            gelslim_diff = self.gelslim_transform(gelslim_diff)
        #     gelslim_diff_diffusion = self.gelslim_transform(gelslim_diff_diffusion)
        # gelslim_diff_diffusion = torch.cat([gelslim_diff_diffusion, torch.flip(gelslim_diff_diffusion, [2])], dim = 0)
        # ADDITIONAL INFORMATION
        info = {
            'bubbles_data': {
                            'bubble_depth_ref': bubbles_ref,
                            'theta': bubbles_data['theta'],
                            'K': bubbles_data['K'],
                            'bubbles_tr_quat': bubbles_data['bubbles_tr_quat'],
                            'tool_tr_quat': bubbles_data['tool_tr_quat'],
                            'finger_tr_quat': bubbles_data['finger_tr_quat'],
                            'grasp_frame_quat': bubbles_data['grasp_frame_quat'],
                            'x': bubbles_data['x'],
                            'y': bubbles_data['y'],
                            'bubble_img_raw': bubbles_img,
                            'bubbles_metrics': {
                                                'gt_min': bubbles_gt_min,
                                                'gt_max': bubbles_gt_max,
                                                'gt_mean': bubbles_gt_mean,
                                                'gt_std': bubbles_gt_std,
                                                # 'gt_diff_min': bubbles_gt_diff_min,
                                                # 'gt_diff_max': bubbles_gt_diff_max,
                                                # 'gt_diff_mean': bubbles_gt_diff_mean,
                                                # 'gt_diff_std': bubbles_gt_diff_std,
                                                'pred_min': bubbles_pred_min,
                                                'pred_max': bubbles_pred_max,
                                                'pred_mean': bubbles_pred_mean,
                                                'pred_std': bubbles_pred_std,
                                                },
                            },
            'gelslim_data': {
                            'gelslim_ref': gelslim_ref,
                            'theta': gelslim_data['theta'],
                            'gelslim_tr_quat': gelslim_data['gelslim_tr_quat'],
                            'tool_tr_quat': gelslim_data['tool_tr_quat'],
                            'finger_tr_quat': gelslim_data['finger_tr_quat'],
                            'grasp_frame_quat': gelslim_data['grasp_frame_quat'],
                            'x': gelslim_data['x'],
                            'y': gelslim_data['y'],
                            'gelslim_img_raw': gelslim_diff,
                            }}
        
        return bubbles_img_prediction, info #, gelslim_diff_diffusion, bubbles_img_diffusion
    
class TST_saving(Dataset):
    def __init__(self, root_dir_bubbles, root_dir_gelslim, device='cpu'):
        """
        Args:
            root_dir: the directory of the dataset
            transform: pytorch transformations.
        """
        self.bubbles_files = sorted(glob.glob(os.path.join(root_dir_bubbles, '*.pt')), key=sort_order)
        self.gelslim_files = sorted(glob.glob(os.path.join(root_dir_gelslim, '*.pt')), key=sort_order)
        self.device = device
        self.bubbles_rotate = transforms.RandomRotation((180,180))

    def __len__(self):
        return len(self.bubbles_files)

    def __getitem__(self, idx):
        # BUBBLES
        bubbles_data = torch.load(self.bubbles_files[idx], map_location=self.device)
        gelslim_data = torch.load(self.gelslim_files[idx], map_location=self.device)
        # torch.save(bubbles_data, os.path.join(self.final_directory, self.tool_name + f'_data_{idx}.pt'))
        return bubbles_data, gelslim_data, idx

class TactileTransferSimCLR(Dataset):
    def __init__(self, root_dir_bubbles, root_dir_gelslim, device, bubbles_transform=None, gelslim_transform=None, augmentations = False, grayscale = False, visualize = False, norm_viz = True):
        """
        Args:
            root_dir: the directory of the dataset
            transform: pytorch transformations.
        """

        self.bubbles_transform = bubbles_transform
        self.gelslim_transform = gelslim_transform
        self.device = device
        self.augmentations = augmentations
        self.grayscale = grayscale
        self.visualize = visualize
        self.norm_viz = norm_viz
        self.bubbles_files = sorted(glob.glob(os.path.join(root_dir_bubbles, '*.pt')), key=sort_order)
        self.gelslim_files = sorted(glob.glob(os.path.join(root_dir_gelslim, '*.pt')), key=sort_order)
        self.simclr_transf = TransformsSimCLR(size=128)

    def __len__(self):        
        # our pairs can be either GG, GB, BG, or BB
        # which means that the first sensor can be either B or G
        return len(self.bubbles_files) * 2

    def __getitem__(self, idx):

        sensor_idx = idx // 2
        
        # GET BUBBLE IMAGE
        bubbles_data = torch.load(self.bubbles_files[sensor_idx], map_location=self.device) 
        bubbles_img = bubbles_data['bubble_imprint']
        bubbles_ref = bubbles_data['bubble_depth_ref']

        # normalization + resize
        if self.bubbles_transform:
            bubbles_img = self.bubbles_transform(bubbles_img)            
            bubbles_img = bubbles_img.repeat(1, 3, 1, 1)

        bubbles_mean = bubbles_img.mean()
        bubbles_std = bubbles_img.std()

        # GET GELSLIM IMAGE
        gelslim_data = torch.load(self.gelslim_files[sensor_idx], map_location=self.device)
        gelslim_img = gelslim_data['gelslim']
        gelslim_ref = gelslim_data['gelslim_ref']
        gelslim_diff = gelslim_img - gelslim_ref

        gelslim_img = gelslim_img / 255
        gelslim_ref = gelslim_ref / 255
        gelslim_diff = gelslim_diff / 255

        # normalization + resize
        if self.gelslim_transform:
            gelslim_diff = self.gelslim_transform(gelslim_diff)

        gelslim_mean = gelslim_diff.mean(dim=[0,2,3])
        gelslim_std = gelslim_diff.std(dim=[0,2,3])
        

        inter_domain = random.choice([0, 1])
        
        
        if idx < len(self.bubbles_files):
            sensor_a = bubbles_img
            sensor_b = bubbles_img if inter_domain == 0 else gelslim_diff
            sensor_a_is_gel = 0
            sensor_b_is_gel = inter_domain and sensor_a_is_gel
        else:
            sensor_a = gelslim_diff
            sensor_b = gelslim_diff if inter_domain == 0 else bubbles_img
            sensor_a_is_gel = 1
            sensor_b_is_gel = inter_domain and sensor_a_is_gel
                
        sensor_a_t = self.simclr_transf(sensor_a, sensor_a_is_gel)
        sensor_b_t = self.simclr_transf(sensor_b, sensor_b_is_gel)

        if self.visualize and not(idx%10):
            print(torch.mean(gelslim_diff[1]))
            gelslim_grid = make_grid(sensor_a_t, normalize=self.norm_viz).permute((1,2,0))
            bubbles_grid = make_grid(sensor_b_t, normalize=self.norm_viz).permute((1,2,0))

            grid = torch.cat([gelslim_grid, bubbles_grid], dim=0)
            angle = np.around(np.array(gelslim_data['theta']) * (180/3.14), decimals=2)
            plt.imshow(np.asarray(grid))
            plt.title('Angle:' + str(angle))
            plt.show()

            # save = input("Enter 1 (for yes) or 0 (for no):")
            save = False

            if save:
                dir = input("Enter left or right or both:")
                
                if dir == "left" or dir == "both":
                    torch.save(bubbles_data[1], "/home/samanta/tactile_style_transfer/processed_data/processed_data_filtered/bubbles_processed_data/" \
                               + self.bubbles_files[idx] + "_l")
                    torch.save(gelslim_data[1], "/home/samanta/tactile_style_transfer/processed_data/processed_data_filtered/gelslim_processed_data/" \
                               + self.gelslim_files[idx] + "_l")
                
                if dir == "right" or dir == "both":
                    torch.save(bubbles_data[0], "/home/samanta/tactile_style_transfer/processed_data/processed_data_filtered/bubbles_processed_data/" \
                               + self.bubbles_files[idx] + "_r")
                    torch.save(gelslim_data[0], "/home/samanta/tactile_style_transfer/processed_data/processed_data_filtered/gelslim_processed_data/" \
                               + self.gelslim_files[idx] + "_r")

        # ADDITIONAL INFORMATION
        info = {
                'bubbles_data': {
                                'bubble_depth_ref': bubbles_ref,
                                'theta': bubbles_data['theta'],
                                'K': bubbles_data['K'],
                                'bubbles_tr_quat': bubbles_data['bubbles_tr_quat'],
                                'tool_tr_quat': bubbles_data['tool_tr_quat'],
                                'finger_tr_quat': bubbles_data['finger_tr_quat'],
                                'grasp_frame_quat': bubbles_data['grasp_frame_quat'],
                                'img_mean': bubbles_mean,
                                'img_std': bubbles_std,
                                },
                'gelslim_data': {
                                'gelslim_ref': gelslim_ref,
                                'theta': gelslim_data['theta'],
                                'gelslim_tr_quat': gelslim_data['gelslim_tr_quat'],
                                'tool_tr_quat': gelslim_data['tool_tr_quat'],
                                'finger_tr_quat': gelslim_data['finger_tr_quat'],
                                'grasp_frame_quat': gelslim_data['grasp_frame_quat'],
                                'img_mean': gelslim_mean,
                                'img_std': gelslim_std,
                                }}

        return sensor_a_t, sensor_b_t, info
        # return sensor_a, sensor_b, info
    


def dataset_loading(previous_dataset,
                    bubbles_path, 
                    gelslim_path, 
                    bubbles_transform, 
                    gelslim_transform, 
                    device, 
                    all = False, 
                    data = 'cross_GB', 
                    grayscale = False, 
                    cropped=False, 
                    single = True, 
                    distortion = True,
                    random_sensor = False, 
                    color_jitter = False, 
                    rotation = False, 
                    flipping = False):
    
    dataset = TactileTransferAllInfoAugment( bubbles_path, gelslim_path, device, bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform, data=data, grayscale=grayscale, cropped=cropped,  single=single, distortion=distortion, random_sensor=random_sensor, color_jitter=color_jitter, rotation=rotation, flipping=flipping)
    dataset_for_val = TactileTransferAllInfoAugment( bubbles_path, gelslim_path, device, bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform, data=data, grayscale=grayscale, cropped=cropped, single=single, distortion=distortion, random_sensor=random_sensor, color_jitter=color_jitter, rotation=rotation, flipping=flipping)
    # TODO: Make sure dataset is balanced
    # if not all:
    #     indices = torch.tensor([i for i in range(0, 460)])
    #     # indices = torch.tensor([i for i in range(0, 100)])
    #     dataset = Subset(dataset, indices)
    #     dataset_for_val = Subset(dataset_for_val, indices)
    
    print('dataset length', dataset.__len__())

    train_len = int(0.8*dataset.__len__()) + 1
    val_len = int(0.5*(dataset.__len__() -  train_len))
    test_len = dataset.__len__() -  train_len - val_len
        
    train_set_add, _, _ = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(0))
    _, val_set_add, test_set_add = random_split(dataset_for_val, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(0))
    # import pdb; pdb.set_trace()
    train_dataset = ConcatDataset([previous_dataset[0], train_set_add])
    val_dataset = ConcatDataset([previous_dataset[1], val_set_add])
    test_dataset = ConcatDataset([previous_dataset[2], test_set_add])
    previous_dataset = (train_dataset, val_dataset, test_dataset)
    return train_set_add, val_set_add, test_set_add, previous_dataset

def val_dataset_loading(bubbles_path, 
                        gelslim_path, 
                        bubbles_transform, 
                        gelslim_transform, 
                        device, data = 'cross_GB', 
                        grayscale = False, 
                        cropped=False, 
                        single = True, 
                        distortion = True,
                        random_sensor = False, 
                        color_jitter = False, 
                        rotation = False, 
                        flipping = False):
    
    dataset = TactileTransferAllInfoAugment( bubbles_path, gelslim_path, device, bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform, data=data, grayscale = grayscale, cropped=cropped, single=single, distortion=distortion, random_sensor=random_sensor, color_jitter=color_jitter, rotation=rotation, flipping=flipping)
    indices = torch.tensor([i for i in range(0, 100)])  
    dataset = Subset(dataset, indices)
    val_set_add, test_set_add = random_split(dataset, [int(0.5*dataset.__len__()), int(0.5*dataset.__len__())], generator=torch.Generator().manual_seed(0))
    return val_set_add, test_set_add

def all_test_dataset_loading(bubbles_data_folders, gelslim_data_folders, bubbles_transform, gelslim_transform, device, data = 'cross_GB', grayscale = False, cropped=False, single = True):
    previous_dataset = []
    tools, bubbles_tools_paths, gelslim_tools_paths = get_tool_paths(bubbles_data_folders, gelslim_data_folders)
    for i in range(len(bubbles_tools_paths)):
        bubbles_path = bubbles_tools_paths[i]
        gelslim_path = gelslim_tools_paths[i]

        print(os.path.basename(bubbles_path))

        dataset = TactileTransferAllInfoAugment( bubbles_path, gelslim_path, device, bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform, data=data, grayscale=grayscale, cropped=cropped,  single=single)
        indices = torch.tensor([i for i in range(0, 101)])  
        dataset = Subset(dataset, indices)
        previous_dataset = ConcatDataset([previous_dataset, dataset])
    return previous_dataset
def all_test_diffusion_dataset_loading(bubbles_data_folders, gelslim_data_folders, bubbles_transform, gelslim_transform, device, data = 'cross_GB', grayscale = False, cropped=False, single = True, dataset_norm = False):
    previous_dataset = []
    # diffusion_path = '/home/samanta/Documents/UMICH/UMICH_Research_Codes/tactile_style_transfer/tactile_style_transfer/scripts/diffusion_results/raw_data_sample_10_guidance_4_new_test'
    diffusion_path = "/home/samanta/Documents/UMICH/UMICH_Research_Codes/tactile_style_transfer/processed_data/diffusion_train_only"
    tools, bubbles_tools_paths, gelslim_tools_paths = get_tool_paths(bubbles_data_folders, gelslim_data_folders)
    for i in range(len(bubbles_tools_paths)):
        bubbles_path = bubbles_tools_paths[i]
        gelslim_path = gelslim_tools_paths[i]
        tool = tools[i]
        print(tool)

        dataset = TactileTransferDiffusion(tool, bubbles_path, gelslim_path, diffusion_path, device, diffusion_idx=0, bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform, dataset_norm=dataset_norm)
        indices = torch.tensor([i for i in range(0, 101)])  
        dataset = Subset(dataset, indices)
        previous_dataset = ConcatDataset([previous_dataset, dataset])
    return previous_dataset
def visualization_samples(previous_samples_inputs, previous_samples_gt, val_set, single = False, mod = '1'):
    # selected_test_img = torch.tensor([1, 17, 33, 42, 50, 72, 100])
    # selected_test_img = torch.tensor([1, 11, 21, 31, 41, 51, 61, 71, 81, 100])
    selected_test_img = torch.tensor([1, 17, 33, 42]) #, 50, 72, 100])
    # selected_test_img = torch.arange(0, 20)
    subset = Subset(val_set, selected_test_img)
    val_dataloader = DataLoader(subset, batch_size=10, shuffle=False)
    # import pdb; pdb.set_trace()
    new_interesting_inputs, new_interesting_gt, info = next(iter(val_dataloader))
    if not single:
        new_interesting_inputs = torch.cat([new_interesting_inputs[:,1], new_interesting_inputs[:,0]], dim = 2)
        new_interesting_gt = torch.cat([new_interesting_gt[:,1], new_interesting_gt[:,0]], dim = 2)

    interesting_inputs = torch.cat([previous_samples_inputs, new_interesting_inputs], dim = 0)
    interesting_gt = torch.cat([previous_samples_gt, new_interesting_gt], dim = 0)
    return interesting_inputs, interesting_gt

def data_selection(dataset_name):
    # bubbles_old_data_path = "/home/samanta/tactile_style_transfer/processed_data/bubbles_processed_data"
    # bubbles_new_data_path = "/home/samanta/tactile_style_transfer/processed_data/new_bubbles_filtered_data"
    # bubbles_task_data_path = "/home/samanta/tactile_style_transfer/processed_data/new_bubbles_peg_in_hole_data"
    # bubbles_test_objs = "/home/samanta/tactile_style_transfer/processed_data/new_bubbles_test_objs"
    # gelslim_old_data_path = "/home/samanta/tactile_style_transfer/processed_data/gelslim_processed_data"
    # gelslim_new_data_path = "/home/samanta/tactile_style_transfer/processed_data/new_gelslim_filtered_data"
    # gelslim_task_data_path = "/home/samanta/tactile_style_transfer/processed_data/new_gelslim_peg_in_hole_data"
    # gelslim_test_objs = "/home/samanta/tactile_style_transfer/processed_data/new_gelslim_test_objs"

    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    datasets_path = os.path.join(project_path, "processed_data")

    bubbles_old_data_path = os.path.join(datasets_path, "back_ups/old_bubbles_processed_data")
    bubbles_new_data_path = os.path.join(datasets_path, "bubbles/bubbles_training_data_filtered_flipped")
    bubbles_task_data_path = os.path.join(datasets_path, "bubbles/bubbles_training_data_processed_flipped_2")
    bubbles_test_objs = os.path.join(datasets_path, "bubbles/bubbles_testing_data_processed_flipped_2")
    gelslim_old_data_path = os.path.join(datasets_path, "back_ups/old_gelslim_processed_data")
    gelslim_new_data_path = os.path.join(datasets_path, "gelslims/gelslim_training_data_filtered_flipped")
    gelslim_task_data_path = os.path.join(datasets_path, "gelslims/gelslim_training_data_processed_flipped_2")
    # gelslim_test_objs = os.path.join(datasets_path, "gelslims/gelslim_testing_data_processed_flipped")
    gelslim_test_objs = os.path.join(datasets_path, "gelslims/gelslim_testing_data_processed_flipped_2")
    # import pdb; pdb.set_trace()

    if dataset_name == 'old':
        bubbles_data_folders = [bubbles_old_data_path]
        gelslim_data_folders = [gelslim_old_data_path]

        test_tools = [
                        'r7p5mm_ati_T_shape',
                        # 'test_obj_nut',
                        # 'test_obj_pen',
                        # 'test_obj_pen_back',
                        # 'test_obj_hex_key'
                     ]
    else:
        bubbles_data_folders = [bubbles_new_data_path, bubbles_task_data_path, bubbles_test_objs]
        gelslim_data_folders = [gelslim_new_data_path, gelslim_task_data_path, gelslim_test_objs]

        test_tools = [
                #   'r7p5mm_ati_T_shape',
                #   'pattern_05_3_lines_angle_2',
                #   'pattern_09_curves_degree_120_radios_10',
                #   'pattern_16_ellipse_1',
                #   'pattern_19_hex_2',
                # 'test_obj_circle_peg_seen',
                # 'test_obj_circle_peg_unseen',
                # 'test_obj_hex_key_seen',
                # 'test_obj_hex_peg_seen',
                # 'test_obj_hex_peg_unseen',
                # 'test_obj_rectangle_peg_seen',
                # 'test_obj_rectangle_peg_unseen',
                # 'test_obj_nut_seen',
                'test_obj_hex_small_peg_seen',
                'test_obj_square_small_peg_seen',
                'test_obj_tilted_square_small_peg_seen',
                # 'test_obj_hex_key_new_seen'
                ]

    
    return bubbles_data_folders, gelslim_data_folders, test_tools

def get_tool_paths(bubbles_data_folders, gelslim_data_folders):
    #Generate tool paths
    tools = []
    bubbles_tools_paths = []
    gelslim_tools_paths = []

    for i in range(len(bubbles_data_folders)):
        bubbles_tools =  sorted(os.listdir(bubbles_data_folders[i]))
        gelslim_tools =  sorted(os.listdir(gelslim_data_folders[i]))

        tools += [tool.replace('bubble_style_transfer_dataset_bubbles_', '') for tool in bubbles_tools]

        for j in range(len(bubbles_tools)):
            bubbles_tools[j] = os.path.join(bubbles_data_folders[i], bubbles_tools[j])
            gelslim_tools[j] = os.path.join(gelslim_data_folders[i], gelslim_tools[j])

        bubbles_tools_paths += bubbles_tools
        gelslim_tools_paths += gelslim_tools

    return tools, bubbles_tools_paths, gelslim_tools_paths

def all_datasets_loading(bubbles_data_folders, 
                         gelslim_data_folders, 
                         test_tools, 
                         bubbles_transform = None, 
                         gelslim_transform = None, 
                         device = 'cpu', 
                         single = False, 
                         all = False, 
                         data = 'cross_GB', 
                         mod = '1', 
                         grayscale = False, 
                         cropped = False, 
                         distortion = True,
                         random_sensor = False, 
                        color_jitter = False, 
                        rotation = False, 
                        flipping = False):
    
    previous_dataset = ([], [], []) # Initializing Train, Val and Test Datasets
    tool_val_names = []
    tool_val_datasets = []
    train_inputs = torch.empty((0), device=device)
    train_gt = torch.empty((0), device=device)
    center_val_inputs = torch.empty((0), device=device)
    center_val_gt = torch.empty((0), device=device)
    tool_val_inputs = torch.empty((0), device=device)
    tool_val_gt = torch.empty((0), device=device)

    tools, bubbles_tools_paths, gelslim_tools_paths = get_tool_paths(bubbles_data_folders, gelslim_data_folders)

    for i in range(len(bubbles_tools_paths)):
        bubbles_path = bubbles_tools_paths[i]
        gelslim_path = gelslim_tools_paths[i]
        if tools[i] in test_tools:
            tool_val_set, _ = val_dataset_loading(bubbles_path, gelslim_path, bubbles_transform, gelslim_transform, device, data = data, grayscale = grayscale, cropped = cropped, single = single, distortion = distortion, random_sensor=random_sensor, color_jitter=color_jitter, rotation=rotation, flipping=flipping)
            tool_val_inputs, tool_val_gt = visualization_samples(tool_val_inputs, tool_val_gt, tool_val_set, single=single, mod=mod)
            tool_val_datasets.append(tool_val_set)
            tool_val_names.append(tools[i])
        else:
            print(i, tools[i])
            train_set, center_val_set, _, previous_dataset = dataset_loading(previous_dataset, bubbles_path, gelslim_path, bubbles_transform, gelslim_transform, device, all = all, data = data, grayscale = grayscale, cropped = cropped, single = single, distortion = distortion, random_sensor=random_sensor, color_jitter=color_jitter, rotation=rotation, flipping=flipping)
            train_inputs, train_gt = visualization_samples(train_inputs, train_gt, train_set, single=single, mod=mod)
            center_val_inputs, center_val_gt = visualization_samples(center_val_inputs, center_val_gt, center_val_set, single=single, mod=mod)
            
    train_imgs = (train_inputs, train_gt)
    center_val_imgs = (center_val_inputs, center_val_gt)
    tool_val_imgs = (tool_val_inputs, tool_val_gt)
    return previous_dataset[0], previous_dataset[1], tool_val_datasets, tool_val_names, train_imgs, center_val_imgs, tool_val_imgs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VQ-VAE')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--diffusion', action='store_true', help='Train the model')

    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    datasets_path = os.path.join(project_path, "processed_data")

    bubbles_old_data_path = os.path.join(datasets_path, "back_ups/old_bubbles_processed_data")
    bubbles_new_data_path = os.path.join(datasets_path, "bubbles/bubbles_training_data_filtered_flipped")
    bubbles_task_data_path = os.path.join(datasets_path, "bubbles/bubbles_training_data_processed_flipped_2")
    bubbles_test_objs = os.path.join(datasets_path, "bubbles/bubbles_testing_data_processed_flipped_2")
    gelslim_old_data_path = os.path.join(datasets_path, "back_ups/old_gelslim_processed_data")
    gelslim_new_data_path = os.path.join(datasets_path, "gelslims/gelslim_training_data_filtered_flipped")
    gelslim_task_data_path = os.path.join(datasets_path, "gelslims/gelslim_training_data_processed_flipped_2")
    gelslim_test_objs = os.path.join(datasets_path, "gelslims/gelslim_testing_data_processed_flipped_2")

    new_dataset_gelslim = '/home/samanta/tactile_style_transfer/new_processed_data/gelslims/data'
    new_dataset_bubbles = '/home/samanta/tactile_style_transfer/new_processed_data/bubbles/data'

    device = 'cuda:0'
    test_tools = ['pattern_05_3_lines_angle_2','pattern_35', 'pattern_36']
    args = parser.parse_args()

    gelslims_mean = torch.tensor([-0.0082, -0.0059, -0.0066])
    gelslims_std = torch.tensor([0.0989, 0.0746, 0.0731])
    bubbles_mean = torch.tensor([0.0382])
    bubbles_std = torch.tensor([0.0424])

    gelslim_transform = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.Normalize(gelslims_mean, gelslims_std)
                                            ])
    bubbles_transform = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.RandomRotation((180,180)),
                                            transforms.Normalize(bubbles_mean, bubbles_std)
                                            ])

    if args.train:
        bubbles_data_folders = [new_dataset_bubbles]
        gelslim_data_folders = [new_dataset_gelslim]
        train_dataset, center_val_dataset, tool_val_datasets, tool_val_names, train_imgs, center_val_imgs, tool_val_imgs = all_datasets_loading(bubbles_data_folders, gelslim_data_folders, test_tools, device=device)
        dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)

    else:
        bubbles_path = "/home/samanta/Documents/UMICH/UMICH_Research_Codes/tactile_style_transfer/processed_data/test_unseen_only/bubbles" #[bubbles_test_objs]
        gelslim_path = "/home/samanta/Documents/UMICH/UMICH_Research_Codes/tactile_style_transfer/processed_data/test_unseen_only/gelslims" #[gelslim_test_objs]
        diffusion_path = "/home/samanta/Documents/UMICH/UMICH_Research_Codes/tactile_style_transfer/processed_data/diffusion_test_unseen"
        tool = 'r7p5mm_ati_T_shape'
        # dataset = all_test_diffusion_dataset_loading(bubbles_data_folders, gelslim_data_folders, bubbles_transform, gelslim_transform, device=device, dataset_norm=False)
        dataset = TactileTransferDiffusion(tool, bubbles_path, gelslim_path, diffusion_path, device, diffusion_idx=0, bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform, dataset_norm=False)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    # tools, bubbles_tools_paths, gelslim_tools_paths = get_tool_paths(bubbles_data_folders, gelslim_data_folders)

    if args.diffusion:
        bubbles_gt_means = 0
        bubbles_gt_stds = 0
        bubbles_gt_min = 1
        bubbles_gt_max = -1000
        bubbles_gt_diff_means = 0
        bubbles_gt_diff_stds = 0
        bubbles_gt_diff_min = 1
        bubbles_gt_diff_max = -1000
        bubbles_pred_means = 0
        bubbles_pred_stds = 0
        bubbles_pred_min = 1
        bubbles_pred_max = -1000
        for _, info in dataloader:
            bubbles_gt_means += info['bubbles_data']['bubbles_metrics']['gt_mean'].mean()
            bubbles_gt_stds += (info['bubbles_data']['bubbles_metrics']['gt_std'].mean())**2
            bubbles_gt_min = min(bubbles_gt_min, info['bubbles_data']['bubbles_metrics']['gt_min'].min())
            bubbles_gt_max = max(bubbles_gt_max, info['bubbles_data']['bubbles_metrics']['gt_max'].max())
            # bubbles_gt_diff_means += info['bubbles_data']['bubbles_metrics']['gt_diff_mean'].mean()
            # bubbles_gt_diff_stds += (info['bubbles_data']['bubbles_metrics']['gt_diff_std'].mean())**2
            # bubbles_gt_diff_min = min(bubbles_gt_diff_min, info['bubbles_data']['bubbles_metrics']['gt_diff_min'].min())
            # bubbles_gt_diff_max = max(bubbles_gt_diff_max, info['bubbles_data']['bubbles_metrics']['gt_diff_max'].max())
            bubbles_pred_means += info['bubbles_data']['bubbles_metrics']['pred_mean'].mean()
            bubbles_pred_stds += (info['bubbles_data']['bubbles_metrics']['pred_std'].mean())**2
            bubbles_pred_min = min(bubbles_pred_min, info['bubbles_data']['bubbles_metrics']['pred_min'].min())
            bubbles_pred_max = max(bubbles_pred_max, info['bubbles_data']['bubbles_metrics']['pred_max'].max())

        print('bubbles_gt_means', bubbles_gt_means/len(dataloader))
        print('bubbles_gt_stds', torch.sqrt(bubbles_gt_stds/len(dataloader)))
        print('bubbles_gt_min', bubbles_gt_min)
        print('bubbles_gt_max', bubbles_gt_max)
        # print('bubbles_gt_diff_means', bubbles_gt_diff_means/len(dataloader))
        # print('bubbles_gt_diff_stds', torch.sqrt(bubbles_gt_diff_stds/len(dataloader)))
        # print('bubbles_gt_diff_min', bubbles_gt_diff_min)
        # print('bubbles_gt_diff_max', bubbles_gt_diff_max)
        print('bubbles_pred_means', bubbles_pred_means/len(dataloader))
        print('bubbles_pred_stds', torch.sqrt(bubbles_pred_stds/len(dataloader)))
        print('bubbles_pred_min', bubbles_pred_min)
        print('bubbles_pred_max', bubbles_pred_max)

    else:
        bubbles_means = 0
        bubbles_stds = 0
        bubbles_min = 1
        bubbles_max = -1000
        gelslims_means = 0
        gelslims_stds = 0
        for images, labels, info in dataloader:
            bubbles_means += info['bubbles_data']['img_mean'].mean()
            bubbles_stds += (info['bubbles_data']['img_std'].mean())**2
            gelslims_means += info['gelslim_data']['img_mean'].mean(dim=0)
            gelslims_stds += (info['gelslim_data']['img_std'].mean(dim=0))**2
            bubbles_min = min(bubbles_min, info['bubbles_data']['min'].min())
            bubbles_max = max(bubbles_max, info['bubbles_data']['max'].max())
            # import pdb; pdb.set_trace()

        bubbles_mean_cal = bubbles_means/len(dataloader)
        bubbles_std_cal = torch.sqrt(bubbles_stds/len(dataloader))
        gelslims_mean_cal = gelslims_means/len(dataloader)
        gelslims_std_cal = torch.sqrt(gelslims_stds/len(dataloader))
        print('bubbles_mean_cal', bubbles_mean_cal)
        print('bubbles_std_cal', bubbles_std_cal)
        print('gelslims_mean_cal', gelslims_mean_cal)
        print('gelslims_std_cal', gelslims_std_cal)
        print('bubbles_min', bubbles_min)
        print('bubbles_max', bubbles_max)
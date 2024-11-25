import os
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from touch2touch.evaluation.metrics import psnr_metric, ssim_metric, fid_metric
from touch2touch.vq_vae.modules import model_definition
from touch2touch.evaluation.ICP_dataset_cal import get_bubbles_data_TST, icp_dataset_cal, get_point_could_images,get_metrics, save_results_data, save_results_images

GELSLIM_MEAN = torch.tensor([-0.0082, -0.0059, -0.0066])
GELSLIM_STD = torch.tensor([0.0989, 0.0746, 0.0731])
BUBBLES_MEAN = torch.tensor([0.00382])
BUBBLES_STD = torch.tensor([0.00424])
# BUBBLES_MEAN = torch.tensor([0.0048])
# BUBBLES_STD = torch.tensor([0.0046])

BH = 140
BW = 175
GH = 53
GW = 71
offset_H = -4
offset_W = 10

def extract_tool_names(train_path):
    train_tools = os.listdir(train_path)

    for i, tool in enumerate(train_tools):
        remove_idx = tool.find('_data_')
        train_tools[i] = tool[:remove_idx]

    train_tools = list(set(train_tools))
    train_tools.sort()
    return train_tools

def logging_image_grid(images, captions, path, ncol=7, normalize = True, save = True):
    if not normalize:
        norm_text = "_not_normalized"
    else:
        norm_text = ""

    grids = [make_grid(img, nrow=ncol,padding=1, normalize=normalize, scale_each=True) for img in images]
    for grid, caption in zip(grids, captions):
        if save:
            save_image(grid, path +  '/' + caption + norm_text + '.png')
        else:
            plt.imshow(np.asarray(grid.permute((1,2,0)).cpu()))
            plt.title(caption)
            plt.axis('off')
            plt.show()
    return

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

def sensor_transforms(mode):
    '''
    mode: transform, transform_inv_paired
    '''
    if mode == "transform":
        gelslim_transform = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.Normalize(GELSLIM_MEAN, GELSLIM_STD)
                                            ])
        bubbles_transform = transforms.Compose([transforms.Resize((128,128)),
                                            # transforms.RandomRotation((180,180)),
                                            transforms.Normalize(BUBBLES_MEAN, BUBBLES_STD)
                                            ])
    elif mode == "transform_rotation":
        gelslim_transform = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.Normalize(GELSLIM_MEAN, GELSLIM_STD)
                                            ])
        bubbles_transform = transforms.Compose([transforms.Resize((128,128)),
                                            # transforms.RandomRotation((180,180)),
                                            transforms.Normalize(BUBBLES_MEAN, BUBBLES_STD)
                                            ])
    elif mode == "transform_inv_rotation":
        gelslim_transform = transforms.Compose([transforms.Resize((320,427)),
                                                    unnormalize(GELSLIM_MEAN, GELSLIM_STD)])
        bubbles_transform = transforms.Compose([transforms.Resize((140,175)),
                                                transforms.RandomRotation((180,180)),
                                                unnormalize(BUBBLES_MEAN, BUBBLES_STD)])

    return gelslim_transform, bubbles_transform

class TST_testing(Dataset):
    def __init__(self, tool_name, model, model_path, bubbles_path, gelslim_path, device):
        self.tool_name = tool_name
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.eval()
        self.bubbles_path = bubbles_path
        self.gelslim_path = gelslim_path
        self.bubbles_data_paths = [path for path in os.listdir(bubbles_path) if tool_name in path]
        self.gelslim_data_paths = [path for path in os.listdir(gelslim_path) if tool_name in path]
        self.gelslim_transform, self.bubbles_transform = sensor_transforms("transform_rotation")
        self.gelslim_transform_inv, self.bubbles_transform_inv = sensor_transforms("transform_inv_rotation")
        self.device = device
        self.rotation = transforms.RandomRotation((180,180))

    def __len__(self):
        return len(self.bubbles_data_paths)

    def __getitem__(self, idx):
        bubbles_data = torch.load(os.path.join(self.bubbles_path, self.bubbles_data_paths[idx]))
        gelslim_data = torch.load(os.path.join(self.gelslim_path, self.gelslim_data_paths[idx]))
        
        gelslim_data_transformed = self.gelslim_transform((gelslim_data['gelslim'] - gelslim_data['gelslim_ref'])/255).to(self.device)
        bubbles_prediction, _, _ = self.model(gelslim_data_transformed)
        bubbles_prediction = self.bubbles_transform_inv(bubbles_prediction)

        info = {
                'bubbles_data': {
                                'bubble_depth_ref': bubbles_data['bubble_depth_ref'].to(self.device),
                                'theta': bubbles_data['theta'].to(self.device),
                                'K': bubbles_data['K'].to(self.device),
                                'bubbles_tr_quat': bubbles_data['bubbles_tr_quat'].to(self.device),
                                'tool_tr_quat': bubbles_data['tool_tr_quat'].to(self.device),
                                'finger_tr_quat': bubbles_data['finger_tr_quat'].to(self.device),
                                'grasp_frame_quat': bubbles_data['grasp_frame_quat'].to(self.device),
                                'x': bubbles_data['x'].to(self.device),
                                'y': bubbles_data['y'].to(self.device),
                                },
                'gelslim_data': {
                                'gelslim_ref': gelslim_data['gelslim_ref'].to(self.device),
                                'theta': gelslim_data['theta'].to(self.device),
                                'gelslim_tr_quat': gelslim_data['gelslim_tr_quat'].to(self.device),
                                'tool_tr_quat': gelslim_data['tool_tr_quat'].to(self.device),
                                'finger_tr_quat': gelslim_data['finger_tr_quat'].to(self.device),
                                'grasp_frame_quat': gelslim_data['grasp_frame_quat'].to(self.device),
                                'x': gelslim_data['x'].to(self.device),
                                'y': gelslim_data['y'].to(self.device),
                                }
                }
        
        return self.rotation(gelslim_data['gelslim'].to(self.device)), self.rotation(gelslim_data['gelslim_ref'].to(self.device)), bubbles_data['bubble_imprint'].to(self.device), bubbles_prediction, info

class TST_diffusion_testing(Dataset):
    def __init__(self, tool_name,  bubbles_path, gelslim_path, diffusion_results_path, device, diffusion_idx = 0, dataset_norm = False, gt_stats = (0.0126, 0.0015), pred_stats = (0.0126, 0.0015), both= False):
        """
        Args:
            root_dir: the directory of the dataset
            transform: pytorch transformations.
        """
        self.bubbles_path = bubbles_path
        self.gelslim_path = gelslim_path
        self.bubbles_data_paths = [path for path in os.listdir(bubbles_path) if tool_name in path]
        self.gelslim_data_paths = [path for path in os.listdir(gelslim_path) if tool_name in path]
        self.gelslim_transform, self.bubbles_transform = sensor_transforms("transform_rotation")
        self.gelslim_transform_inv, self.bubbles_transform_inv = sensor_transforms("transform_inv_rotation")
        self.device = device
        self.diffusion_idx = diffusion_idx

        if both:
            diffusion_results_path = diffusion_results_path.replace('results_samples', 'results_samples_right')

        diffusion_results_files = []
        for filename in os.listdir(diffusion_results_path):
            # Check if the file is a .pt file
            if filename.endswith('.pt'):
                if tool_name in filename:
                    diffusion_results_files.append(os.path.join(diffusion_results_path, filename))

        
        self.diffusion_results_files = diffusion_results_files
        # import pdb; pdb.set_trace()
        print(len(self.diffusion_results_files))
        self.rotation = transforms.RandomRotation((180,180))
        self.bubbles_post_processing = transforms.Resize((140,175))
        
        self.dataset_norm = dataset_norm
        gt_mean = torch.tensor([gt_stats[0]])
        gt_std = torch.tensor([gt_stats[1]])
        pred_mean = torch.tensor([pred_stats[0]])
        pred_std =  torch.tensor([pred_stats[1]])
        self.diffusion_pred_norm = transforms.Compose([transforms.Normalize(pred_mean, pred_std),
                                                       unnormalize(gt_mean, gt_std)
                                                      ])
        
        self.BUBBLES_MIN, self.BUBBLES_MAX = -0.0116, 0.0225
        self.both = both
        

    def __len__(self):
        return len(self.bubbles_data_paths)

    def __getitem__(self, idx):
        bubbles_data = torch.load(os.path.join(self.bubbles_path, self.bubbles_data_paths[idx]))
        gelslim_data = torch.load(os.path.join(self.gelslim_path, self.gelslim_data_paths[idx]))

        if self.both:
            path_r = self.diffusion_results_files[idx]
            x_rec_r = torch.load(path_r, map_location=self.device)['prediction'][self.diffusion_idx].float()
            bubbles_img_prediction_r = transforms.functional.rgb_to_grayscale(x_rec_r).unsqueeze(0)
            bubbles_img_prediction_r = self.bubbles_post_processing(bubbles_img_prediction_r)

            path_l = self.diffusion_results_files[idx].replace('results_samples_right', 'results_samples_left')
            x_rec_l = torch.load(path_l, map_location=self.device)['prediction'][self.diffusion_idx].float()
            bubbles_img_prediction_l = transforms.functional.rgb_to_grayscale(x_rec_l).unsqueeze(0)
            bubbles_img_prediction_l = self.bubbles_post_processing(bubbles_img_prediction_l)

            if self.dataset_norm:
                bubbles_img_prediction_r = self.diffusion_pred_norm(bubbles_img_prediction_r)
                bubbles_img_prediction_l = self.diffusion_pred_norm(bubbles_img_prediction_l)
            
            bubbles_prediction = torch.cat([bubbles_img_prediction_r, bubbles_img_prediction_l], dim = 0)
            bubbles_gt = bubbles_data['bubble_imprint'].to(self.device)

        else:
            x_rec = torch.load(self.diffusion_results_files[idx], map_location=self.device)['prediction'][self.diffusion_idx].float()
            bubbles_img_prediction = transforms.functional.rgb_to_grayscale(x_rec).unsqueeze(0)
            
            if self.dataset_norm:
                bubbles_img_prediction = self.diffusion_pred_norm(bubbles_img_prediction)

            # TODO: Check right and left
            bubbles_img_prediction = self.bubbles_post_processing(bubbles_img_prediction)
            bubbles_prediction = torch.cat([bubbles_img_prediction, torch.flip(bubbles_img_prediction, [2])], dim = 0)
            bubbles_gt = torch.cat([bubbles_data['bubble_imprint'][0].unsqueeze(0), torch.flip(bubbles_data['bubble_imprint'][0].unsqueeze(0), [2])], dim = 0).to(self.device)

        # ADDITIONAL INFORMATION
        info = {
                'bubbles_data': {
                                'bubble_depth_ref': bubbles_data['bubble_depth_ref'].to(self.device),
                                'theta': bubbles_data['theta'].to(self.device),
                                'K': bubbles_data['K'].to(self.device),
                                'bubbles_tr_quat': bubbles_data['bubbles_tr_quat'].to(self.device),
                                'tool_tr_quat': bubbles_data['tool_tr_quat'].to(self.device),
                                'finger_tr_quat': bubbles_data['finger_tr_quat'].to(self.device),
                                'grasp_frame_quat': bubbles_data['grasp_frame_quat'].to(self.device),
                                'x': bubbles_data['x'].to(self.device),
                                'y': bubbles_data['y'].to(self.device),
                                },
                'gelslim_data': {
                                'gelslim_ref': gelslim_data['gelslim_ref'].to(self.device),
                                'theta': gelslim_data['theta'].to(self.device),
                                'gelslim_tr_quat': gelslim_data['gelslim_tr_quat'].to(self.device),
                                'tool_tr_quat': gelslim_data['tool_tr_quat'].to(self.device),
                                'finger_tr_quat': gelslim_data['finger_tr_quat'].to(self.device),
                                'grasp_frame_quat': gelslim_data['grasp_frame_quat'].to(self.device),
                                'x': gelslim_data['x'].to(self.device),
                                'y': gelslim_data['y'].to(self.device),
                                }
                }
        
        return self.rotation(gelslim_data['gelslim'].to(self.device)), self.rotation(gelslim_data['gelslim_ref'].to(self.device)), bubbles_gt, bubbles_prediction, info

class TST_diffusion_metrics(Dataset):
    def __init__(self, tool_name,  bubbles_path, gelslim_path, diffusion_results_path, device, diffusion_idx = 0, dataset_norm = False, gt_stats = (0.0126, 0.0015), pred_stats = (0.0126, 0.0015), both= False):
        """
        Args:
            root_dir: the directory of the dataset
            transform: pytorch transformations.
        """
        self.bubbles_path = bubbles_path
        self.gelslim_path = gelslim_path
        self.bubbles_data_paths = [path for path in os.listdir(bubbles_path) if tool_name in path]
        self.gelslim_data_paths = [path for path in os.listdir(gelslim_path) if tool_name in path]
        self.gelslim_transform = transforms.Resize((128,128))
        self.bubbles_transform_inv = sensor_transforms("transform_inv_rotation")[1]
        self.device = device
        self.diffusion_idx = diffusion_idx

        if both:
            diffusion_results_path = diffusion_results_path.replace('results_samples', 'results_samples_right')

        diffusion_results_files = []
        for filename in os.listdir(diffusion_results_path):
            # Check if the file is a .pt file
            if filename.endswith('.pt'):
                if tool_name in filename:
                    diffusion_results_files.append(os.path.join(diffusion_results_path, filename))

        
        self.diffusion_results_files = diffusion_results_files
        # import pdb; pdb.set_trace()
        # print(len(self.diffusion_results_files))
        self.rotation = transforms.RandomRotation((180,180))
        self.dataset_norm = dataset_norm
        gt_mean = torch.tensor([gt_stats[0]])
        gt_std = torch.tensor([gt_stats[1]])
        pred_mean = torch.tensor([pred_stats[0]])
        pred_std =  torch.tensor([pred_stats[1]])
        self.diffusion_pred_norm = transforms.Compose([transforms.Normalize(pred_mean, pred_std),
                                                       unnormalize(gt_mean, gt_std)
                                                      ])

    def __len__(self):
        return len(self.bubbles_data_paths)

    def __getitem__(self, idx):
        bubbles_data = torch.load(os.path.join(self.bubbles_path, self.bubbles_data_paths[idx]))
        bubbles_img = bubbles_data['bubble_imprint'].to(self.device)

        x_rec = torch.load(self.diffusion_results_files[idx], map_location=self.device)['prediction'][self.diffusion_idx].float()
        bubbles_img_prediction = (transforms.functional.rgb_to_grayscale(x_rec).unsqueeze(0))

        # import pdb; pdb.set_trace()

        #shift image distribution and apply inverse transform just for rotation and resizing
        if self.dataset_norm:
            bubbles_img_prediction = self.diffusion_pred_norm(bubbles_img_prediction)
        # bubbles_img_prediction = self.bubbles_transform_inv(bubbles_img_prediction)

        bubbles_gt_min = bubbles_img.min()
        bubbles_gt_max = bubbles_img.max()
        bubbles_gt_mean = bubbles_img.mean()
        bubbles_gt_std = bubbles_img.std()
        bubbles_gt_psum = bubbles_img.sum()
        bubbles_gt_psum_sq = (bubbles_img**2).sum()
        

        bubbles_pred_min = bubbles_img_prediction.min()
        bubbles_pred_max = bubbles_img_prediction.max()
        bubbles_pred_mean = bubbles_img_prediction.mean()
        bubbles_pred_std = bubbles_img_prediction.std()
        bubbles_pred_psum = bubbles_img_prediction.sum()
        bubbles_pred_psum_sq = (bubbles_img_prediction**2).sum()

        # ADDITIONAL INFORMATION
        bubbles_gt_metrics = {
                                'gt_min': bubbles_gt_min,
                                'gt_max': bubbles_gt_max,
                                'gt_mean': bubbles_gt_mean,
                                'gt_std': bubbles_gt_std,
                                'gt_psum': bubbles_gt_psum,
                                'gt_psum_sq': bubbles_gt_psum_sq,
                                'gt_shape': bubbles_img.shape,
                                }
        
        bubbles_pred_data = {   
                                'pred_min': bubbles_pred_min,
                                'pred_max': bubbles_pred_max,
                                'pred_mean': bubbles_pred_mean,
                                'pred_std': bubbles_pred_std,
                                'pred_psum': bubbles_pred_psum,
                                'pred_psum_sq': bubbles_pred_psum_sq,
                                'pred_shape': bubbles_img_prediction.shape,
                                }
        
        return bubbles_gt_metrics, bubbles_pred_data
    
def get_data(tool_name, bubbles_path, gelslim_path, device, data_model, model_results_path = '', stats = ((0.0126, 0.0015), (0.0126, 0.0015)), both = False):
    if data_model == "vq_vae":
        print('Loading VQ-VAE Model')
        model = model_definition("VQ-VAE-small", 3, 1, 256, 16385, device, single=True)
        # model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'working_model', 'VQ-VAE-small_dataset_new_data_cross_GB_mod_3_run_model.pt')
        dataset = TST_testing(tool_name, model, model_results_path, bubbles_path, gelslim_path, device)
        dataloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
        return next(iter(dataloader))
    elif data_model == "diffusion_norm":
        print('Loading Diffusion Model with Normalization')
        print('Diffusion Norm Stats: ', stats)
        dataset = TST_diffusion_testing(tool_name, bubbles_path, gelslim_path, model_results_path, device, diffusion_idx = 0, dataset_norm = True, gt_stats=stats[0], pred_stats=stats[1], both=both)
        dataloader = DataLoader(dataset, dataset.__len__(), shuffle=False)
        return next(iter(dataloader))
    else:
        print('Loading Other Model')
        dataset = TST_diffusion_testing(tool_name, bubbles_path, gelslim_path, model_results_path, device, diffusion_idx = 0, dataset_norm = False, both=both)
        dataloader = DataLoader(dataset, dataset.__len__(), shuffle=False)
        return next(iter(dataloader))
    
    
    return

def visual_qualitative(gelslim_input, gelslim_ref, bubbles_gt, bubbles_pred, info, project_path,):
    idxs = [11, 22, 33, 44]
    # idxs = list(range(20))
    icp_model_path = os.path.join(project_path, 'checkpoints/masking_unet/masking_sdf_new_dataset_model_new_dataset_tools_E30_B32_LR0.00001.pth')

    gelslim_input = gelslim_input[idxs]
    gelslim_input_viz_single = gelslim_input[:,1]
    gelslim_input_viz = torch.cat((gelslim_input[:,1], gelslim_input[:,0]), dim=2)

    # import pdb; pdb.set_trace()
    gelslim_diff = gelslim_input - gelslim_ref[idxs]
    gelslim_diff_viz_single = gelslim_diff[:,1]
    gelslim_diff_viz = torch.cat((gelslim_diff[:,1], gelslim_diff[:,0]), dim=2)

    bubbles_gt = bubbles_gt[idxs]
    bubbles_gt_viz_single = bubbles_gt[:,1]
    bubbles_gt_viz = torch.cat((bubbles_gt[:,1], bubbles_gt[:,0]), dim=2)

    bubbles_pred = bubbles_pred[idxs]
    bubbles_pred_viz_single = bubbles_pred[:,1]
    bubbles_pred_viz = torch.cat((bubbles_pred[:,1], bubbles_pred[:,0]), dim=2)
    # print('Bubbles GT mean:', bubbles_gt.mean())
    # print('Bubbles GT std:', bubbles_gt.std())
    # print('Bubbles Pred mean:', bubbles_pred.mean())
    # print('Bubbles Pred std:', bubbles_pred.std())
    bubbles_gt_data = get_bubbles_data_TST(bubbles_gt, info)
    bubbles_pred_data = get_bubbles_data_TST(bubbles_pred, info)
    bubbles_gt_pcd_viz = get_point_could_images(bubbles_gt_data, True, icp_model_path)
    bubbles_pred_pcd_viz = get_point_could_images(bubbles_pred_data, True, icp_model_path)

    visual_qualitative_results = {
                                'gelslim_input_viz': gelslim_input_viz,
                                'gelslim_input_viz_single': gelslim_input_viz_single,
                                'gelslim_diff_viz': gelslim_diff_viz,
                                'gelslim_diff_viz_single': gelslim_diff_viz_single,
                                'bubbles_gt_viz': bubbles_gt_viz,
                                'bubbles_gt_viz_single': bubbles_gt_viz_single,
                                'bubbles_pred_viz': bubbles_pred_viz,
                                'bubbles_pred_viz_single': bubbles_pred_viz_single,
                                'bubbles_gt_pcd_viz': bubbles_gt_pcd_viz,
                                'bubbles_pred_pcd_viz': bubbles_pred_pcd_viz
                                }
    
    return visual_qualitative_results

def visual_quantitative(bubbles_gt, bubbles_pred, device):
    bubbles_gt = torch.cat([bubbles_gt[:,1], bubbles_gt[:,0]], dim=0)
    bubbles_pred = torch.cat([bubbles_pred[:,1], bubbles_pred[:,0]], dim=0)

    psnr_error = psnr_metric(bubbles_gt, bubbles_pred)
    ssim_error = ssim_metric(bubbles_gt, bubbles_pred, device=device)
    fid_error = fid_metric(bubbles_gt.squeeze(1), bubbles_pred.squeeze(1), batch_size = 50, device=device)
    mse_error = F.mse_loss(bubbles_pred.detach().cpu(), bubbles_gt.detach().cpu(), reduction = 'sum')
    mse_error /= torch.numel(bubbles_pred)

    visual_qualitative_results = {
                                'psnr_error': psnr_error,
                                'ssim_error': ssim_error,
                                'fid_error': fid_error,
                                'mse_error': mse_error
                                }
    return visual_qualitative_results

def functional_quantitative(tool_path, bubbles_gt, bubbles_pred, info, project_path, output_path, output_path_gt):
    icp_model_path = os.path.join( project_path, 'checkpoints/masking_unet/masking_sdf_new_dataset_model_new_dataset_tools_E30_B32_LR0.00001.pth')
   # Get the data for ICP
    bubbles_gt_data = get_bubbles_data_TST(bubbles_gt, info)
    bubbles_pred_data = get_bubbles_data_TST(bubbles_pred, info)

    # Get the ICP results
    angles_gt, angles_p, _, _ = icp_dataset_cal(tool_path, bubbles_gt_data, output_path_gt, True, icp_model_path)
    angles_pred, angles_p2, _, _ = icp_dataset_cal(tool_path, bubbles_pred_data, output_path, True, icp_model_path)

    # Get the metrics
    gt_error, gt_cal_mean, gt_cal_std, gt_cal_acc_5, gt_cal_acc_10 = get_metrics(angles_p, angles_gt)
    pred_error, pred_cal_mean, pred_cal_std, pred_cal_acc_5, pred_cal_acc_10 = get_metrics(angles_p2, angles_pred)

    functional_results = {
                          'p_angles': angles_p,
                          'gt_results': {
                                        'gt_angles': angles_gt,
                                        'gt_error': gt_error,
                                        'gt_cal_mean': gt_cal_mean,
                                        'gt_cal_std': gt_cal_std,
                                        'gt_cal_acc_5': gt_cal_acc_5,
                                        'gt_cal_acc_10': gt_cal_acc_10
                                        },

                          'pred_results': {
                                        'pred_angles': angles_pred,
                                        'pred_error': pred_error,
                                        'pred_cal_mean': pred_cal_mean,
                                        'pred_cal_std': pred_cal_std,
                                        'pred_cal_acc_5': pred_cal_acc_5,
                                        'pred_cal_acc_10': pred_cal_acc_10
                                        }
                        }
    
    return functional_results

def diffusion_shift_stats(train_tools, train_path, model_results_path, gt_stats, pred_stats, args):
    if not os.path.exists(os.path.join(model_results_path, 'stats.pt')):
        for tool_name in train_tools:
            dataset = TST_diffusion_metrics(tool_name, os.path.join(train_path, 'bubbles'), os.path.join(train_path, 'gelslims'), train_results_path, 'cpu', diffusion_idx = 0, dataset_norm = False, gt_stats = gt_stats, pred_stats = pred_stats, both=args.both)
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            bubbles_gt_metrics, bubbles_pred_metrics = next(iter(dataloader))

            bubbles_gt_psum = bubbles_gt_metrics['gt_psum'].sum()
            bubbles_gt_psum_sq = bubbles_gt_metrics['gt_psum_sq'].sum()
            bubbles_gt_count = len(bubbles_gt_metrics['gt_shape'][0]) * bubbles_gt_metrics['gt_shape'][0][0] * bubbles_gt_metrics['gt_shape'][1][0] * bubbles_gt_metrics['gt_shape'][2][0] * bubbles_gt_metrics['gt_shape'][3][0]
            bubbles_gt_mean = bubbles_gt_psum / bubbles_gt_count
            bubbles_gt_std = torch.sqrt((bubbles_gt_psum_sq / bubbles_gt_count) - bubbles_gt_mean**2)

            bubbles_pred_psum = bubbles_pred_metrics['pred_psum'].sum()
            bubbles_pred_psum_sq = bubbles_pred_metrics['pred_psum_sq'].sum()
            bubbles_pred_count = len(bubbles_pred_metrics['pred_shape'][0]) * bubbles_pred_metrics['pred_shape'][0][0] * bubbles_pred_metrics['pred_shape'][1][0] * bubbles_pred_metrics['pred_shape'][2][0] * bubbles_pred_metrics['pred_shape'][3][0]
            bubbles_pred_mean = bubbles_pred_psum / bubbles_pred_count
            bubbles_pred_std = torch.sqrt((bubbles_pred_psum_sq / bubbles_pred_count) - bubbles_pred_mean**2)

            bubbles_gt_totat_psum += bubbles_gt_psum
            bubbles_gt_totat_psum_sq += bubbles_gt_psum_sq
            bubbles_pred_totat_psum += bubbles_pred_psum
            bubbles_pred_totat_psum_sq += bubbles_pred_psum_sq
            bubbles_gt_total_count += bubbles_gt_count
            bubbles_pred_total_count += bubbles_pred_count

            # print('Tool:', tool_name)
            # print('GT mean:', bubbles_gt_metrics['gt_mean'].mean())
            # print('GT std:', (bubbles_gt_metrics['gt_std']).mean())
            # print('Pred mean:', bubbles_pred_metrics['pred_mean'].mean())
            # print('Pred std:', (bubbles_pred_metrics['pred_std']).mean())
            # print('GT mean new:', bubbles_gt_mean)
            # print('GT std new:', bubbles_gt_std)
            # print('Pred mean new:', bubbles_pred_mean)
            # print('Pred std new:', bubbles_pred_std)
        
        bubbles_gt_mean = bubbles_gt_totat_psum / bubbles_gt_total_count
        bubbles_gt_std = torch.sqrt((bubbles_gt_totat_psum_sq / bubbles_gt_total_count) - bubbles_gt_mean**2)
        bubbles_pred_mean = bubbles_pred_totat_psum / bubbles_pred_total_count
        bubbles_pred_std = torch.sqrt((bubbles_pred_totat_psum_sq / bubbles_pred_total_count) - bubbles_pred_mean**2)

        print('Total GT mean:', bubbles_gt_mean)
        print('Total GT std:', bubbles_gt_std)
        print('Total Pred mean:', bubbles_pred_mean)
        print('Total Pred std:', bubbles_pred_std)

        gt_stats = (bubbles_gt_mean, bubbles_gt_std)
        pred_stats = (bubbles_pred_mean, bubbles_pred_std)

        torch.save({'gt_stats': gt_stats, 'pred_stats': pred_stats}, os.path.join(model_results_path, 'stats.pt'))
        return gt_stats, pred_stats
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TST Testing')
    parser.add_argument('--model', type=str, default='vq_vae', help='Model options: vq_vae, diffusion, diffusion_norm')
    parser.add_argument('--name', type=str, default='rot_flip', help='Model name')
    parser.add_argument('--both' , action='store_true')
    parser.add_argument('--visual_qual' , action='store_true')
    parser.add_argument('--visual_quant' , action='store_true')
    parser.add_argument('--functional' , action='store_true')
    parser.add_argument('--all_metrics' , action='store_true')
    args = parser.parse_args()

    # Model details
    method = args.model
    name = args.name
    project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    output_folder_name = 'results'
    print('Project Path:', project_path)
    print('Method:', method)
    print('Model:', name)

    # Get data paths
    datasets_path = os.path.join(project_path, "data")
    train_path = os.path.join(datasets_path, "test", "train_only")
    test_path = os.path.join(datasets_path, "test", "test_only")
    test_unseen_path = test_path

    if method == 'vq_vae':
        model_results_path = os.path.join(project_path, "checkpoints", "vq_vae", name)
        model_path = os.path.join(model_results_path, "checkpoint.pt")
        results_paths = [model_path, model_path, model_path]

    elif method == 'diffusion' or method == 'diffusion_norm':
        model_results_path = os.path.join(datasets_path, "testing", "diffusion_results", name)
        train_results_path = os.path.join(model_results_path, "results_samples", "train_only")
        test_results_path = os.path.join(model_results_path, "results_samples", "test_only")
        # results_paths = [train_results_path, test_results_path, test_results_path]
        results_paths = [test_results_path]
    
    else:
        model_results_path = os.path.join(datasets_path, "testing", method + "_results", name)
        train_results_path = os.path.join(model_results_path, "results_samples", "train_only")
        test_results_path = os.path.join(model_results_path, "results_samples", "test_only")
        results_paths = [train_results_path, test_results_path, test_results_path]
        results_paths = [test_results_path, test_results_path]

    # Get tools
    train_tools = torch.load(os.path.join(model_results_path, "tools.pt"))['train_tools']
    train_tools.sort()
    test_tools = torch.load(os.path.join(model_results_path, "tools.pt"))['train_tools']
    test_tools.sort()
    test_unseen_tools = torch.load(os.path.join(model_results_path, "tools.pt"))['test_tools']
    test_unseen_tools.sort()

    print('Train tools:', len(train_tools))
    for tool in train_tools:
        print(tool)
    print('-----------------------------')
    print('Test tools:', len(test_tools))
    for tool in test_tools:
        print(tool)
    print('-----------------------------')
    print('Test unseen tools:', len(test_unseen_tools))
    for tool in test_unseen_tools:
        print(tool)
    print('-----------------------------')


    # Get Shift Stats
    bubbles_gt_totat_psum = 0
    bubbles_gt_totat_psum_sq = 0
    bubbles_gt_total_count = 0
    bubbles_pred_totat_psum = 0
    bubbles_pred_totat_psum_sq = 0
    bubbles_pred_total_count = 0
    gt_stats = (0.0046, 0.0047)
    pred_stats = (0.0219, 0.0024)

    if method == 'diffusion_norm' or method == 'diffusion':
        gt_stats, pred_stats = diffusion_shift_stats(train_tools, train_path, model_results_path, gt_stats, pred_stats, args)

    stats = (gt_stats, pred_stats)
    datasets_stats = [stats, stats, stats]

    # Define datasets
    dataset_paths = [test_path, test_unseen_path]
    datasets = ["test", "test_unseen"]
    datasets_tools = [test_tools, test_unseen_tools]


    for j in range(len(dataset_paths)):
        dataset_path = dataset_paths[j]
        model_results_path = results_paths[j]
        dataset = datasets[j]
        dataset_tools = datasets_tools[j]
        stats = datasets_stats[j]
        print('Dataset:', dataset)

        for i in range(len(dataset_tools)):
            print(dataset_tools[i])
            tool_name = dataset_tools[i]
            tool_stl = datasets_path + '/tools_stls/' + tool_name + '.stl'
            
            if args.both:
                output_path = os.path.join(project_path, output_folder_name, "models", method + '_' + name  + '_both', dataset, tool_name)
            else:
                output_path = os.path.join(project_path, output_folder_name, "models", method + '_' + name, dataset, tool_name)
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Get All Data (including inference)
            gelslim_input, gelslim_ref, bubbles_gt, bubbles_pred, info = get_data(tool_name, os.path.join(dataset_path, 'bubbles'), os.path.join(dataset_path, 'gelslims'), 'cpu', method, model_results_path, stats, args.both)

            # Visual Qualitative Results
            if args.visual_qual or args.all_metrics:
                if not os.path.exists(output_path + '/visual_qualitative_results.pt'):
                    visual_qualitative_results = visual_qualitative(gelslim_input, gelslim_ref, bubbles_gt, bubbles_pred, info, project_path)
                    torch.save(visual_qualitative_results, output_path + '/visual_qualitative_results.pt')
                else:
                    visual_qualitative_results = torch.load(output_path + '/visual_qualitative_results.pt')

            # Visual Quantitative Results
            if args.visual_quant or args.all_metrics:
                if not os.path.exists(output_path + '/visual_quantitative_results.pt'):
                    visual_quantitative_results = visual_quantitative(bubbles_gt, bubbles_pred, 'cpu')
                    torch.save(visual_quantitative_results, output_path + '/visual_quantitative_results.pt')
                else:
                    visual_quantitative_results = torch.load(output_path + '/visual_quantitative_results.pt')

            # Quantitative Results
            if args.functional or args.all_metrics:
                output_path_gt = os.path.join(project_path, output_folder_name, "ground_truth", dataset, tool_name, 'functional_icp_results')
                if not os.path.exists(output_path_gt):
                    os.makedirs(output_path_gt)
                if not os.path.exists(output_path + '/functional_results.pt'):
                    functional_results = functional_quantitative(tool_stl, bubbles_gt, bubbles_pred, info, project_path, output_path + '/functional_icp_results', output_path_gt)
                    torch.save(functional_results, output_path + '/functional_results.pt')
                else:
                    functional_results = torch.load(output_path + '/functional_results.pt')
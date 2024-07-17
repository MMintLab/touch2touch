import argparse
import os
import open3d as o3d
import numpy as np
import re
import glob as glob
import torch
import transformations as tr
import random
import quaternion
import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import csv
import os
import copy
from torchvision.utils import make_grid
import io
import PIL.Image
from torchvision import transforms
from torch.utils.data import DataLoader
from touch2touch.vq_vae.datasets_loading import TactileTransferAllInfoAugment, TactileTransferDiffusion
from touch2touch.evaluation.camera_utils import tr_pointcloud, pack_o3d_pcd
from touch2touch.evaluation.pose_estimators import ICP2DPoseEstimator
from touch2touch.evaluation.point_cloud_utils import get_imprint_pc, project_depth_image, get_img_pixel_coordinates, W_LOW_LIMIT, H_LOW_LIMIT
import pickle 
import shutil
from scipy.spatial import KDTree
from touch2touch.evaluation.unet_model import UNet
from PIL import Image
import torch.nn.functional as F
import pytorch_volumetric as pv
import time
import torch.nn as nn

GELSLIM_MEAN = torch.tensor([-0.0082, -0.0059, -0.0066])
GELSLIM_STD = torch.tensor([0.0989, 0.0746, 0.0731])
BUBBLES_MEAN = torch.tensor([0.00382])
BUBBLES_STD = torch.tensor([0.00424])

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
    
def sensor_transforms(mode):
    '''
    mode: transform, transform_inv_paired
    '''
    if mode == "transform":
        gelslim_transform = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.Normalize(GELSLIM_MEAN, GELSLIM_STD)
                                            ])
        bubbles_transform = transforms.Compose([transforms.Resize((128,128)),
                                            # transforms.RandomRotation((180,180)),s
                                            transforms.Normalize(BUBBLES_MEAN, BUBBLES_STD)
                                            ])
    elif mode == "transform_inv_paired":
        gelslim_transform = transforms.Compose([transforms.Resize((320,427)),
                                                    unnormalize(GELSLIM_MEAN, GELSLIM_STD)])
        bubbles_transform = transforms.Compose([transforms.Resize((140,175)),
                                                transforms.RandomRotation((180,180)),
                                                unnormalize(BUBBLES_MEAN, BUBBLES_STD)])

    return gelslim_transform, bubbles_transform

def transformation_from_quat(quat):
    p_final = quat[:3].numpy()
    qi_final = quat[3:].numpy()
    q_final = np.quaternion(qi_final[3], qi_final[0], qi_final[1], qi_final[2])
    R_final = quaternion.as_rotation_matrix(q_final)
    T_final = np.eye(4)
    T_final[:3,:3] = R_final
    T_final[:3,3] = p_final

    return T_final

def transformation_3D_from_2D(angle, translation):
    '''
    angle: in degrees (rotation around x axis)
    translation: in meters (translation along y, z axis - plane perpendicular to x axis)
    '''
    angle = angle*(np.pi/180)
    translation = np.array(translation)
    pose = np.eye(4)
    pose[:3,:3] = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
    pose[1:3,3] = translation
    return pose

def obtain_bestPose(icp_estimator, pcd, pcd_all):
    rotations = np.arange(-np.pi/8, np.pi/8 + np.pi/36, np.pi/36).tolist()
    best_pose = None
    best_fitness = 0.0
    best_inlier_rmse = float('inf')
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    for r in rotations:
        R = np.eye(4)
        R[:3, :3] = mesh.get_rotation_matrix_from_xyz((r, 0, 0))
        init_tr = R
        pose, fitness, inlier_rmse = icp_estimator.estimate_pose(pcd, pcd_all, init_tr)
        al, be, ga = tr.euler_from_matrix(pose, 'rxyz')
        # print(al*(180/np.pi))
        if inlier_rmse < best_inlier_rmse:
            best_pose = pose
            best_r = r
            best_init_tr = init_tr
            al, be, ga = tr.euler_from_matrix(best_pose, 'rxyz')
            best_inlier_rmse = inlier_rmse
    
    return best_pose, best_init_tr

def obtain_bestPose_random(icp_estimator, pcd, pcd_all):
    rotations = np.linspace(-np.pi/8, np.pi/8, 10)
    best_pose = None
    best_fitness = 0.0
    best_inlier_rmse = float('inf')
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    for i in range(10):
        R = np.eye(4)
        r = np.random.choice(rotations)
        # print(r)
        R[:3, :3] = mesh.get_rotation_matrix_from_xyz((r, 0, 0))
        y = np.random.uniform(-0.01, 0.01)
        z = np.random.uniform(-0.01, 0.01)
        R[:3, 3] = np.array([0, y, z])
        init_tr = R
        pose, fitness, inlier_rmse = icp_estimator.estimate_pose(pcd, pcd_all, init_tr)
        al, be, ga = tr.euler_from_matrix(pose, 'rxyz')
        # print(al*(180/np.pi))
        if inlier_rmse < best_inlier_rmse:
            best_pose = pose
            best_r = r
            best_init_tr = init_tr
            al, be, ga = tr.euler_from_matrix(best_pose, 'rxyz')
            best_inlier_rmse = inlier_rmse
    
    return best_pose, best_init_tr

def RANSAC_place_fit(pts, thresh=0.05, minPoints=100, maxIteration=1000):
    """
    Find the best equation for a plane.

    :param pts: 3D point cloud as a `np.array (N,3)`.
    :param thresh: Threshold distance from the plane which is considered inlier.
    :param maxIteration: Number of maximum iteration which RANSAC will loop over.
    :returns:
    - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
    - `self.inliers`: points from the dataset considered inliers

    ---
    """
    n_points = pts.shape[0]
    best_eq = []
    best_inliers = []
    inliers = []
    equation = []

    for it in range(maxIteration):

        # Samples 3 random points
        id_samples = random.sample(range(0, n_points), 3)
        pt_samples = pts[id_samples]

        # We have to find the plane equation described by those 3 points
        # We find first 2 vectors that are part of this plane
        # A = pt2 - pt1
        # B = pt3 - pt1

        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecB = pt_samples[2, :] - pt_samples[0, :]

        # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
        vecC = np.cross(vecA, vecB)

        # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
        # We have to use a point to find k
        vecC = vecC / np.linalg.norm(vecC)
        k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
        plane_eq = [vecC[0], vecC[1], vecC[2], k]

        # Distance from a point to a plane
        # https://mathworld.wolfram.com/Point-PlaneDistance.html
        pt_id_inliers = []  # list of inliers ids
        dist_pt = (
            plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
        ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers
        inliers = best_inliers
        equation = best_eq

    return equation, inliers

def from_bubbles_to_pcd(depth_map_ref, depth_map, K, med_quat_pcd, med_quat_grasp, tresh = 0.007, percentile=None, ransac = False, thresh_ransac=0.0005, single = False):
    depth_map_ref_l = depth_map_ref[1].detach().numpy()
    depth_map_ref_r = depth_map_ref[0].detach().numpy()
    depth_map_l = depth_map[1].detach().numpy()
    depth_map_r = depth_map[0].detach().numpy()

    us, vs = get_img_pixel_coordinates(depth_map_ref_l)

    us = us + H_LOW_LIMIT
    vs = vs + W_LOW_LIMIT
    usvs = (us, vs)

    pcd_left = get_imprint_pc(depth_map_ref_l[0], depth_map_l[0], tresh, K[1].detach().numpy(), percentile=percentile, usvs=usvs)
    pcd_right = get_imprint_pc(depth_map_ref_r[0], depth_map_r[0], tresh, K[0].detach().numpy(), percentile=percentile, usvs=usvs)
    pcd_left_all_nc = project_depth_image(depth_map_l[0], K[1].detach().numpy(), usvs)
    pcd_left_all = np.concatenate([pcd_left_all_nc, np.zeros_like(pcd_left_all_nc)], axis=-1).reshape(-1,6)
    pcd_right_all_nc = project_depth_image(depth_map_r[0], K[1].detach().numpy(), usvs)
    pcd_right_all = np.concatenate([pcd_right_all_nc, np.zeros_like(pcd_right_all_nc)], axis=-1).reshape(-1,6)

    ref_pcd_left_all_nc = project_depth_image(depth_map_ref_l, K[1].detach().numpy(), usvs)
    ref_pcd_left_all = np.concatenate([ref_pcd_left_all_nc, np.zeros_like(ref_pcd_left_all_nc)], axis=-1).reshape(-1,6)
    ref_pcd_right_all_nc = project_depth_image(depth_map_ref_r, K[0].detach().numpy(), usvs)
    ref_pcd_right_all = np.concatenate([ref_pcd_right_all_nc, np.zeros_like(ref_pcd_right_all_nc,)], axis=-1).reshape(-1,6)

    T_left_pcd = transformation_from_quat(med_quat_pcd[1][0]) # Transformation from point cloud to med base
    T_right_pcd = transformation_from_quat(med_quat_pcd[0][0]) # Transformation from point cloud to med base
    T_grasp = transformation_from_quat(med_quat_grasp[0]) # Transformation from grasp to med base

    # Transformation from point cloud to grasp
    T_left = np.matmul(np.linalg.inv(T_grasp), T_left_pcd)
    R_left = T_left[:3,:3]
    p_left = T_left[:3, 3]
    T_right = np.matmul(np.linalg.inv(T_grasp), T_right_pcd)
    R_right = T_right[:3,:3]
    p_right = T_right[:3, 3]

    # print('T_left: ' + str(T_left))
    # print('T_right: ' + str(T_right))
    pcd_left = tr_pointcloud(pcd_left, R_left, p_left)
    pcd_right = tr_pointcloud(pcd_right, R_right, p_right)

    if ransac:
        _, pcd_left_inliers = RANSAC_place_fit(pcd_left[:,:3], thresh=thresh_ransac)
        _, pcd_right_inliers = RANSAC_place_fit(pcd_right[:,:3], thresh=thresh_ransac)
        pcd_left = pcd_left[pcd_left_inliers]
        pcd_right = pcd_right[pcd_right_inliers]

    pcd_left_all = tr_pointcloud(pcd_left_all, R_left, p_left)
    pcd_right_all = tr_pointcloud(pcd_right_all, R_right, p_right)
    ref_pcd_left_all = tr_pointcloud(ref_pcd_left_all, R_left, p_left)
    ref_pcd_right_all = tr_pointcloud(ref_pcd_right_all, R_right, p_right)
    # import pdb; pdb.set_trace()
    pcd = np.concatenate((pcd_left, pcd_right), axis=0)
    pcd_all = np.concatenate((pcd_left_all, pcd_right_all), axis=0)
    ref_pcd_all = np.concatenate([ref_pcd_left_all, ref_pcd_right_all], axis=0)

    if single:
        pcd = pcd_left
        pcd_all = pcd_left_all

    pcd[:,3] = 1
    pcd[:,5] = 1
    pcd_all[:,3:] = 0.25
    ref_pcd_all[:, 4] = 1
    ref_pcd_all[:, 5] = 1
    pcd_all = pcd_all[pcd_all[:, 2] <= 0.03]

    return pcd, pcd_all, ref_pcd_all

def from_bubbles_to_pcd_create_mask(depth_map, K, med_quat_pcd, med_quat_grasp, model_stl, tresh=0.007):
    model_pcd = model_stl.sample_points_uniformly(number_of_points=10000)
    depth_map_l = depth_map[1].detach().numpy()
    depth_map_r = depth_map[0].detach().numpy()

    us, vs = get_img_pixel_coordinates(depth_map_l)

    us = us + H_LOW_LIMIT
    vs = vs + W_LOW_LIMIT
    usvs = (us, vs)

    pcd_left_all_nc = project_depth_image(depth_map_l[0], K[1].detach().numpy(), usvs)
    pcd_left_all = np.concatenate([pcd_left_all_nc, np.zeros_like(pcd_left_all_nc)], axis=-1).reshape(-1,6)
    pcd_right_all_nc = project_depth_image(depth_map_r[0], K[1].detach().numpy(), usvs)
    pcd_right_all = np.concatenate([pcd_right_all_nc, np.zeros_like(pcd_right_all_nc)], axis=-1).reshape(-1,6)

    T_left_pcd = transformation_from_quat(med_quat_pcd[1][0]) # Transformation from point cloud to med base
    T_right_pcd = transformation_from_quat(med_quat_pcd[0][0]) # Transformation from point cloud to med base
    T_grasp = transformation_from_quat(med_quat_grasp[0]) # Transformation from grasp to med base

    # Transformation from point cloud to grasp
    T_left = np.matmul(np.linalg.inv(T_grasp), T_left_pcd)
    R_left = T_left[:3,:3]
    p_left = T_left[:3, 3]
    T_right = np.matmul(np.linalg.inv(T_grasp), T_right_pcd)
    R_right = T_right[:3,:3]
    p_right = T_right[:3, 3]

    pcd_left_all = tr_pointcloud(pcd_left_all, R_left, p_left)
    pcd_right_all = tr_pointcloud(pcd_right_all, R_right, p_right)
    pcd_all = np.concatenate((pcd_left_all, pcd_right_all), axis=0)

    # Create KDTree from pcd_all without color
    model_tree = KDTree(np.asarray(model_pcd.points))

    # PCD Left
    pcd_left_nc = pcd_left_all[:,:3]
    distances, _ = model_tree.query(pcd_left_nc)
    pcd_left = pcd_left_all[distances <= tresh]

    # PCD Right
    pcd_right_nc = pcd_right_all[:,:3]
    distances, _ = model_tree.query(pcd_right_nc)
    pcd_right = pcd_right_all[distances <= tresh]

    pcd = np.concatenate((pcd_left, pcd_right), axis=0)


    pcd[:,3] = 1
    pcd[:,5] = 1
    pcd_all[:,3:] = 0.25

    return pcd, pcd_all, pcd_left, pcd_right

def from_bubbles_to_pcd_create_mask_sdf(depth_map, K, med_quat_pcd, med_quat_grasp, model_obj, gt_bubble_pose, tresh=0.001):
    depth_map_l = depth_map[1].detach().numpy()
    depth_map_r = depth_map[0].detach().numpy()

    us, vs = get_img_pixel_coordinates(depth_map_l)

    us = us + H_LOW_LIMIT
    vs = vs + W_LOW_LIMIT
    usvs = (us, vs)

    pcd_left_all_nc = project_depth_image(depth_map_l[0], K[1].detach().numpy(), usvs)
    pcd_left_all = np.concatenate([pcd_left_all_nc, np.zeros_like(pcd_left_all_nc)], axis=-1).reshape(-1,6)
    pcd_right_all_nc = project_depth_image(depth_map_r[0], K[1].detach().numpy(), usvs)
    pcd_right_all = np.concatenate([pcd_right_all_nc, np.zeros_like(pcd_right_all_nc)], axis=-1).reshape(-1,6)

    T_left_pcd = transformation_from_quat(med_quat_pcd[1][0]) # Transformation from point cloud to med base
    T_right_pcd = transformation_from_quat(med_quat_pcd[0][0]) # Transformation from point cloud to med base
    T_grasp = transformation_from_quat(med_quat_grasp[0]) # Transformation from grasp to med base

    # Transformation from point cloud to grasp
    T_left = np.matmul(np.linalg.inv(T_grasp), T_left_pcd)
    R_left = T_left[:3,:3]
    p_left = T_left[:3, 3]
    T_right = np.matmul(np.linalg.inv(T_grasp), T_right_pcd)
    R_right = T_right[:3,:3]
    p_right = T_right[:3, 3]

    pcd_left_all = tr_pointcloud(pcd_left_all, R_left, p_left)
    pcd_right_all = tr_pointcloud(pcd_right_all, R_right, p_right)
    pcd_all = np.concatenate((pcd_left_all, pcd_right_all), axis=0)

    # SDF Mask
    sdf = pv.MeshSDF(model_obj)
    query_points_l = tr_pointcloud(pcd_left_all[:,:3], gt_bubble_pose[:3,:3], gt_bubble_pose[:3,3])
    query_points_r = tr_pointcloud(pcd_right_all[:,:3], gt_bubble_pose[:3,:3], gt_bubble_pose[:3,3])
    # query_points_l = pcd_left_all[:,:3]
    # query_points_r = pcd_right_all[:,:3]

    # call the sdf
    sdf_val_l, sdf_grad_l = sdf(query_points_l)
    sdf_val_r, sdf_grad_r = sdf(query_points_r)

    pcd_left = pcd_left_all[sdf_val_l <= tresh]
    pcd_right = pcd_right_all[sdf_val_r <= tresh]

    # import pdb; pdb.set_trace()

    pcd = np.concatenate((pcd_left, pcd_right), axis=0)


    pcd[:,3] = 1
    pcd[:,5] = 1
    pcd_all[:,3:] = 0.25

    return pcd, pcd_all, pcd_left, pcd_right

def from_pcd_to_bubbles(pcd_left, pcd_right, K, med_quat_pcd, med_quat_grasp, shape):
    T_left_pcd = transformation_from_quat(med_quat_pcd[1][0]) # Transformation from point cloud to med base
    T_right_pcd = transformation_from_quat(med_quat_pcd[0][0]) # Transformation from point cloud to med base
    T_grasp = transformation_from_quat(med_quat_grasp[0]) # Transformation from grasp to med base
    #check here
    T_grasp_left = np.matmul(np.linalg.inv(T_left_pcd), T_grasp)
    T_grasp_right = np.matmul(np.linalg.inv(T_right_pcd), T_grasp)
    pcd_grasp_l = tr_pointcloud(pcd_left, T_grasp_left[:3,:3], T_grasp_left[:3,3])
    pcd_grasp_r = tr_pointcloud(pcd_right, T_grasp_right[:3,:3], T_grasp_right[:3,3])

    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]

    us_l = np.clip((pcd_grasp_l[:,0] * fx[0] / pcd_grasp_l[:,2]) + cx[0] - H_LOW_LIMIT, 0, shape[1] - 1)
    vs_l = np.clip((pcd_grasp_l[:,1] * fy[0] / pcd_grasp_l[:,2]) + cy[0] - W_LOW_LIMIT, 0, shape[0] - 1)

    depth_img_l = np.zeros(shape)
    depth_img_l[vs_l.astype(int), us_l.astype(int)] = 1

    us_r = np.clip((pcd_grasp_r[:,0] * fx[1] / pcd_grasp_r[:,2]) + cx[1] - H_LOW_LIMIT, 0, shape[1] - 1)
    vs_r = np.clip((pcd_grasp_r[:,1] * fy[1] / pcd_grasp_r[:,2]) + cy[1]- W_LOW_LIMIT, 0, shape[0] - 1)

    depth_img_r = np.zeros(shape)
    depth_img_r[vs_r.astype(int), us_r.astype(int)] = 1

    depth_img = np.concatenate([np.expand_dims(depth_img_r, 0), np.expand_dims(depth_img_l, 0)], axis=0)
    return depth_img

def project_masked_depth_image(depth_img, mask, K, med_quat_pcd, med_quat_grasp):
    depth_map_l = depth_img[1].detach().numpy()
    depth_map_r = depth_img[0].detach().numpy()
    us, vs = get_img_pixel_coordinates(depth_map_l)

    us = us + H_LOW_LIMIT
    vs = vs + W_LOW_LIMIT
    usvs = (us, vs)

    pcd_left_all_nc = project_depth_image(depth_map_l, K[1].detach().numpy(), usvs)
    pcd_right_all_nc = project_depth_image(depth_map_r, K[0].detach().numpy(), usvs)
    pcd_left_all = np.concatenate([pcd_left_all_nc, np.zeros_like(pcd_left_all_nc)], axis=-1).reshape(-1,6)
    pcd_right_all = np.concatenate([pcd_right_all_nc, np.zeros_like(pcd_right_all_nc)], axis=-1).reshape(-1,6)

    pcd_left = pcd_left_all_nc[np.where(mask[1] == 1)] # shape (N, 3) where N is the number of imprint points where d>th
    pcd_left = np.concatenate([pcd_left, np.zeros_like(pcd_left)], axis=-1) # default color is black

    pcd_right = pcd_right_all_nc[np.where(mask[0] == 1)] # shape (N, 3) where N is the number of imprint points where d>th
    pcd_right = np.concatenate([pcd_right, np.zeros_like(pcd_right)], axis=-1) # default color is black

    T_left_pcd = transformation_from_quat(med_quat_pcd[1][0]) # Transformation from point cloud to med base
    T_right_pcd = transformation_from_quat(med_quat_pcd[0][0]) # Transformation from point cloud to med base
    T_grasp = transformation_from_quat(med_quat_grasp[0]) # Transformation from grasp to med base

    # Transformation from point cloud to grasp
    T_left = np.matmul(np.linalg.inv(T_grasp), T_left_pcd)
    R_left = T_left[:3,:3]
    p_left = T_left[:3, 3]
    T_right = np.matmul(np.linalg.inv(T_grasp), T_right_pcd)
    R_right = T_right[:3,:3]
    p_right = T_right[:3, 3]

    # print('T_left: ' + str(T_left))
    # print('T_right: ' + str(T_right))

    pcd_left = tr_pointcloud(pcd_left, R_left, p_left)
    pcd_right = tr_pointcloud(pcd_right, R_right, p_right)
    pcd_left_all = tr_pointcloud(pcd_left_all, R_left, p_left)
    pcd_right_all = tr_pointcloud(pcd_right_all, R_right, p_right)
    pcd = np.concatenate((pcd_left, pcd_right), axis=0)
    pcd_all = np.concatenate((pcd_left_all, pcd_right_all), axis=0)


    pcd[:,3] = 1
    pcd[:,5] = 1
    pcd_all[:,3:] = 0.25

    return pcd, pcd_all


def errorAngle(theta1, theta2):
    # THETA1: GT (degrees)
    # THETA2: INPUT (degrees)

    theta1 = np.array(theta1)
    theta2 = np.array(theta2)

    error = np.abs(theta2 - theta1)
    return error

def get_metrics(angles_ref, angles):
    # Ground Truth Metrics
    error = errorAngle(angles_ref, angles)
    cal_mean = np.mean(error)
    cal_std = np.std(error)
    cal_acc_5 = (np.abs(error) <= 5).sum()
    batch_size = len(angles_ref)
    cal_acc_5 *= (100/batch_size)
    cal_acc_10 = (np.abs(error) <= 10).sum()
    cal_acc_10 *= (100/batch_size)

    return error, cal_mean, cal_std, cal_acc_5, cal_acc_10

def angles_plot(debug_path, angles_gt, angles, show = False):
    _, indices = torch.sort(torch.tensor(angles_gt))
    sorted_angles_gt = torch.tensor(angles_gt)[indices]
    sorted_angles = torch.tensor(angles)[indices]
    plt.figure()
    plt.plot(sorted_angles_gt)
    plt.plot(sorted_angles)
    plt.xlabel('idx')
    plt.ylabel('Angles')
    plt.title('Angles')
    plt.legend(['Ground Truth', ' ICP Estimated'])
    plt.ylim([-90, 90])
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(debug_path, 'angles_plot.png'))
    plt.close()
    return

def visualize_bubbles_icp_results(object_mesh = None, bubbles_pcd = None, icp_pose = None, gt_pose = None , pcd_additional = None, initial = True, initial_pose = np.eye(4), side = 'left', show = False):
    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005, origin=np.array([0.0, 0.0, 0.0]))
    if object_mesh is not None:
        object_mesh.paint_uniform_color([0, 1, 1])  # Cyan color
        object_pcd_icp = copy.deepcopy(object_mesh.sample_points_uniformly(number_of_points=10000))
        object_mesh_gt = copy.deepcopy(object_mesh)
        object_mesh.rotate(np.linalg.inv(initial_pose[:3,:3]), center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=show)
    bubbles_pcd.paint_uniform_color([1, 0, 1])  # Magenta color
    vis.add_geometry(bubbles_pcd)
    if pcd_additional is not None:
        vis.add_geometry(pcd_additional)
    if initial:
        vis.add_geometry(object_mesh)
    if icp_pose is not None:
        object_pcd_icp.rotate(np.linalg.inv(icp_pose[:3,:3]), center=(0, 0, 0))
        object_pcd_icp.translate(-icp_pose[:3,3])
        object_pcd_icp.paint_uniform_color([1, 1, 0])
        vis.add_geometry(object_pcd_icp)
    if gt_pose is not None:
        object_mesh_gt.rotate(np.linalg.inv(gt_pose[:3,:3]), center=(0, 0, 0))
        object_mesh_gt.translate(-gt_pose[:3,3])
        object_mesh_gt.paint_uniform_color([0, 1, 0])
        object_mesh_gt = object_mesh_gt.sample_points_uniformly(number_of_points=10000)
        vis.add_geometry(object_mesh_gt)

    
    vis.add_geometry(frame_mesh)

    view_ctl = vis.get_view_control()
    camera_params = view_ctl.convert_to_pinhole_camera_parameters()
    if side == 'left':
        camera_params.extrinsic = np.array([
                                        [0, -1, 0, 0],
                                        [0, 0, 1, 0],
                                        [1, 0, 0, 0],
                                        [0, 0, 0, 1]
                                    ])
    else:
        camera_params.extrinsic = np.array([
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [-1, 0, 0, 0],
                                            [0, 0, 0, 1]
                                        ])

    view_ctl.convert_from_pinhole_camera_parameters(camera_params)
    view_ctl.scale(35)
    if show:
        vis.run()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    return image

def visualize_mesh(mesh_stl = None, frame_mesh = None, side = 'rigth', show = False):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # mesh_stl.paint_uniform_color([0.5, 0.5, 0.5])
    if mesh_stl is not None:
        vis.add_geometry(mesh_stl)

    if frame_mesh is not None:
        vis.add_geometry(frame_mesh)

    view_ctl = vis.get_view_control()
    camera_params = view_ctl.convert_to_pinhole_camera_parameters()
    
    if side == 'left':
        camera_params.extrinsic = np.array([
                                        [0, -1, 0, 0],
                                        [0, 0, 1, 0],
                                        [1, 0, 0, 0],
                                        [0, 0, 0, 1]
                                    ])
    else:
        camera_params.extrinsic = np.array([
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [-1, 0, 0, 0],
                                            [0, 0, 0, 1]
                                        ])

    view_ctl.convert_from_pinhole_camera_parameters(camera_params)
    view_ctl.scale(45)
    if show:
        vis.run()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return image

def visualize_image_title(image, title, debug=False, grayscale=False):
    plt.figure()
    if grayscale:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = np.array(image)
    plt.close()
    if debug:
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    return image

def save_grid_image(image, debug_path, name, show=False):
    image_grid = make_grid(image, nrow=10, normalize=True, scale_each=True)
    plt.figure()
    plt.imshow(image_grid.permute(1,2,0))
    plt.axis('off')
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(debug_path, name + '.png'))
    return

def get_bubbles_masks(bubbles_img, icp_model_path):
    scale = 1.0
    is_mask = False
    mask_values = [0, 255]
    full_img_size = (140, 175)
    min = -0.0116 #-1.0
    max = 0.0225 #1.0
    bubbles_img = bubbles_img.to('cpu')
    bubbles_img = (bubbles_img - min) / (max - min)
    # 1. Load UNET model
    net = UNet(n_channels=1, n_classes=2, bilinear=False)
    state_dict = torch.load(icp_model_path, map_location='cpu')
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    net.eval()
    # 2. Modify bubbles image to match the input processed by the UNET
    # load img
    images = []
    for i in range(bubbles_img.shape[0]):
        bubble_img_2 = (bubbles_img[i].squeeze(0).detach().numpy()*255).astype(np.uint8)
        bubble_img_2 = Image.fromarray(bubble_img_2)
        # preprocess
        w, h = bubble_img_2.size
        newW, newH = int(scale * w), int(scale * h)
        bubble_img_2 = bubble_img_2.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(bubble_img_2)
        if img.ndim == 2:
                img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0
        images.append(torch.as_tensor(img.copy()).float().contiguous())
    images = torch.stack(images, dim=0)
    # 3. Modify output mask
    out_masks = net(images)
    out_masks = out_masks.argmax(dim=1).unsqueeze(1).float()
    out_masks = F.interpolate(out_masks, (full_img_size[0], full_img_size[1]), mode='bilinear')
    return out_masks

def icp_dataset_cal(tool_path, bubbles_data,  output_path, icp_masking = False, icp_model_path = "", masking_gt = False, tresh=0.007, percentile = 0.4, thresh_ransac = None, diff_init = False, recalculate=False, verbose=False, max_iter = 200, stl_points = 5000, model_target = False):
    '''
    Inputs:
    - folder fo dataset
    - path to model stl
    - tresh
    - percentile
    - thresh_ransac
    - different initializations
    Outputs:
    Folder named with icp settings, inlcuding:
    - settings (csv in debug mode)
    - metrics (csv in debug mode)
    - angles and indexes
        On debud mode:
        - img for dept maps (6x5 - 30 samples from start, middle, end)
        - img filtered pointclouds (6x5 - 30 samples from start, middle, end)
        - img pcd + intil config (6x5 - 30 samples from start, middle, end)
        - img pcd + final config (6x5 - 30 samples from start, middle, end)
        - angles graph
    '''
    np.random.seed(0)
    ransac = True
    if thresh_ransac == None:
        ransac = False

    # Import tool
    mesh_stl_icp = o3d.io.read_triangle_mesh(tool_path)
    object_pcd_icp = mesh_stl_icp.sample_points_uniformly(number_of_points=stl_points)

    # Set ICP estimator
    projection_axis = np.array([1, 0, 0])
    #TODO: icp_estimator = chsel.CHSEL(...)
    icp_estimator = ICP2DPoseEstimator(obj_model=object_pcd_icp, object_mesh = mesh_stl_icp, view=False, verbose=False, projection_axis=projection_axis, is_model_target=model_target)
    icp_estimator.max_num_iterations = max_iter

    # ICP fitting for dataset
    angles = []
    angles_gt = []
    angles_init = []
    best_init_poses = []
    poses = []

    if verbose:
        if icp_masking:
            print('ICP Masking')
            print('icp_model_path: ' + icp_model_path)
        else:
            print('tresh: ' + str(tresh), end=' ')
            print('percentile: ' + str(percentile), end=' ')
            print('ransac: ' + str(ransac), end=' ')
            print('thresh_ransac: ' + str(thresh_ransac), end=' ')
            print('diff_init: ' + str(diff_init))

    # if icp_masking:
    #     output_path = os.path.join(output_path, 'icp_masking', os.path.basename(icp_model_path).replace('.pth', ''))
    # else:
    #     output_path = os.path.join(output_path, 'tresh_' + str(tresh) + '_percentile_' + str(percentile) + '_ransac_' + str(ransac) + '_thresh_ransac_' + str(thresh_ransac) + '_diff_init_' + str(diff_init))
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    angles_path = os.path.join(output_path, 'angles.pt')
    angles_gt_path = os.path.join(output_path, 'angles_gt.pt')
    angles_init_path = os.path.join(output_path, 'angles_init.pt')
    best_init_pose_path = os.path.join(output_path, 'best_init_pose.pt')
    poses_path = os.path.join(output_path, 'poses.pt')


    start_time = time.time()
    
    if os.path.exists(angles_path) and not recalculate:
            angles = torch.load(angles_path)
            angles_gt = torch.load(angles_gt_path)
            angles_init = torch.load(angles_init_path)
            best_init_poses = torch.load(best_init_pose_path)
            poses = torch.load(poses_path)
    else:
        for i in tqdm(range(len(bubbles_data))): 
            # import pdb; pdb.set_trace()
            bubbles_img = bubbles_data[i]['bubble_imprint'].to('cpu')
            bubbles_ref = bubbles_data[i]['bubble_depth_ref'].to('cpu')
            bubbles_K = bubbles_data[i]['K'].to('cpu')
            med_quat_pcd = bubbles_data[i]['bubbles_tr_quat'].to('cpu')
            med_quat_grasp = bubbles_data[i]['grasp_frame_quat'].to('cpu')
            angle_gt = bubbles_data[i]['theta'].to('cpu')*(180/np.pi)
            y = -bubbles_data[i]['x'].to('cpu')
            z = -bubbles_data[i]['y'].to('cpu')
            pose_gt = transformation_3D_from_2D(angle_gt, [y,z])

            # Point cloud from depth maps
            if icp_masking:
                if masking_gt:
                    obj = pv.MeshObjectFactory(tool_path)
                    obj.precompute_sdf()
                    gt_pcd , gt_pcd_all, _, _ = from_bubbles_to_pcd_create_mask_sdf(bubbles_ref - bubbles_img,bubbles_K,med_quat_pcd, med_quat_grasp, obj, pose_gt)
                else:
                    bubbles_mask = get_bubbles_masks(bubbles_img, icp_model_path)
                    gt_pcd, gt_pcd_all = project_masked_depth_image(bubbles_ref - bubbles_img, bubbles_mask, bubbles_K, med_quat_pcd, med_quat_grasp)
                    # import pdb; pdb.set_trace()
            else:
                gt_pcd , gt_pcd_all, _ = from_bubbles_to_pcd(bubbles_ref, bubbles_ref - bubbles_img,bubbles_K,med_quat_pcd, med_quat_grasp, tresh=tresh, percentile=percentile, ransac=False, thresh_ransac=thresh_ransac, single=False)

            # ICP pose calculation for bubbles sample
            if diff_init:
                pose_inv, best_init_pose_inv = obtain_bestPose_random(icp_estimator, gt_pcd, gt_pcd_all)
                best_init_pose = np.linalg.inv(best_init_pose_inv)
                al_init, _, _ = tr.euler_from_matrix(best_init_pose, 'rxyz')
                angle_init = al_init*(180/np.pi)
            else:
                pose_inv, _, _ = icp_estimator.estimate_pose(gt_pcd , gt_pcd_all)
                # icp_estimator.register()
                best_init_pose = np.eye(4)
                angle_init = 0

            pose = np.linalg.inv(pose_inv)
            al, _, _ = tr.euler_from_matrix(pose, 'rxyz')
            angle = al*(180/np.pi)
            

            angles.append(angle)
            angles_gt.append(angle_gt.item())
            angles_init.append(angle_init)
            best_init_poses.append(best_init_pose)
            poses.append(pose)

            # print('GT:', angle_gt.item())
            # print('Pred:', angle)
                    
    end_time = time.time()

    metrics_path = os.path.join(output_path, 'metrics_results.pt')
    if os.path.exists(metrics_path):
        metrics_results_saved = torch.load(metrics_path)
        if 'execution_time' in metrics_results_saved.keys():
            execution_time = metrics_results_saved['execution_time']
        else:
            execution_time = end_time - start_time
    else:
            execution_time = end_time - start_time

    if recalculate:
        execution_time = end_time - start_time
        
    print(f"Execution time: {execution_time} seconds")
    # Metrics
    error, cal_mean, cal_std, cal_acc_5, cal_acc_10 = get_metrics(angles_gt, angles)

    # ICP settings and results
    icp_settings = {
        'model': tool_path,
        'tresh': tresh,
        'percentile': percentile,
        'thresh_ransac': thresh_ransac,
        'diff_init': diff_init
    }
    
    metrics_results = {
        'mean': cal_mean,
        'std': cal_std,
        'acc_5': cal_acc_5,
        'acc_10': cal_acc_10,
        'execution_time': execution_time
    }

    torch.save(angles, angles_path)
    torch.save(angles_gt, angles_gt_path)
    torch.save(angles_init, angles_init_path)
    torch.save(best_init_poses, best_init_pose_path)
    torch.save(poses, poses_path)
    torch.save(icp_settings, os.path.join(output_path, 'icp_settings.pt'))
    torch.save(metrics_results, os.path.join(output_path, 'metrics_results.pt'))

    # Save icp_settings as CSV
    with open(os.path.join(output_path, 'icp_settings.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=icp_settings.keys())
        writer.writeheader()
        writer.writerow(icp_settings)

    # Save metrics_results as CSV
    with open(os.path.join(output_path, 'metrics_results.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics_results.keys())
        writer.writeheader()
        writer.writerow(metrics_results)

    # import pdb; pdb.set_trace()    

    return angles, angles_gt, metrics_results, icp_settings

def get_point_could_images(bubbles_data, icp_masking = False, icp_model_path = "", tresh=0.007, percentile = 0.4, thresh_ransac = None, diff_init = False, add_name = ''):
    ransac = True
    if thresh_ransac == None:
        ransac = False

    rotate_pc = transforms.Compose([transforms.RandomRotation((-90,-90)),
                                    transforms.Resize((128,128))])
    
    
    pcd_viz = []
    counter = 0

    for i in tqdm(range(len(bubbles_data)), desc='Point clouds'):
        bubbles_img = bubbles_data[i]['bubble_imprint'].to('cpu')
        bubbles_ref = bubbles_data[i]['bubble_depth_ref'].to('cpu')
        bubbles_K = bubbles_data[i]['K'].to('cpu')
        med_quat_pcd = bubbles_data[i]['bubbles_tr_quat'].to('cpu')
        med_quat_grasp = bubbles_data[i]['grasp_frame_quat'].to('cpu')

        # Point cloud from depth maps
        if icp_masking:
            bubbles_mask = get_bubbles_masks(bubbles_img, icp_model_path)
            gt_pcd, gt_pcd_all = project_masked_depth_image(bubbles_ref - bubbles_img, bubbles_mask, bubbles_K, med_quat_pcd, med_quat_grasp)
        else:
            gt_pcd , gt_pcd_all = from_bubbles_to_pcd(bubbles_ref, bubbles_ref - bubbles_img,bubbles_K,med_quat_pcd, med_quat_grasp, tresh=tresh, percentile=percentile, ransac=ransac, thresh_ransac=thresh_ransac, single=False)

        pcd_viz_image_l =  visualize_bubbles_icp_results(None, pack_o3d_pcd(gt_pcd), pcd_additional = pack_o3d_pcd(gt_pcd_all), initial = False, side = 'left')
        pcd_viz_image_r = visualize_bubbles_icp_results(None, pack_o3d_pcd(gt_pcd), pcd_additional = pack_o3d_pcd(gt_pcd_all), initial = False, side = 'right')
        
        pcd_viz_image_torch_l = rotate_pc(torch.from_numpy(np.asarray(pcd_viz_image_l)).permute(2,0,1).float()/255)
        pcd_viz_image_torch_r = rotate_pc(torch.from_numpy(np.asarray(pcd_viz_image_r)).permute(2,0,1).float()/255)
        pcd_viz_image_torch = torch.cat([pcd_viz_image_torch_l, pcd_viz_image_torch_r], dim=1)
        pcd_viz.append(pcd_viz_image_torch)
        counter += 1

    all_pcd_viz = torch.stack(pcd_viz, dim=0)

    return all_pcd_viz

def save_results_data(bubbles_data, tool_path, output_path ,icp_masking = False, icp_model_path = "", tresh=0.007, percentile = 0.4, thresh_ransac = None, diff_init = False, add_name = ''):
    ransac = True
    if thresh_ransac == None:
        ransac = False

    # Import tool
    mesh_stl_icp = o3d.io.read_triangle_mesh(tool_path)
    rotate_depth_map = transforms.RandomRotation((90,90))
    
    angles_viz = []
    angles_gt_viz = []
    angles_init_viz = []
    depth_maps_viz = []
    pcd_viz = []
    pcd_viz_init = []
    pcd_viz_final = []
    counter = 0

    # if icp_masking:
    #     output_path = os.path.join(output_path_0, 'icp_masking', os.path.basename(icp_model_path).replace('.pth', ''))
    # else:
    #     output_path = os.path.join(output_path_0, 'tresh_' + str(tresh) + '_percentile_' + str(percentile) + '_ransac_' + str(ransac) + '_thresh_ransac_' + str(thresh_ransac) + '_diff_init_' + str(diff_init))
    
    angles_path = os.path.join(output_path, 'angles.pt')
    angles_gt_path = os.path.join(output_path, 'angles_gt.pt')
    angles_init_path = os.path.join(output_path, 'angles_init.pt')
    best_init_pose_path = os.path.join(output_path, 'best_init_pose.pt')
    poses_path = os.path.join(output_path, 'poses.pt')
    angles = torch.load(angles_path)
    angles_gt = torch.load(angles_gt_path)
    angles_init = torch.load(angles_init_path)
    best_init_poses = torch.load(best_init_pose_path)
    poses = torch.load(poses_path)

    for i in tqdm(range(len(bubbles_data)), desc='Saving data'):
        bubbles_img = bubbles_data[i]['bubble_imprint'].to('cpu')
        bubbles_ref = bubbles_data[i]['bubble_depth_ref'].to('cpu')
        bubbles_K = bubbles_data[i]['K'].to('cpu')
        med_quat_pcd = bubbles_data[i]['bubbles_tr_quat'].to('cpu')
        med_quat_grasp = bubbles_data[i]['grasp_frame_quat'].to('cpu')
        angle_gt = bubbles_data[i]['theta'].to('cpu')*(180/np.pi)
        y = -bubbles_data[i]['x'].to('cpu')
        z = -bubbles_data[i]['y'].to('cpu')

        # Point cloud from depth maps
        if icp_masking:
            bubbles_mask = get_bubbles_masks(bubbles_img, icp_model_path)
            gt_pcd, gt_pcd_all = project_masked_depth_image(bubbles_ref - bubbles_img, bubbles_mask, bubbles_K, med_quat_pcd, med_quat_grasp)
        else:
            gt_pcd , gt_pcd_all = from_bubbles_to_pcd(bubbles_ref, bubbles_ref - bubbles_img,bubbles_K,med_quat_pcd, med_quat_grasp, tresh=tresh, percentile=percentile, ransac=False, thresh_ransac=thresh_ransac, single=False)

        if (i % 35 == 0):
            counter = 0
        if (counter <= 9):
            angle = angles[i]
            angle_init = angles_init[i]
            best_init_pose = best_init_poses[i]
            pose = poses[i]
            angles_init_viz.append(angle_init)
            angles_viz.append(angle)
            angles_gt_viz.append(angle_gt.item())
            pose_gt = transformation_3D_from_2D(angle_gt, [y,z])
            depth_maps_viz.append(torch.cat([rotate_depth_map(bubbles_img[1]), rotate_depth_map(bubbles_img[0])], dim=1))
            pcd_viz_image_l =  visualize_bubbles_icp_results(mesh_stl_icp, pack_o3d_pcd(gt_pcd), pcd_additional = pack_o3d_pcd(gt_pcd_all), initial = False, side = 'left')
            pcd_viz_image_r = visualize_bubbles_icp_results(mesh_stl_icp, pack_o3d_pcd(gt_pcd), pcd_additional = pack_o3d_pcd(gt_pcd_all), initial = False, side = 'right')
            pcd_viz_image = np.concatenate([pcd_viz_image_l, pcd_viz_image_r], axis=0)
            pcd_viz.append(pcd_viz_image)
            pcd_viz_init_image_l =  visualize_bubbles_icp_results(mesh_stl_icp, pack_o3d_pcd(gt_pcd), pcd_additional = pack_o3d_pcd(gt_pcd_all), initial = True, initial_pose=best_init_pose, side = 'left')
            pcd_viz_init_image_r = visualize_bubbles_icp_results(mesh_stl_icp, pack_o3d_pcd(gt_pcd), pcd_additional = pack_o3d_pcd(gt_pcd_all), initial = True, initial_pose=best_init_pose, side = 'right')
            pcd_viz_init_image = np.concatenate([pcd_viz_init_image_l, pcd_viz_init_image_r], axis=0)
            pcd_viz_init.append(pcd_viz_init_image)
            pcd_viz_final_image_l =  visualize_bubbles_icp_results(mesh_stl_icp, pack_o3d_pcd(gt_pcd), icp_pose=pose, gt_pose=pose_gt, pcd_additional = pack_o3d_pcd(gt_pcd_all), initial = False, side = 'left')
            pcd_viz_final_image_r = visualize_bubbles_icp_results(mesh_stl_icp, pack_o3d_pcd(gt_pcd), icp_pose=pose, gt_pose=pose_gt, pcd_additional = pack_o3d_pcd(gt_pcd_all), initial = False, side = 'right')
            pcd_viz_final_image = np.concatenate([pcd_viz_final_image_l, pcd_viz_final_image_r], axis=0)
            pcd_viz_final.append(pcd_viz_final_image)
            counter += 1

    temp_folder = os.path.join('temp', add_name, os.path.basename(output_path))
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    torch.save(depth_maps_viz, os.path.join(temp_folder,'depth_maps_viz.pt'))
    torch.save(pcd_viz, os.path.join(temp_folder,'pcd_viz.pt'))
    torch.save(pcd_viz_init, os.path.join(temp_folder,'pcd_viz_init.pt'))
    torch.save(pcd_viz_final, os.path.join(temp_folder,'pcd_viz_final.pt'))
    torch.save(angles_init_viz, os.path.join(temp_folder,'angles_init.pt'))
    torch.save(angles_viz, os.path.join(temp_folder,'angles.pt'))
    torch.save(angles_gt_viz, os.path.join(temp_folder,'angles_gt.pt'))

    return

def save_results_images(output_path, icp_masking = False, icp_model_path = "", tresh=0.007, percentile = 0.4, thresh_ransac = None, diff_init = False, add_name = '', show = False):
    ransac = True
    if thresh_ransac == None:
        ransac = False

    # if icp_masking:
    #     output_path = os.path.join(output_path_0, 'icp_masking', os.path.basename(icp_model_path).replace('.pth', ''))
    # else:
    #     output_path = os.path.join(output_path_0, 'tresh_' + str(tresh) + '_percentile_' + str(percentile) + '_ransac_' + str(ransac) + '_thresh_ransac_' + str(thresh_ransac) + '_diff_init_' + str(diff_init))

    temp_folder = os.path.join('temp', add_name, os.path.basename(output_path))
    depth_maps_viz = torch.load(os.path.join(temp_folder,'depth_maps_viz.pt'))
    # import pdb; pdb.set_trace()
    pcd_viz = torch.load(os.path.join(temp_folder,'pcd_viz.pt'))
    pcd_viz_init = torch.load(os.path.join(temp_folder,'pcd_viz_init.pt'))
    pcd_viz_final = torch.load(os.path.join(temp_folder,'pcd_viz_final.pt'))
    angles_init_viz = torch.load(os.path.join(temp_folder,'angles_init.pt'))
    angles_gt_viz = torch.load(os.path.join(temp_folder,'angles_gt.pt'))
    angles_viz = torch.load(os.path.join(temp_folder,'angles.pt'))
    angles_gt = torch.load(os.path.join(output_path, 'angles_gt.pt'))
    angles = torch.load(os.path.join(output_path, 'angles.pt'))
    for i in tqdm(range(len(depth_maps_viz)), desc = 'Saving images'):
        depth_maps_viz_image = visualize_image_title(depth_maps_viz[i].detach().numpy().transpose(1,2,0), 'GT: ' + str(round(angles_gt_viz[i], 2)), grayscale=True)[:, 650:1300, :]
        depth_maps_viz_image_torch = torch.from_numpy(depth_maps_viz_image).permute(2,0,1).float()/255
        depth_maps_viz[i] = depth_maps_viz_image_torch
        pcd_viz_image = visualize_image_title(pcd_viz[i], 'GT: ' + str(round(angles_gt_viz[i], 2)))[:, 650:1300, :]
        pcd_viz_image_torch = torch.from_numpy(pcd_viz_image).permute(2,0,1).float()/255
        pcd_viz[i] = pcd_viz_image_torch
        pcd_viz_init_image = visualize_image_title(pcd_viz_init[i], 'Init: ' + str(round(angles_init_viz[i], 2)))[:, 650:1300, :]
        pcd_viz_init_image_torch = torch.from_numpy(pcd_viz_init_image).permute(2,0,1).float()/255
        pcd_viz_init[i] = pcd_viz_init_image_torch
        pcd_viz_final_image = visualize_image_title(pcd_viz_final[i], 'ICP: ' + str(round(angles_viz[i], 2)))[:, 650:1300, :]
        pcd_viz_final_image_torch = torch.from_numpy(pcd_viz_final_image).permute(2,0,1).float()/255
        pcd_viz_final[i] = pcd_viz_final_image_torch
    
    angles_plot(output_path, angles_gt, angles, show=show)
    save_grid_image(depth_maps_viz, output_path, 'depth_maps_viz', show=show)
    save_grid_image(pcd_viz, output_path, 'pcd_viz', show=show)
    save_grid_image(pcd_viz_init, output_path, 'pcd_viz_init', show=show)
    save_grid_image(pcd_viz_final, output_path, 'pcd_viz_final', show=show)
    shutil.rmtree(temp_folder)  # Remove the temp folder
    
    return

def fine_tune_icp_dataset_cal(tool_path, bubbles_data, output_path):
    treshs = [0.005, 0.007, 0.009, 0.011]
    percentiles = [None, 0.4]
    threshs_ransac = [None, 0.0001, 0.0005, 0.001, 0.005, 0.01, 1.0]
    diff_inits = [False] #, True]
    best_error = 1000
    counter = 0
    icp_settings = []
    for tresh in treshs:
        for percentile in percentiles:
            for thresh_ransac in threshs_ransac:
                for diff_init in diff_inits:
                    angles, angles_gt, metrics_results, icp_settings_run = icp_dataset_cal(tool_path, bubbles_data, tresh, output_path, percentile = percentile, thresh_ransac = thresh_ransac, diff_init = diff_init, verbose=True)
                    icp_settings.append({'name': 'trial_' + str(counter), 'tresh': tresh, 'percentile': percentile, 'ransac_tresh': thresh_ransac, 'angle_error_mean': metrics_results['mean'], 'angle_error_std': metrics_results['std']})
                    error = np.abs(metrics_results['mean']) + metrics_results['std']

                    if error < best_error:
                        best_error = error
                        best_mean = metrics_results['mean']
                        best_std = metrics_results['std']
                    counter += 1

    best_icp_settings = {'name': 'Best: t_' + str(counter),'tresh': tresh, 'percentile': percentile, 'ransac_tresh': thresh_ransac, 'angle_error_mean': best_mean, 'angle_error_std': best_std}
    icp_settings.append(best_icp_settings)
    file_path = os.path.join(output_path, '_icp_settings.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(icp_settings, file)
    return

def best_icp_settings(dataset, tool):
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'processed_data')
    icp_settings_path = os.path.join(root_dir, 'bubbles', 'icp_fine_tuning', dataset, tool,'_icp_settings.pkl')

    with open(icp_settings_path, 'rb') as f:
        data = pickle.load(f)

    idx = min(range(len(data)), key=lambda i: data[i]['angle_error_mean'] + data[i]['angle_error_std'])
    tresh = data[idx]['tresh']
    percentile = data[idx]['percentile']
    ransac_tresh = data[idx]['ransac_tresh']
    return tresh, percentile, ransac_tresh, idx, data

def vis_best_icp(dataset, tool):
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'processed_data')
    bubbles_path = os.path.join(root_dir, 'bubbles', dataset, 'bubble_style_transfer_dataset_bubbles_' + tool)
    tool_path = os.path.join(root_dir, 'tools_stls', tool + '.stl')
    icp_results_path = os.path.join(root_dir, 'bubbles', 'icp_fine_tuning', dataset, tool)

    tresh, percentile, ransac_tresh, idx, data = best_icp_settings(dataset, tool)

    print('tresh: ', tresh, 'percentile: ', percentile, 'ransac_tresh: ', ransac_tresh, 'angle_error_mean: ', data[idx]['angle_error_mean'], 'angle_error_std: ', data[idx]['angle_error_std'])

    bubbles_data = get_bubbles_data_direct(bubbles_path)
    save_results_data(bubbles_data, tool_path, icp_results_path, tresh, percentile, ransac_tresh)
    save_results_images(icp_results_path, tresh, percentile, ransac_tresh, show=True)
    return
    

def get_bubbles_data_direct(bubbles_path, len_data = 100):
    bubbles_files = sorted(glob.glob(os.path.join(bubbles_path, '*.pt')), key=sort_order)
    bubbles_files = bubbles_files[:len_data]
    bubbles_data = []
    for i, data_file in tqdm(enumerate(bubbles_files)):
        bubbles_data.append(torch.load(data_file))
        
    return bubbles_data

def get_bubbles_data_TST(bubbles_gt, info, b_transforms=nn.Identity()):
    bubbles_data = []
    for i in range(bubbles_gt.shape[0]):
        bubbles_data.append({
            'bubble_imprint': (b_transforms(bubbles_gt[i])).squeeze(0),
            'bubble_depth_ref': info['bubbles_data']['bubble_depth_ref'][i],
            'K': info['bubbles_data']['K'][i],
            'bubbles_tr_quat': info['bubbles_data']['bubbles_tr_quat'][i],
            'grasp_frame_quat': info['bubbles_data']['grasp_frame_quat'][i],
            'theta': info['bubbles_data']['theta'][i],
            'x': info['bubbles_data']['x'][i],
            'y': info['bubbles_data']['y'][i]
        })

        # save_grid_image(info['bubbles_data']['bubble_depth_ref'][i], '', 'bubbles_gt', show=True)

        # import pdb; pdb.set_trace()
    return bubbles_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICP dataset')
    parser.add_argument('--dataset_path', type=str, default='dataset_path', help='path to dataset')
    parser.add_argument('--tool_path', type=str, default='tool_path', help='path to model stl')
    parser.add_argument('--tresh', type=float, default=0.007, help='tresh')
    parser.add_argument('--percentile', type=float, default=0.4, help='percentile')
    parser.add_argument('--thresh_ransac', type=float, default=None, help='thresh_ransac')
    parser.add_argument('--diff_init', action='store_true')
    parser.add_argument('--recalculate', action='store_true')
    parser.add_argument('--icp_masking', action='store_true')

    args = parser.parse_args()

    tool_name = os.path.basename(args.dataset_path)
    tool_name = tool_name.replace('bubble_style_transfer_dataset_bubbles_', '')
    output_path = os.path.join(os.path.dirname(os.path.dirname(args.dataset_path)), 'icp_TST', os.path.basename(os.path.dirname(args.dataset_path)), tool_name)
    # import pdb; pdb.set_trace()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gelslim_transform, bubbles_transform = sensor_transforms('transform')
    
    bubbles_dataset_path = args.dataset_path
    gelslim_dataset_path = args.dataset_path.replace('bubble_style_transfer_dataset_bubbles', 'gelslim_style_transfer_dataset_gelslim')
    gelslim_dataset_path = gelslim_dataset_path.replace('bubbles/bubbles_', 'gelslims/gelslim_')

    # import pdb; pdb.set_trace()
    
    dataset = TactileTransferAllInfoAugment(bubbles_dataset_path, gelslim_dataset_path, device, bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform, single=False)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    gelslim_inputs, bubbles_gt, info = next(iter(dataloader))
    _, b_transforms = sensor_transforms('transform_inv_paired')
    bubbles_data = get_bubbles_data_TST(bubbles_gt, info, b_transforms)

#     diffusion_results_path = '/home/samanta/Documents/UMICH/UMICH_Research_Codes/tactile_style_transfer/tactile_style_transfer/scripts/diffusion_results/raw_data'
#     dataset = TactileTransferDiffusion(tool_name,  bubbles_dataset_path, gelslim_dataset_path, diffusion_results_path, device, diffusion_idx = 0, bubbles_transform=bubbles_transform, gelslim_transform=gelslim_transform)
#     # import pdb; pdb.set_trace()
#     dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
#     gelslim_inputs, bubbles_gt, bubbles_prediction, info = next(iter(dataloader))
#     _, b_transforms = sensor_transforms('transform_inv_paired')
#     bubbles_data = get_bubbles_data_TST(bubbles_prediction, info, b_transforms)
#     # import pdb; pdb.set_trace()
#     # bubbles_data = get_bubbles_data_direct(args.dataset_path, len_data = 100)
    print('ICP dataset:' + tool_name)
    # import pdb; pdb.set_trace()
    icp_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "Pytorch-UNet/models/model_train-tools_E30_B8_LR0.00001.pth")
    angles, angles_gt, metrics_results, icp_settings = icp_dataset_cal(args.tool_path, bubbles_data, output_path, args.icp_masking, icp_model_path, args.tresh, args.percentile, args.thresh_ransac,  args.diff_init, recalculate=args.recalculate, verbose=True)
    # save_results_data(bubbles_data, args.tool_path, output_path, args.icp_masking, icp_model_path, args.tresh, args.percentile, args.thresh_ransac,  args.diff_init)
    # save_results_images(output_path, args.icp_masking, icp_model_path, args.tresh, args.percentile, args.thresh_ransac,  args.diff_init)


    
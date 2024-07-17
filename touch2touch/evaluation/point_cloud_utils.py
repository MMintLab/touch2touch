import numpy as np
import torch
from matplotlib import cm

W_LOW_LIMIT = 20
W_HIGH_LIMIT = 160
H_LOW_LIMIT = 25
H_HIGH_LIMIT = 200
ORIGINAL_SHAPE = (171, 224)
PROCESSED_SHAPE = (W_HIGH_LIMIT - W_LOW_LIMIT, H_HIGH_LIMIT - H_LOW_LIMIT)

def project_depth_image(depth_img, K, usvs=None):
    """
    Return xyz coordinates in the optical frame (z-axis is the camera axis)
    Args:
        depth_img: (...,w,h) array or tensor
        K: Intrinsic matrix (...,3,3) array or tensor
    Returns: (..., w, h, 3) array of the (x,y,z) coordiantes for each pixel in the image
    """
    is_tensor = torch.is_tensor(depth_img)
    # reshape to make the operation batched
    input_size = depth_img.shape
    if len(input_size) > 2:
        depth_img = depth_img.reshape(-1, *input_size[-2:])
        K = K.reshape(-1, 3, 3)

    if is_tensor and not torch.is_tensor(K):
        K = torch.from_numpy(K)

    if usvs is None:
        us, vs = get_img_pixel_coordinates(depth_img)
    else:
        us, vs = usvs
        us = us.reshape(depth_img.shape)
        vs = vs.reshape(depth_img.shape)

    xs, ys, zs = project_depth_points(us, vs, depth_img, K)
    if is_tensor:
        # pytorch tensors
        img_xyz = torch.stack([xs, ys, zs], dim=-1)
    else:
        # numpy array
        img_xyz = np.stack([xs, ys, zs], axis=-1)
    img_xyz = img_xyz.reshape(*input_size, 3)
    return img_xyz


def get_img_pixel_coordinates(img):
    # img: tensor or array of size (..., w, h)
    input_shape = img.shape
    if len(input_shape) >= 2:
        img = img.reshape(-1, *input_shape[-2:])
    is_tensor = torch.is_tensor(img)
    w, h = img.shape[-2:]
    batch_size = int(img.shape[0])
    if is_tensor:
        # pytorch tensors
        vs, us = torch.meshgrid(torch.arange(w), torch.arange(h))
        us = us.unsqueeze(0).repeat_interleave(batch_size, dim=0).reshape(img.shape).to(img.device)
        vs = vs.unsqueeze(0).repeat_interleave(batch_size, dim=0).reshape(img.shape).to(img.device)
    else:
        # numpy arrays
        us, vs = np.meshgrid(np.arange(h), np.arange(w))
        us = np.repeat(np.expand_dims(us, 0), batch_size, axis=0).reshape(img.shape)  # stack as many us as num_batches
        vs = np.repeat(np.expand_dims(vs, 0), batch_size, axis=0).reshape(img.shape)  # stack as many vs as num_batches
    us = us.reshape(*input_shape)
    vs = vs.reshape(*input_shape)
    return us, vs


def project_depth_points(us, vs, depth, K):
    """
    Return xyz coordinates in the optical frame (z-axis is the camera axis)
    Args:
        us: (scalar or any shaped array) image height coordinates (top is 0)
        vs: (scalar or any shaped array, matching us) image width coordinates (left is 0)
        depth: (scalar or any shaped array, matching us) image depth coordinates
        K: Intrinsic matrix (3x3)
    Returns: (scalar or any shaped array) of the (x,y,z) coordinates for each given point
    """
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    try:
        # if input is batched
        w, h = us.shape[-2:]
        num_batch = np.prod(us.shape[:-2])
        is_tensor = torch.is_tensor(K)
        if is_tensor:
            cx = cx.unsqueeze(-1).unsqueeze(-1).repeat_interleave(w, dim=-2).repeat_interleave(h, dim=-1)
            cy = cy.unsqueeze(-1).unsqueeze(-1).repeat_interleave(w, dim=-2).repeat_interleave(h, dim=-1)
            fx = fx.unsqueeze(-1).unsqueeze(-1).repeat_interleave(w, dim=-2).repeat_interleave(h, dim=-1)
            fy = fy.unsqueeze(-1).unsqueeze(-1).repeat_interleave(w, dim=-2).repeat_interleave(h, dim=-1)
        else:
            # numpy case
            cx = np.repeat(np.repeat(np.expand_dims(cx, [-2, -1]), w, axis=-2), h, axis=-1)
            cy = np.repeat(np.repeat(np.expand_dims(cy, [-2, -1]), w, axis=-2), h, axis=-1)
            fx = np.repeat(np.repeat(np.expand_dims(fx, [-2, -1]), w, axis=-2), h, axis=-1)
            fy = np.repeat(np.repeat(np.expand_dims(fy, [-2, -1]), w, axis=-2), h, axis=-1)
    except:
        # code below also works for non-batched inputs
        pass
    
    xs = (us - cx) * depth / fx
    ys = (vs - cy) * depth / fy
    zs = depth
    return xs, ys, zs

def get_imprint_mask(dth_img_1, dth_img_2, threshold, percentile=None, num_mask_points=None):
    """
    Args:
        dth_img_1: <np.ndarray> reference depth image
        dth_img_2: <np.ndarray> deformed depth_image
        threshold: distance to be considered that the point has changed
    Returns: <np.ndarray> binary mask where 1 are the imprint points
    """
    delta_depth = (dth_img_1 - dth_img_2).squeeze()
    imprint_mask = np.zeros_like(delta_depth)
    imprint_mask[np.where(delta_depth > threshold)] = 1
    # import pdb; pdb.set_trace()
    if percentile is not None or num_mask_points is not None:
        #
        num_points = np.prod(delta_depth.shape[-2:]) # consider all points. # TODO: extend for a batched case
        if percentile is not None:
            k = int(np.floor(percentile * num_points))
        if num_mask_points is not None:
            k = num_mask_points
        _indxs = np.argpartition(delta_depth.ravel(), delta_depth.size-k)[-k:]
        top_k_indxs = np.column_stack(np.unravel_index(_indxs, delta_depth.shape))
        percentile_mask = np.zeros_like(delta_depth)
        percentile_mask[top_k_indxs[:, 0], top_k_indxs[:, 1]] = 1
        imprint_mask = imprint_mask * percentile_mask # combine both masks
    return imprint_mask

def get_imprint_pc(dth_img_1, dth_img_2, threshold, K, K2=None, percentile=None, usvs=None):
    """
    Args:
        dth_img_1: <np.ndarray> reference depth image
        dth_img_2: <np.ndarray> deformed depth_image
        threshold: distance to be considered that the point has changed
        K: Intrinsic matrix for dth_img_1
        K2: Intrinsic matrix for dth_img_2 (in case they are different)
    Returns: <np.ndarray> (N, 3) containing the xyz coordinates of the imprint points
    """
    if K2 is None:
        K2 = K
    if usvs is None:
        xyz_1 = project_depth_image(dth_img_1, K)
        xyz_2 = project_depth_image(dth_img_2, K2)
    else:
        xyz_1 = project_depth_image(dth_img_1, K, usvs)
        xyz_2 = project_depth_image(dth_img_2, K2, usvs)

    imprint_mask = get_imprint_mask(dth_img_1, dth_img_2, threshold, percentile=percentile)
    # import pdb; pdb.set_trace()
    imprint_xyz = xyz_2[np.where(imprint_mask == 1)] # shape (N, 3) where N is the number of imprint points where d>th
    imprint_pc = np.concatenate([imprint_xyz, np.zeros_like(imprint_xyz)], axis=-1) # default color is black
    return imprint_pc

def get_bubble_deformation_pc(bubble_deformed_depth, bubble_ref_depth, K, cmap='jet', filtered=False):
    # bubble_deformed_depth and bubble_ref_depth are expected to be (..., w, h, 1)
    def_xyz = project_depth_image(bubble_deformed_depth.squeeze(-1), K=K)
    # ref_xyz = project_depth_image(bubble_ref_depth, K=K)
    colors = get_bubble_color(bubble_deformed_depth.squeeze(-1), color_mode='delta_depth', ref_depth_ar=bubble_ref_depth.squeeze(-1),
                            cmap=cmap)
    deformation_pc = np.concatenate([def_xyz, colors], axis=-1)
    if filtered:
        deformation_pc = process_bubble_img(deformation_pc)
    return deformation_pc # preserves spatial correspondences

def get_bubble_color(depth_ar, color_mode='delta_depth', ref_depth_ar=None, cmap='jet'):
    # depth_ar and ref_depth_ar expected shapes are (w, h)
    color_mode_options = ['camera_depth', 'delta_depth']
    if type(cmap) is str:
        cmap = cm.get_cmap(cmap)
    depth_values = depth_ar
    # import pdb; pdb.set_trace()
    if color_mode == 'camera_depth':
        max_depth = 0.15
        min_depth = 0.05
        mapped_depth = ((depth_values - min_depth) / (max_depth - min_depth) - 1) * (-1)  # between (0,1)
        # max_depth = np.max(depth_values)
        colors = cmap(mapped_depth)[..., :3]
    elif color_mode == 'delta_depth':
        max_delta_depth = 0.025
        min_delta_depth = -0.005
        if ref_depth_ar is None:
            delta_depth = depth_ar # we direclty provide the delta depth.
        else:
            ref_depth = ref_depth_ar
            delta_depth = ref_depth - depth_values
        mapped_depth = ((delta_depth - min_delta_depth) / (max_delta_depth - min_delta_depth))  # between (0,1)
        colors = cmap(mapped_depth)[..., :3]
    else:
        raise NotImplementedError(
            'Color Mode {} implemented yet. Please, select instead within {}'.format(color_mode, color_mode_options))
    return colors


def process_bubble_img(bubble_img):
    # bubble_img: numpy array or torch tensor of bubble images (..., w, h, num_channels)
    # remove the noisy areas of the images:
    bubble_img_out = bubble_img[..., W_LOW_LIMIT:W_HIGH_LIMIT, H_LOW_LIMIT:H_HIGH_LIMIT, :]
    return bubble_img_out


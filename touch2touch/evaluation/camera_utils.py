import numpy as np
import open3d as o3d

def pack_o3d_pcd(pc_array):
    """
    Given a pointcloud as an array (N,6), convert it to open3d PointCloud
    Args:
        pc_array: <np.ndarray> of size (N,6) containing the point in x y z r g b
    Returns: o3d.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array[:, :3])
    if pc_array.shape[-1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(pc_array[:, 3:7])
    
    return pcd

def tr_pointcloud(pc, R, t):
    """
    Transform a point cloud given the homogeneous transformation represented by R,t
    R and t can be seen as a tranformation new_frame_X_old_frame where pc is in the old_frame and we want it to be in the new_frame
    Args:
        pc:
        R:
        t:
    Returns:
    """
    pc_xyz = pc[:, :3]
    pc_xyz_tr = pc_xyz@R.T+ t
    # handle RGB info held in the other columns
    if pc.shape[-1] > 3:
        pc_rgb = pc[:, 3:7]
        pc_tr = np.concatenate([pc_xyz_tr, pc_rgb], axis=-1)
    else:
        pc_tr = pc_xyz_tr
    return pc_tr

def view_pointcloud(pc, frame=False, scale=1.):
    """
    Simple visualization of pointclouds
    Args:
        pc: pointcloud array or a list of pointcloud arrays
    Returns:
    """
    pcds = []
    if type(pc) is not list:
        pc = [pc]
    for pc_i in pc:
        if (not isinstance(pc_i, o3d.geometry.PointCloud)) and (not isinstance(pc_i, o3d.geometry.TriangleMesh)):
            pcd_i = pack_o3d_pcd(pc_i)
        else:
            pcd_i = pc_i
        pcds.append(pcd_i)
    view_pcd(pcds, frame=frame, scale=scale)


def view_pcd(pcds, frame=False, scale=1.0):
    if type(pcds) is not list:
        pcds = [pcds]
    first_pcd = pcds[2]
    first_points = np.asarray(first_pcd.points)
    # last_pcd = pcds[2]
    # last_points = np.asarray(last_pcd.points)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.base_color = np.array([1, 1, 1, .5])
    if frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.5 * np.std(first_points),
                                                                       origin=[0, 0, 0])
        # mesh_frame_last = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.5 * np.std(last_points),
                                                                    #    origin=[0, 0, 0])
        pcds.append(mesh_frame)
        # pcds.append(mesh_frame_last)
    o3d.visualization.draw_geometries(pcds)

class term_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = "\033[1;31m"
    BLUE = "\033[1;34m"
    CYAN = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    REVERSE = "\033[;7m"
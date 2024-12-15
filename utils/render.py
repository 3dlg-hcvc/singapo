import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import trimesh
import pyrender
import numpy as np
import open3d as o3d
from copy import deepcopy
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from utils.refs import semantic_color_ref, graph_color_ref, joint_color_ref

def get_rotation_axis_angle(k, theta):
    '''
    Rotation matrix converter from axis-angle using Rodrigues' rotation formula

    Args:
        k (np.ndarray): 3D unit vector representing the axis to rotate about.
        theta (float): Angle to rotate with in radians.

    Returns:
        R (np.ndarray): 3x3 rotation matrix.
    '''
    if np.linalg.norm(k) == 0.:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def rescale_axis(jtype, axis_d, axis_o, box_center):
    '''
    Function to rescale the axis for rendering
    
    Args:
    - jtype (int): joint type
    - axis_d (np.array): axis direction
    - axis_o (np.array): axis origin
    - box_center (np.array): bounding box center

    Returns:
    - center (np.array): rescaled axis origin
    - axis_d (np.array): rescaled axis direction
    '''
    if jtype == 0 or jtype == 1:
        return [0., 0., 0.], [0., 0., 0.]
    if jtype == 3 or jtype == 4:
        center = box_center
    else:
        center = axis_o + np.dot(axis_d, box_center-axis_o) * axis_d
    return center.tolist(), axis_d.tolist()

def get_axis_mesh(k, axis_o, bbox_center, joint_type):
    '''
    Function to get the axis mesh

    Args:
    - k (np.array): axis direction
    - center (np.array): axis origin
    - bbox_center (np.array): bounding box center
    - joint_type (int): joint type
    '''
    if joint_type == 0 or joint_type == 1 or np.linalg.norm(k) == 0. :
        return None
    
    k = k / np.linalg.norm(k)

    if joint_type == 3 or joint_type == 4: # prismatic or screw
        axis_o = bbox_center
    else: # revolute or continuous
        axis_o = axis_o + np.dot(k, bbox_center-axis_o) * k
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.015, cone_radius=0.03, cylinder_height=1.0, cone_height=0.08)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow, k) 
    rad = np.arccos(np.dot(arrow, k))
    R_arrow = get_rotation_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(axis_o[:3])
    axis.compute_vertex_normals()
    vertices = np.asarray(axis.vertices)
    faces = np.asarray(axis.triangles)
    trimesh_axis = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh_axis.visual.vertex_colors = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
    return trimesh_axis

def get_camera_pose(eye, look_at, up):
    """
    Compute the 4x4 transformation matrix for a camera pose.
    
    Parameters:
        eye (np.ndarray): 3D position of the camera.
        look_at (np.ndarray): 3D point the camera is looking at.
        up (np.ndarray): Up vector.
        
    Returns:
        pose (np.ndarray): 4x4 transformation matrix representing the camera pose.
    """
    # Compute the forward, right, and new up vectors
    forward = (look_at - eye)
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    
    # Create rotation matrix
    pose = np.eye(4)
    pose[0:3, 0] = right
    pose[0:3, 1] = new_up
    pose[0:3, 2] = -forward  # Negative because the camera looks along the negative Z axis in its local coordinate
    pose[0:3, 3] = eye
    
    return pose

def get_bbox_mesh_pair(center, size, radius=0.01, jtype=None, jrange=None, axis_d=None, axis_o=None):
    '''
    Function to get the bounding box mesh pair

    Args:
    - center (np.array): bounding box center
    - size (np.array): bounding box size
    - radius (float): radius of the cylinder
    - jtype (int): joint type
    - jrange (list): joint range
    - axis_d (np.array): axis direction
    - axis_o (np.array): axis origin

    Returns:
    - trimesh_box (trimesh object): trimesh object for the bbox at resting state
    - trimesh_box_anim (trimesh object): trimesh object for the bbox at opening state
    '''

    size = np.clip(size, a_max=3, a_min=0.005)
    center = np.clip(center, a_max=3, a_min=-3)

    line_box = o3d.geometry.TriangleMesh()
    z_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size[2])
    y_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size[1])
    R_y = get_rotation_axis_angle(np.array([1., 0., 0.], dtype=np.float32), np.pi / 2)
    y_cylinder.rotate(R_y, center=(0, 0, 0))
    x_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size[0])
    R_x = get_rotation_axis_angle(np.array([0., 1., 0.], dtype=np.float32), np.pi / 2)
    x_cylinder.rotate(R_x, center=(0, 0, 0))
    
    
    z1 = deepcopy(z_cylinder)
    z1.translate(np.array([-size[0] / 2, size[1] / 2, 0.], dtype=np.float32))
    line_box += z1.translate(center[:3])
    z2 = deepcopy(z_cylinder)
    z2.translate(np.array([size[0] / 2, size[1] / 2, 0.], dtype=np.float32))
    line_box += z2.translate(center[:3])
    z3 = deepcopy(z_cylinder)
    z3.translate(np.array([-size[0] / 2, -size[1] / 2, 0.], dtype=np.float32))
    line_box += z3.translate(center[:3])
    z4 = deepcopy(z_cylinder)
    z4.translate(np.array([size[0] / 2, -size[1] / 2, 0.], dtype=np.float32))
    line_box += z4.translate(center[:3])
    
    y1 = deepcopy(y_cylinder)
    y1.translate(np.array([-size[0] / 2, 0., size[2] / 2], dtype=np.float32))
    line_box += y1.translate(center[:3])
    y2 = deepcopy(y_cylinder)
    y2.translate(np.array([size[0] / 2, 0., size[2] / 2], dtype=np.float32))
    line_box += y2.translate(center[:3])
    y3 = deepcopy(y_cylinder)
    y3.translate(np.array([-size[0] / 2, 0., -size[2] / 2], dtype=np.float32))
    line_box += y3.translate(center[:3])
    y4 = deepcopy(y_cylinder)
    y4.translate(np.array([size[0] / 2, 0., -size[2] / 2], dtype=np.float32))
    line_box += y4.translate(center[:3])
    
    x1 = deepcopy(x_cylinder)
    x1.translate(np.array([0., -size[1] / 2, size[2] / 2], dtype=np.float32))
    line_box += x1.translate(center[:3])
    x2 = deepcopy(x_cylinder)
    x2.translate(np.array([0., size[1] / 2, size[2] / 2], dtype=np.float32))
    line_box += x2.translate(center[:3])
    x3 = deepcopy(x_cylinder)
    x3.translate(np.array([0., -size[1] / 2, -size[2] / 2], dtype=np.float32))
    line_box += x3.translate(center[:3])
    x4 = deepcopy(x_cylinder)
    x4.translate(np.array([0., size[1] / 2, -size[2] / 2]))
    line_box += x4.translate(center[:3])

    # transform
    line_box_anim = deepcopy(line_box)
    if jtype == 2: # revolute
        theta = np.deg2rad(jrange[1])
        line_box_anim.translate(-axis_o)
        R = get_rotation_axis_angle(axis_d, theta)
        line_box_anim.rotate(R, center=(0, 0, 0))
        line_box_anim.translate(axis_o)
    elif jtype == 3: # prismatic
        dist = np.array(jrange[1], dtype=np.float32)
        line_box_anim.translate(axis_d * dist)
    elif jtype == 4: # screw
        dist = np.array(jrange[1], dtype=np.float32)
        theta = 0.25 * np.pi
        R = get_rotation_axis_angle(axis_d, theta)
        line_box_anim.translate(-axis_o)
        line_box_anim.rotate(R, center=(0, 0, 0))
        line_box_anim.translate(axis_o)
        line_box_anim.translate(axis_d * dist)
    elif jtype == 5: # continuous
        theta = 0.25 * np.pi
        R = get_rotation_axis_angle(axis_d, theta)
        line_box_anim.translate(-axis_o)
        line_box_anim.rotate(R, center=(0, 0, 0))
        line_box_anim.translate(axis_o)
    
    vertices = np.asarray(line_box.vertices)
    faces = np.asarray(line_box.triangles)
    trimesh_box = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh_box.visual.vertex_colors = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    vertices_anim = np.asarray(line_box_anim.vertices)
    faces_anim = np.asarray(line_box_anim.triangles)
    trimesh_box_anim = trimesh.Trimesh(vertices=vertices_anim, faces=faces_anim)
    trimesh_box_anim.visual.vertex_colors = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    return trimesh_box, trimesh_box_anim


def get_color_from_palette(palette, idx):
    '''
    Function to get the color from the palette

    Args:
    - palette (list): list of color reference
    - idx (int): index of the color

    Returns:
    - color (np.array): color in the index of idx
    '''
    ref = palette[idx % len(palette)]
    color = np.array([[int(i) for i in ref[4:-1].split(',')]], dtype=np.float32) / 255.
    return color



def render_anim_parts(aabbs, axiss, resolution=256):
    '''
    Function to render the 3D bounding boxes and axes in the scene

    Args:
        aabbs: list of trimesh objects for the bounding box of each part
        axiss: list of trimesh objects for the axis of each part
        resolution: resolution of the rendered image

    Returns:
        color_img: rendered image
    '''
    n_parts = len(aabbs)
    # build mesh for each 3D bounding box
    scene = pyrender.Scene()
    for i in range(n_parts):
        scene.add(aabbs[i])
        if axiss[i] is not None:
            scene.add(axiss[i])

    # Add light to the scene
    scene.ambient_light = np.full(shape=3, fill_value=1.0, dtype=np.float32)
    
    # Add camera to the scene
    pose = get_camera_pose(eye=np.array([1.5, 1.2, 4.5]), look_at=np.array([0, 0, 0]), up=np.array([0, 1, 0]))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0, aspectRatio=1.0)
    scene.add(camera, pose=pose)

    # Offscreen Rendering
    offscreen_renderer = pyrender.OffscreenRenderer(resolution, resolution)

    # Render the scene
    color_img, _ = offscreen_renderer.render(scene)

    # Cleanup
    offscreen_renderer.delete()
    scene.clear()
    return color_img


def draw_boxes_axiss_anim(aabbs_0, aabbs_1, axiss, mode='graph', resolution=256, types=None):
    '''
    Function to draw the 3D bounding boxes and axes of the two frames

    Args:
        aabbs_0: list of trimesh objects for the bounding box of each part in the resting state
        aabbs_1: list of trimesh objects for the bounding box of each part in the open state
        axiss: list of trimesh objects for the axis of each part
        mode: 
            'graph'     using palette corresponding to graph node, 
            'jtype'     using palette corresponding to joint type, 
            'semantic'  using palette corresponding to semantic label
        resolution: resolution of the rendered image
        types: ids corresponding to each joint type or semantic label, if mode is 'jtype' or 'semantic'
    '''
    n_parts = len(aabbs_0)
    ren_aabbs_0 = []
    ren_aabbs_1 = []
    ren_axiss = []
    if mode == 'graph':
        palette = graph_color_ref
        # Add meshes to the scene
        for i in range(n_parts):
            color = get_color_from_palette(palette, i)
            aabb_0 = pyrender.Mesh.from_trimesh(aabbs_0[i], smooth=False)
            aabb_0.primitives[0].color_0 = color.repeat(aabb_0.primitives[0].positions.shape[0], axis=0)
            ren_aabbs_0.append(aabb_0)
            aabb_1 = pyrender.Mesh.from_trimesh(aabbs_1[i], smooth=False)
            aabb_1.primitives[0].color_0 = color.repeat(aabb_1.primitives[0].positions.shape[0], axis=0)
            ren_aabbs_1.append(aabb_1)
            if axiss[i] is not None:
                axis = pyrender.Mesh.from_trimesh(axiss[i], smooth=False)
                axis.primitives[0].color_0 = color.repeat(axis.primitives[0].positions.shape[0], axis=0)
                ren_axiss.append(axis)
            else:
                ren_axiss.append(None)     
    elif mode == 'jtype' or mode == 'semantic':
        assert types is not None
        palette = joint_color_ref if mode == 'jtype' else semantic_color_ref
        # Add meshes to the scene
        for i in range(n_parts):
            color = get_color_from_palette(palette, types[i])
            aabb_0 = pyrender.Mesh.from_trimesh(aabbs_0[i], smooth=False)
            aabb_0.primitives[0].color_0 = color.repeat(aabb_0.primitives[0].positions.shape[0], axis=0)
            ren_aabbs_0.append(aabb_0)
            aabb_1 = pyrender.Mesh.from_trimesh(aabbs_1[i], smooth=False)
            aabb_1.primitives[0].color_0 = color.repeat(aabb_1.primitives[0].positions.shape[0], axis=0)
            ren_aabbs_1.append(aabb_1)

            if axiss[i] is not None:
                axis = pyrender.Mesh.from_trimesh(axiss[i], smooth=False)
                ren_axiss.append(axis)
            else:
                ren_axiss.append(None)
    else:
        raise ValueError('mode must be either graph or type')

    img0 = render_anim_parts(ren_aabbs_0, ren_axiss, resolution=resolution)
    img1 = render_anim_parts(ren_aabbs_1, ren_axiss, resolution=resolution)
    return np.concatenate([img0, img1], axis=1)

def prepare_meshes(info_dict):
    """
    Function to prepare the bbox and axis meshes for visualization

    Args:
    - info_dict (dict): output json containing the graph information
    """
    from utils.refs import joint_ref, sem_ref
    tree = info_dict["diffuse_tree"]
    bbox_0, bbox_1, axiss, labels, jtypes = [], [], [], [], []
    root_id = 0
    # get root id
    for node in tree:
        if node["parent"] == -1:
            root_id = node["id"]
    for node in tree:
        # retrieve info
        box_cen = np.array(node["aabb"]["center"], dtype=np.float32)
        box_size = np.array(node["aabb"]["size"], dtype=np.float32)
        axis_d = np.array(node["joint"]["axis"]["direction"], dtype=np.float32)
        axis_o = np.array(node["joint"]["axis"]["origin"], dtype=np.float32)
        jtype = joint_ref["fwd"][node["joint"]["type"]]
        # construct meshes for bbox in two states (closed and fully open)
        if node["id"] == root_id or node["parent"] == root_id:  # use the joint info directly
            bb_0, bb_1 = get_bbox_mesh_pair(
                box_cen,
                box_size,
                jtype=jtype,
                jrange= node["joint"]["range"],
                axis_d=axis_d,
                axis_o=axis_o,
            )
        else: # use the parent joint info
            parent_id = node["parent"]
            bb_0, bb_1 = get_bbox_mesh_pair(
                box_cen,
                box_size,
                jtype=joint_ref["fwd"][tree[parent_id]["joint"]["type"]],
                jrange=tree[parent_id]["joint"]["range"],
                axis_d=np.array(tree[parent_id]["joint"]["axis"]["direction"], dtype=np.float32),
                axis_o=np.array(tree[parent_id]["joint"]["axis"]["origin"], dtype=np.float32),
            )
        # construct mesh for joint axis
        axis_mesh = get_axis_mesh(axis_d, axis_o, box_cen, node["joint"]["type"])
        # append
        bbox_0.append(bb_0)
        bbox_1.append(bb_1)
        axiss.append(axis_mesh)
        labels.append(sem_ref["fwd"][node["name"]])
        jtypes.append(jtype)

    return {
        "bbox_0": bbox_0,
        "bbox_1": bbox_1,
        "axiss": axiss,
        "labels": labels,
        "jtypes": jtypes,
    }
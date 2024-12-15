"""
This script computes the Chamfer Distance (CD) between two objects\n
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import trimesh
from copy import deepcopy
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from objects.motions import transform_all_parts
from objects.dict_utils import (
    zero_center_object,
    rescale_object,
    compute_overall_bbox_size,
    get_base_part_idx,
    find_part_mapping
)

def _load_and_combine_plys(dir, ply_files, scale=None, z_rotate=None, translate=None):
    """
    Load and combine the ply files into one PyTorch3D mesh

    - dir: the directory of the object in which the ply files are from\n
    - ply_files: the list of ply files\n
    - scale: the scale factor to apply to the vertices\n
    - z_rotate: whether to rotate the object around the z-axis by 90 degrees\n
    - translate: the translation to apply to the vertices\n

    Return:\n
    - mesh: one PyTorch3D mesh of the combined ply files
    """

    # Combine the ply files into one
    meshes = []
    for ply_file in ply_files:
        meshes.append(trimesh.load(os.path.join(dir, ply_file), force="mesh"))
    full_part_mesh = trimesh.util.concatenate(meshes)

    # Apply the transformations
    full_part_mesh.vertices -= full_part_mesh.bounding_box.centroid
    transformation = trimesh.transformations.compose_matrix(
        scale=scale,
        angles=[0, 0, np.radians(90) if z_rotate else 0],
        translate=translate,
    )
    full_part_mesh.apply_transform(transformation)

    # Create the PyTorch3D mesh
    mesh = Meshes(
        verts=torch.as_tensor(full_part_mesh.vertices, dtype=torch.float32, device='cuda').unsqueeze(
            0
        ),
        faces=torch.as_tensor(full_part_mesh.faces, dtype=torch.int32, device='cuda').unsqueeze(0),
    )

    return mesh


def _compute_chamfer_distance(
    obj1_part_points, obj2_part_points, part_mapping=None, exclude_id=-1
):
    """
    Compute the chamfer distance between the two set of points representing the two objects

    - obj1_part_points: the set of points representing the first object\n
    - obj2_part_points: the set of points representing the second object\n
    - part_mapping (optional): the part mapping from the first object to the second object, if provided, the chamfer distance will be computed between the corresponding parts\n
    - exclude_id (optional): the part id to exclude from the chamfer distance computation, the default if provided is the base part id\n

    Return:\n
    - distance: the chamfer distance between the two objects
    """
    if part_mapping is not None:
        n_parts = part_mapping.shape[0]
        distance = 0
        for i in range(n_parts):
            if i == exclude_id:
                continue
            obj1_part_points_i = obj1_part_points[i]
            obj2_part_points_i = obj2_part_points[int(part_mapping[i, 0])]
            with torch.no_grad():
                obj1_part_points_i = obj1_part_points_i.cuda()
                obj2_part_points_i = obj2_part_points_i.cuda()
                # symmetric chamfer distance
                forward_distance, _ = chamfer_distance(
                    obj1_part_points_i[None, :],
                    obj2_part_points_i[None, :],
                    batch_reduction=None,
                )
                backward_distance, _ = chamfer_distance(
                    obj2_part_points_i[None, :],
                    obj1_part_points_i[None, :],
                    batch_reduction=None,
                )
                distance += (forward_distance.item() + backward_distance.item()) * 0.5
        distance /= n_parts
    else:
        # Merge the points of all parts into one tensor
        obj1_part_points = obj1_part_points.reshape(-1, 3)
        obj2_part_points = obj2_part_points.reshape(-1, 3)

        # Compute the chamfer distance between the two objects
        with torch.no_grad():
            obj1_part_points = obj1_part_points.cuda()
            obj2_part_points = obj2_part_points.cuda()
            forward_distance, _ = chamfer_distance(
                obj1_part_points[None, :],
                obj2_part_points[None, :],
                batch_reduction=None,
            )
            backward_distance, _ = chamfer_distance(
                obj2_part_points[None, :],
                obj1_part_points[None, :],
                batch_reduction=None,
            )
            distance = (forward_distance.item() + backward_distance.item()) * 0.5

    return distance


def _get_scores(
    src_dict,
    tgt_dict,
    original_src_part_points,
    original_tgt_part_points,
    part_mapping,
    num_states,
    include_base,
    src_base_idx,
):

    chamfer_distances = np.zeros(num_states, dtype=np.float32)
    joint_states = np.linspace(0, 1, num_states)
    for state_idx, state in enumerate(joint_states):

        # Reset the part point clouds
        src_part_points = deepcopy(original_src_part_points)
        tgt_part_points = deepcopy(original_tgt_part_points)

        # Transform the part point clouds to the current state using the joints
        transform_all_parts(src_part_points.numpy(), src_dict, state, dry_run=False)
        transform_all_parts(tgt_part_points.numpy(), tgt_dict, state, dry_run=False)

        # Compute the chamfer distance between the two objects
        chamfer_distances[state_idx] = _compute_chamfer_distance(
            src_part_points,
            tgt_part_points,
            part_mapping=part_mapping,
            exclude_id=-1 if include_base else src_base_idx,
        )

    # Compute the ID
    aid_cd = np.mean(chamfer_distances)
    rid_cd = chamfer_distances[0]

    return {
        "AS-CD": float(aid_cd),
        "RS-CD": float(rid_cd),
    }


def CD(
    gen_obj_dict,
    gen_obj_path,
    gt_obj_dict,
    gt_obj_path,
    num_states=5,
    num_samples=2048,
    include_base=False,
):
    """
    Compute the Chamfer Distance\n
    This metric is the average of per-part chamfer distance between the two objects over a number of articulation states\n

    - gen_obj_dict: the generated object dictionary\n
    - gen_obj_path: the directory to the predicted object\n
    - gt_obj_dict: the ground truth object dictionary\n
    - gt_obj_path: the directory to the ground truth object\n
    - num_states (optional): the number of articulation states to compute the metric\n
    - num_samples (optional): the number of samples to use\n
    - include_base (optional): whether to include the base part in the chamfer distance computation\n

    Return:\n
    - aid_score: the score over the sampled articulated states\n
    - rid_score: the score at the resting state\n
        - The score is in the range of [0, inf), lower is better
    """
    # Make copies of the dictionaries to avoid modifying the original dictionaries
    gen_dict = deepcopy(gen_obj_dict)
    gt_dict = deepcopy(gt_obj_dict)

    # Zero center the objects
    zero_center_object(gen_dict)
    zero_center_object(gt_dict)

    # Compute the scale factor by comparing the overall bbox size and scale the candidate object as a whole
    gen_bbox_size = compute_overall_bbox_size(gen_dict)
    gt_bbox_size = compute_overall_bbox_size(gt_dict)
    scale_factor = gen_bbox_size / gt_bbox_size
    rescale_object(gen_obj_dict, scale_factor)

    # Record the indices of the base parts of the two objects
    gen_base_idx = get_base_part_idx(gen_dict)
    gt_base_idx = get_base_part_idx(gt_dict)

    # Find mapping between the parts of the two objects based on closest bbox centers
    mapping_gen2gt = find_part_mapping(gen_dict, gt_dict, use_hungarian=True)
    mapping_gt2gen = find_part_mapping(gt_dict, gen_dict, use_hungarian=True)

    # Get the number of parts of the two objects
    gen_tree = gen_dict["diffuse_tree"]
    gt_tree = gt_dict["diffuse_tree"]
    gen_num_parts = len(gen_tree)
    gt_num_parts = len(gt_tree)

    # Get the paths of the ply files of the two objects
    gen_part_ply_paths = [
        {"dir": gen_obj_path, "files": gen_tree[i]["plys"]}
        for i in range(gen_num_parts)
    ]
    gt_part_ply_paths = [
        {"dir": gt_obj_path, "files": gt_tree[i]["plys"]}
        for i in range(gt_num_parts)
    ]

    # Load the ply files of the two objects and sample points from them
    gen_part_points = torch.zeros(
        (gen_num_parts, num_samples, 3), dtype=torch.float32
    )
    for i in range(gen_num_parts):
        part_mesh = _load_and_combine_plys(
            gen_part_ply_paths[i]["dir"],
            gen_part_ply_paths[i]["files"],
            scale=scale_factor,
            translate=gen_tree[i]["aabb"]["center"],
        )
        gen_part_points[i] = sample_points_from_meshes(
            part_mesh, num_samples=num_samples
        ).squeeze(0).cpu()

    gt_part_points = torch.zeros(
        (gt_num_parts, num_samples, 3), dtype=torch.float32
    )
    for i in range(gt_num_parts):
        part_mesh = _load_and_combine_plys(
            gt_part_ply_paths[i]["dir"],
            gt_part_ply_paths[i]["files"],
            translate=gt_tree[i]["aabb"]["center"],
        )
        gt_part_points[i] = sample_points_from_meshes(
            part_mesh, num_samples=num_samples
        ).squeeze(0).cpu()

    cd_gen2gt = _get_scores(
        gen_dict,
        gt_dict,
        gen_part_points,
        gt_part_points,
        mapping_gen2gt,
        num_states,
        include_base,
        gen_base_idx,
    )

    cd_gt2gen = _get_scores(
        gt_dict,
        gen_dict,
        gt_part_points,
        gen_part_points,
        mapping_gt2gen,
        num_states,
        include_base,
        gt_base_idx,
    )

    return {
        "AS-CD": (cd_gen2gt["AS-CD"] + cd_gt2gen["AS-CD"]) / 2,
        "RS-CD": (cd_gen2gt["RS-CD"] + cd_gt2gen["RS-CD"]) / 2,
    }

"""
This file computes the IoU-based and centroid-distance-based metrics in a symmetric manner\n
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from copy import deepcopy
from objects.dict_utils import (
    get_base_part_idx,
    get_bbox_vertices,
    remove_handles,
    compute_overall_bbox_size,
    rescale_object,
    find_part_mapping,
    zero_center_object,
)
from objects.motions import transform_all_parts
from metrics.giou import sampling_giou, sampling_cDist


def _get_scores(
    src_dict,
    tgt_dict,
    original_src_bbox_vertices,
    original_tgt_bbox_vertices,
    mapping,
    num_states,
    rotation_fix_range,
    num_samples,
    iou_include_base,
):
    # Record the indices of the base parts of the src objects
    src_base_idx = get_base_part_idx(src_dict)

    # Compute the sum of IoU between the generated object and the candidate object over a number of articulation states
    num_parts_in_src = len(src_dict["diffuse_tree"])
    iou_per_part_and_state = np.zeros((num_parts_in_src, num_states), dtype=np.float32)
    cDist_per_part_and_state = np.zeros(
        (num_parts_in_src, num_states), dtype=np.float32
    )
    
    states = np.linspace(0, 1, num_states)
    for state_idx, state in enumerate(states):

        # Get a fresh copy of the bounding box vertices in rest pose
        src_bbox_vertices = deepcopy(original_src_bbox_vertices)
        tgt_bbox_vertices = deepcopy(original_tgt_bbox_vertices)

        # Transform the objects to the current state using the joints
        src_part_transfomrations = transform_all_parts(
            src_bbox_vertices,
            src_dict,
            state,
            rotation_fix_range=rotation_fix_range,
        )

        tgt_part_transfomrations = transform_all_parts(
            tgt_bbox_vertices,
            tgt_dict,
            state,
            rotation_fix_range=rotation_fix_range,
        )
        
        # Compute the IoU between the two objects using the transformed bounding boxes and the part mapping
        for src_part_idx in range(num_parts_in_src):

            # Get the index of the corresponding part in the candidate object
            tgt_part_idx = int(mapping[src_part_idx, 0])

            # Always use a fresh copy of the bounding box vertices in rest pose in case dry_run=False is incorrectly set
            src_part_bbox_vertices = deepcopy(original_src_bbox_vertices)[src_part_idx]
            tgt_part_bbox_vertices = deepcopy(original_tgt_bbox_vertices)[tgt_part_idx]

            # Compute the sampling-based IoU between the two parts

            iou_per_part_and_state[src_part_idx, state_idx] = sampling_giou(
                src_part_bbox_vertices,
                tgt_part_bbox_vertices,
                src_part_transfomrations[src_part_idx],
                tgt_part_transfomrations[tgt_part_idx],
                num_samples=num_samples,
            )
            # Compute the centriod distance between the two matched parts
            cDist_per_part_and_state[src_part_idx, state_idx] = sampling_cDist(
                src_dict["diffuse_tree"][src_part_idx],
                tgt_dict["diffuse_tree"][tgt_part_idx],
                src_part_transfomrations[src_part_idx],
                tgt_part_transfomrations[tgt_part_idx],
            )

    # IoU and cDist at the resting state
    per_part_iou_avg_at_rest = iou_per_part_and_state[:, 0]
    per_part_cDist_avg_at_rest = cDist_per_part_and_state[:, 0]

    # Average the IoU over the states
    per_part_iou_avg_over_states = np.sum(iou_per_part_and_state, axis=1) / num_states
    # Average the cDist over the states
    per_part_cDist_avg_over_states = (
        np.sum(cDist_per_part_and_state, axis=1) / num_states
    )

    # Remove the base part if specified
    if not iou_include_base:
        per_part_iou_avg_over_states = np.delete(
            per_part_iou_avg_over_states, src_base_idx
        )
        per_part_iou_avg_at_rest = np.delete(per_part_iou_avg_at_rest, src_base_idx)
        per_part_cDist_avg_over_states = np.delete(
            per_part_cDist_avg_over_states, src_base_idx
        )
        per_part_cDist_avg_at_rest = np.delete(per_part_cDist_avg_at_rest, src_base_idx)

    aid_iou = float(np.mean(per_part_iou_avg_over_states)) if len(per_part_iou_avg_over_states) > 0 else 0
    aid_cdist = float(np.mean(per_part_cDist_avg_over_states)) if len(per_part_cDist_avg_over_states) > 0 else 1
    rid_iou = float(np.mean(per_part_iou_avg_at_rest)) if len(per_part_iou_avg_at_rest) > 0 else 0
    rid_cdist = float(np.mean(per_part_cDist_avg_at_rest)) if len(per_part_cDist_avg_at_rest) > 0 else 1

    return {
        "AS-IoU": 1. - aid_iou,
        "AS-cDist": aid_cdist,
        "RS-IoU": 1. - rid_iou,
        "RS-cDist": rid_cdist
    }


def IoU_cDist(
    gen_obj_dict,
    gt_obj_dict,
    num_states=5,
    compare_handles=False,
    iou_include_base=False,
    rotation_fix_range=True,
    num_samples=10000,
):
    """
    Compute the IoU-based and centroid-distance-based metrics\n
    This metric is the average sum of IoU between parts in the two objects over the sampled articulation states and at the resting state\n

    - gen_obj_dict: the dictionary of the generated object\n
    - gt_obj_dict: the dictionary of the gt object\n
    - num_states: the number of articulation states to compute the metric\n
    - compare_handles (optional): whether to compare the handles\n
    - iou_include_base (optional): whether to include the base part in the IoU computation\n
    - rotation_fix_range (optional): whether to fix the rotation range to 90 degrees for revolute joints\n
    - num_samples (optional): the number of samples to use\n

    Return:\n
    - scores: a dictionary of the computed scores\n
        - "AS-IoU": the average IoU over the articulation states\n
        - "AS-cDist": the average centroid distance over the articulation states\n
        - "RS-IoU": the average IoU at the resting state\n
        - "RS-cDist": the average centroid distance at the resting state\n
    """
    # Make copies of the dictionaries to avoid modifying the original dictionaries
    gen_dict = deepcopy(gen_obj_dict)
    gt_dict = deepcopy(gt_obj_dict)

    # Strip the handles from the object if not comparing them
    if not compare_handles:
        gen_dict = remove_handles(gen_dict)
        gt_dict = remove_handles(gt_dict)

    # Zero center the objects
    zero_center_object(gen_dict)
    zero_center_object(gt_dict)

    # scale the generated object as a whole to match the size of the gt object
    gen_bbox_size = compute_overall_bbox_size(gen_dict)
    gt_bbox_size = compute_overall_bbox_size(gt_dict)
    scale_factor = gt_bbox_size / gen_bbox_size
    rescale_object(gen_dict, scale_factor)

    mapping_gen2gt = find_part_mapping(gen_dict, gt_dict, use_hungarian=True)
    mapping_gt2gen = find_part_mapping(gt_dict, gen_dict, use_hungarian=True)

    # Save the original bounding box vertices in rest pose
    original_gen_bbox_vertices = np.array(
        [get_bbox_vertices(gen_dict, i) for i in range(len(gen_dict["diffuse_tree"]))],
        dtype=np.float32,
    )
    original_gt_bbox_vertices = np.array(
        [get_bbox_vertices(gt_dict, i) for i in range(len(gt_dict["diffuse_tree"]))],
        dtype=np.float32,
    )

    scores_gen2gt = _get_scores(
        gen_dict,
        gt_dict,
        original_gen_bbox_vertices,
        original_gt_bbox_vertices,
        mapping_gen2gt,
        num_states,
        rotation_fix_range,
        num_samples,
        iou_include_base,
    )

    scores_gt2gen = _get_scores(
        gt_dict,
        gen_dict,
        original_gt_bbox_vertices,
        original_gen_bbox_vertices,
        mapping_gt2gen,
        num_states,
        rotation_fix_range,
        num_samples,
        iou_include_base,
    )


    scores = {
        "AS-IoU": (scores_gen2gt["AS-IoU"] + scores_gt2gen["AS-IoU"]) / 2,
        "AS-cDist": (scores_gen2gt["AS-cDist"] + scores_gt2gen["AS-cDist"]) / 2,
        "RS-IoU": (scores_gen2gt["RS-IoU"] + scores_gt2gen["RS-IoU"]) / 2,
        "RS-cDist": (scores_gen2gt["RS-cDist"] + scores_gt2gen["RS-cDist"]) / 2,
    }

    return scores

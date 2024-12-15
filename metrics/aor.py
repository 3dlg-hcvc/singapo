import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from copy import deepcopy
from metrics.iou import sampling_iou
from objects.motions import transform_all_parts

from objects.dict_utils import get_bbox_vertices

'''
This file computes the Average Overlap Ratio (AOR) metric\n
'''

def AOR(tgt, num_states=10, transform_use_plucker=False):
    tree = tgt["diffuse_tree"]
    states = np.linspace(0, 1, num_states)
    original_bbox_vertices = np.array([get_bbox_vertices(tgt, i) for i in range(len(tgt["diffuse_tree"]))], dtype=np.float32) 
    
    ious = []
    for state_idx, state in enumerate(states):
        ious_per_state = []
        bbox_vertices = deepcopy(original_bbox_vertices)
        part_trans = transform_all_parts(bbox_vertices, tgt, state, transform_use_plucker)
        for node in tree:
            children = node['children']
            num_children = len(children)
            if num_children < 2:
                continue
            for i in range(num_children-1):
                for j in range(i+1, num_children):
                    child_id = children[i]
                    sibling_id = children[j]
                    bbox_v_0 = deepcopy(bbox_vertices[child_id])
                    bbox_v_1 = deepcopy(bbox_vertices[sibling_id])
                    iou = sampling_iou(bbox_v_0, bbox_v_1, part_trans[child_id], part_trans[sibling_id], num_samples=10000)
                    if np.isnan(iou):
                        continue
                    ious_per_state.append(iou)
        if len(ious_per_state) > 0:
            ious.append(np.mean(ious_per_state))
    if len(ious) == 0:
        return -1
    return float(np.mean(ious))


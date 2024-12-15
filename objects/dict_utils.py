import numpy as np
from scipy.optimize import linear_sum_assignment


def get_base_part_idx(obj_dict):
    """
    Get the index of the base part in the object dictionary\n

    - obj_dict: the object dictionary\n

    Return:\n
    - base_part_idx: the index of the base part
    """

    # Adjust for NAP's corner case
    base_part_ids = np.where(
        [part["parent"] == -1 for part in obj_dict["diffuse_tree"]]
    )[0]
    if len(base_part_ids) > 0:
        return base_part_ids[0].item()
    else:
        raise ValueError("No base part found")


def get_bbox_vertices(obj_dict, part_idx):
    """
    Get the 8 vertices of the bounding box\n
    The order of the vertices is the same as the order that pytorch3d.ops.box3d_overlap expects\n
    (This order is not necessary since we are not using pytorch3d.ops.box3d_overlap anymore)\n

    - bbox_center: the center of the bounding box in the form: [cx, cy, cz]\n
    - bbox_size: the size of the bounding box in the form: [lx, ly, lz]\n

    Return:\n
    - bbox_vertices: the 8 vertices of the bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]
    """

    part = obj_dict["diffuse_tree"][part_idx]
    bbox_center = np.array(part["aabb"]["center"], dtype=np.float32)
    bbox_size_half = np.array(part["aabb"]["size"], dtype=np.float32) / 2

    bbox_vertices = np.zeros((8, 3), dtype=np.float32)

    # Get the 8 vertices of the bounding box in the order that pytorch3d.ops.box3d_overlap expects:
    # 0: (x0, y0, z0)    # 1: (x1, y0, z0)    # 2: (x1, y1, z0)    # 3: (x0, y1, z0)
    # 4: (x0, y0, z1)    # 5: (x1, y0, z1)    # 6: (x1, y1, z1)    # 7: (x0, y1, z1)
    bbox_vertices[0, :] = bbox_center - bbox_size_half
    bbox_vertices[1, :] = bbox_center + np.array(
        [bbox_size_half[0], -bbox_size_half[1], -bbox_size_half[2]], dtype=np.float32
    )
    bbox_vertices[2, :] = bbox_center + np.array(
        [bbox_size_half[0], bbox_size_half[1], -bbox_size_half[2]], dtype=np.float32
    )
    bbox_vertices[3, :] = bbox_center + np.array(
        [-bbox_size_half[0], bbox_size_half[1], -bbox_size_half[2]], dtype=np.float32
    )
    bbox_vertices[4, :] = bbox_center + np.array(
        [-bbox_size_half[0], -bbox_size_half[1], bbox_size_half[2]], dtype=np.float32
    )
    bbox_vertices[5, :] = bbox_center + np.array(
        [bbox_size_half[0], -bbox_size_half[1], bbox_size_half[2]], dtype=np.float32
    )
    bbox_vertices[6, :] = bbox_center + bbox_size_half
    bbox_vertices[7, :] = bbox_center + np.array(
        [-bbox_size_half[0], bbox_size_half[1], bbox_size_half[2]], dtype=np.float32
    )

    return bbox_vertices


def compute_overall_bbox_size(obj_dict):
    """
    Compute the overall bounding box size of the object\n

    - obj_dict: the object dictionary\n

    Return:\n
    - bbox_size: the overall bounding box size in the form: [lx, ly, lz]
    """

    bbox_min = np.zeros((len(obj_dict["diffuse_tree"]), 3), dtype=np.float32)
    bbox_max = np.zeros((len(obj_dict["diffuse_tree"]), 3), dtype=np.float32)

    # For each part, compute the bounding box and store the min and max vertices
    for part_idx, part in enumerate(obj_dict["diffuse_tree"]):
        bbox_center = np.array(part["aabb"]["center"], dtype=np.float32)
        bbox_size_half = np.array(part["aabb"]["size"], dtype=np.float32) / 2
        bbox_min[part_idx] = bbox_center - bbox_size_half
        bbox_max[part_idx] = bbox_center + bbox_size_half

    # Compute the overall bounding box size
    bbox_min = np.min(bbox_min, axis=0)
    bbox_max = np.max(bbox_max, axis=0)
    bbox_size = bbox_max - bbox_min
    return bbox_size


def remove_handles(obj_dict):
    """
    Remove the handles from the object dictionary and adjust the id, parent, and children of the parts\n

    - obj_dict: the object dictionary\n

    Return:\n
    - obj_dict: the object dictionary without the handles
    """

    # Find the indices of the handles
    handle_idxs = np.array(
        [
            i
            for i in range(len(obj_dict["diffuse_tree"]))
            if obj_dict["diffuse_tree"][i]["name"] == "handle"
            and obj_dict["diffuse_tree"][i]["parent"] != -1
        ]
    )  # Added to avoid corner case of NAP where the handle is the base part

    # Remove the handles from the object dictionary and adjust the id, parent, and children of the parts
    for handle_idx in handle_idxs:
        handle = obj_dict["diffuse_tree"][handle_idx]
        parent_idx = handle["parent"]
        if handle_idx in obj_dict["diffuse_tree"][parent_idx]["children"]:
            obj_dict["diffuse_tree"][parent_idx]["children"].remove(handle_idx)
        obj_dict["diffuse_tree"].pop(handle_idx)

        # Adjust the id, parent, and children of the parts
        for part in obj_dict["diffuse_tree"]:
            if part["id"] > handle_idx:
                part["id"] -= 1
            if part["parent"] > handle_idx:
                part["parent"] -= 1
            for i in range(len(part["children"])):
                if part["children"][i] > handle_idx:
                    part["children"][i] -= 1

        handle_idxs -= 1

    return obj_dict


# def normalize_object(obj_dict):
#     """
#     Normalize the object as a whole\n
#     Make the base part to be centered at the origin and have a size of 2\n

#     obj_dict: the object dictionary
#     """
#     # Find the base part and compute the translation and scaling factors
#     tree = obj_dict["diffuse_tree"]
#     for part in tree:
#         if part["parent"] == -1:
#             translate = -np.array(part["aabb"]["center"], dtype=np.float32)
#             scale = 2.0 / np.array(part["aabb"]["size"], dtype=np.float32)
#             break

#     for part in tree:
#         part["aabb"]["center"] = (
#             np.array(part["aabb"]["center"], dtype=np.float32) + translate
#         ) * scale
#         part["aabb"]["size"] = np.array(part["aabb"]["size"], dtype=np.float32) * scale
#         if part["joint"]["type"] != "fixed":
#             part["joint"]["axis"]["origin"] = (
#                 np.array(part["joint"]["axis"]["origin"], dtype=np.float32) + translate
#             ) * scale

def zero_center_object(obj_dict):
    """
    Zero center the object as a whole\n

    - obj_dict: the object dictionary
    """

    bbox_min = np.zeros((len(obj_dict["diffuse_tree"]), 3))
    bbox_max = np.zeros((len(obj_dict["diffuse_tree"]), 3))

    # For each part, compute the bounding box and store the min and max vertices
    for part_idx, part in enumerate(obj_dict["diffuse_tree"]):
        bbox_center = np.array(part["aabb"]["center"])
        bbox_size_half = np.array(part["aabb"]["size"]) / 2
        bbox_min[part_idx] = bbox_center - bbox_size_half
        bbox_max[part_idx] = bbox_center + bbox_size_half
    
    # Compute the overall bounding box size
    bbox_min = np.min(bbox_min, axis=0)
    bbox_max = np.max(bbox_max, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2

    translate = -bbox_center

    for part in obj_dict["diffuse_tree"]:
        part["aabb"]["center"] = np.array(part["aabb"]["center"]) + translate
        if part["joint"]["type"] != "fixed":
            part["joint"]["axis"]["origin"] = np.array(part["joint"]["axis"]["origin"]) + translate


def rescale_object(obj_dict, scale_factor):
    """
    Rescale the object as a whole\n

    - obj_dict: the object dictionary\n
    - scale_factor: the scale factor to rescale the object
    """

    for part in obj_dict["diffuse_tree"]:
        part["aabb"]["center"] = (
            np.array(part["aabb"]["center"], dtype=np.float32) * scale_factor
        )
        part["aabb"]["size"] = (
            np.array(part["aabb"]["size"], dtype=np.float32) * scale_factor
        )
        if part["joint"]["type"] != "fixed":
            part["joint"]["axis"]["origin"] = (
                np.array(part["joint"]["axis"]["origin"], dtype=np.float32)
                * scale_factor
            )


def find_part_mapping(obj1_dict, obj2_dict, use_hungarian=False):
    """
    Find the correspondences from the first object to the second object based on closest bbox centers\n

    - obj1_dict: the first object dictionary\n
    - obj2_dict: the second object dictionary\n

    Return:\n
    - mapping: the mapping from the first object to the second object in the form: [[obj_part_idx, distance], ...]
    """
    if use_hungarian:
        return hungarian_matching(obj1_dict, obj2_dict)

    # Initialize the distances to be +inf
    mapping = np.ones((len(obj1_dict["diffuse_tree"]), 2)) * np.inf

    # For each part in the first object, find the closest part in the second object based on the bounding box center
    for req_part_idx, req_part in enumerate(obj1_dict["diffuse_tree"]):
        for obj_part_idx, obj_part in enumerate(obj2_dict["diffuse_tree"]):
            distance = np.linalg.norm(
                np.array(req_part["aabb"]["center"])
                - np.array(obj_part["aabb"]["center"])
            )
            if distance < mapping[req_part_idx, 1]:
                mapping[req_part_idx, :] = [obj_part_idx, distance]

    return mapping


def hungarian_matching(obj1_dict, obj2_dict):
    """
    Find the correspondences from the first object to the second object based on closest bbox centers using Hungarian algorithm\n

    - obj1_dict: the first object dictionary\n
    - obj2_dict: the second object dictionary\n

    Return:\n
    - mapping: the mapping from the first object to the second object in the form: [[obj_part_idx], ...]
    """
    INF = 9999999

    tree1 = obj1_dict["diffuse_tree"]
    tree2 = obj2_dict["diffuse_tree"]

    n_parts1 = len(tree1)
    n_parts2 = len(tree2)
    n_parts_max = max(n_parts1, n_parts2)

    # Initialize the cost matrix
    cost_matrix = np.ones((n_parts_max, n_parts_max), dtype=np.float32) * INF
    for i in range(n_parts1):
        for j in range(n_parts2):
            cost_matrix[i, j] = np.linalg.norm(
                np.array(tree1[i]["aabb"]["center"], dtype=np.float32)
                - np.array(tree2[j]["aabb"]["center"], dtype=np.float32)
            )

    # Find the correspondences using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Valid correspondences are those with all cost less than INF
    valid_correspondences = np.where(cost_matrix[row_ind, col_ind] < INF)[0]
    invalid_correspondences = np.where(np.logical_not(cost_matrix[row_ind, col_ind] < INF))[0]

    row_i = row_ind[valid_correspondences]
    col_i = col_ind[valid_correspondences]

    # Construct the mapping
    mapping = np.zeros(
        (n_parts1, 2), dtype=np.float32
    ) 
    mapping[row_i, 0] = col_i
    mapping[row_i, 1] = cost_matrix[row_i, col_i]

    # assign the index of the most closely matched part
    if n_parts1 > n_parts2: 
        row_j = row_ind[invalid_correspondences]
        col_j = cost_matrix[row_j, :].argmin(axis=1)
        mapping[row_j, 0] = col_j
        mapping[row_j, 1] = cost_matrix[row_j, col_j]

    return mapping

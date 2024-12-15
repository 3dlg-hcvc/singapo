import os
import sys
import random
import json
import numpy as np
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.iou_cdist import IoU_cDist
import networkx as nx

all_categories = [
    "Table",
    "StorageFurniture",
    "WashingMachine",
    "Microwave",
    "Dishwasher",
    "Refrigerator",
    "Oven",
]


def get_hash(file, key="diffuse_tree", ignore_handles=True, dag=False):
    tree = file[key]
    if dag:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for node in tree:
        if ignore_handles and "handle" in node["name"].lower():
            continue
        G.add_node(node["id"])
        if node["parent"] != -1:
            G.add_edge(node["id"], node["parent"])
    hashcode = nx.weisfeiler_lehman_graph_hash(G)
    return hashcode


def _verify_mesh_exists(dir, ply_files, verbose=False):
    """
    Verify that the mesh files exist\n

    - dir: the directory of the object\n
    - ply_files: the list of mesh files\n
    - verbose (optional): whether to print the progress\n

    return:\n
    - True if the mesh files exist, False otherwise
    """

    for ply_file in ply_files:
        if not os.path.exists(os.path.join(dir, ply_file)):
            if verbose:
                print(f" - {os.path.join(dir, ply_file)} does not exist!!!")
            return False
    return True


def _generate_output_part_dicts(
    candidate_dict,
    part_idx,
    candidate_dir,
    requirement_part_bbox_sizes,
    bbox_size_eps=1e-3,
    verbose=False,
):
    """
    Generate the output part dictionary for all parts that are fulfilled by the candidate part and computing the scale factor of the parts\n

    - candidate_dict: the candidate object dictionary\n
    - part_idx: the index of the part in the candidate object\n
    - candidate_dir: the directory of the candidate object\n
    - requirement_part_bbox_sizes: the bounding box sizes of the requirement part in the form: [[lx1, ly1, lz1], [lx2, ly2, lz2], ...]\n
    - bbox_size_eps (optional): the epsilon to avoid zero volume parts\n
    - verbose (optional): whether to print the progress\n

    Return:\n
    - part_dicts: the output part dictionaries in the form:
        - [{name, dir, files, scale_factor=[sx, sy, sz]}, z_rotate_90]
            - z_rotate_90 is True if the part needs to be rotated by 90 degrees around the z-axis
    - [{}, ...] if any of the mesh files do not exist
    """

    part_dicts = [{} for _ in range(len(requirement_part_bbox_sizes))]
    fixed_portion = {
        "name": candidate_dict["diffuse_tree"][part_idx]["name"],
        "dir": candidate_dir,
        "files": candidate_dict["diffuse_tree"][part_idx]["plys"],
        "z_rotate_90": False,
    }

    # Verify that the mesh files exist
    if not _verify_mesh_exists(fixed_portion["dir"], fixed_portion["files"], verbose):
        if verbose:
            print(
                f" - ! Found invalid mesh files in {fixed_portion['dir']}, skipping..."
            )
        return part_dicts  # List of empty dicts

    candidate_bbox_size = np.array(
        candidate_dict["diffuse_tree"][part_idx]["aabb"]["size"]
    )
    candidate_bbox_size = np.maximum(
        candidate_bbox_size, bbox_size_eps
    )  # Avoid zero volume parts

    for i, requirement_part_bbox_size in enumerate(requirement_part_bbox_sizes):
        part_dicts[i] = deepcopy(fixed_portion)

        # For non-handle parts, compute the scale factor normally
        if fixed_portion["name"] != "handle":
            part_dicts[i]["scale_factor"] = list(
                np.array(requirement_part_bbox_size) / candidate_bbox_size
            )

        # For handles, need to consider the orientation of the selected handle and the orientation of the requirement handle
        else:
            requirement_handle_is_horizontal = (
                requirement_part_bbox_size[0] > requirement_part_bbox_size[1]
            )
            candidate_handle_is_horizontal = (
                candidate_bbox_size[0] > candidate_bbox_size[1]
            )

            # If the orientations are different, rotate the requirement handle by 90 degrees around the z-axis before computing the scale factor
            if requirement_handle_is_horizontal != candidate_handle_is_horizontal:
                rotated_requirement_part_bbox_size = [
                    requirement_part_bbox_size[1],
                    requirement_part_bbox_size[0],
                    requirement_part_bbox_size[2],
                ]
                part_dicts[i]["scale_factor"] = list(
                    np.array(rotated_requirement_part_bbox_size) / candidate_bbox_size
                )
                part_dicts[i]["z_rotate_90"] = True

            # If the orientations are the same, compute the scale factor normally
            else:
                part_dicts[i]["scale_factor"] = list(
                    np.array(requirement_part_bbox_size) / candidate_bbox_size
                )

    return part_dicts


def find_obj_candidates(
    requirement_dict,
    dataset_dir,
    hashbook_path,
    num_states=5,
    metric_compare_handles=False,
    metric_iou_include_base=True,
    metric_num_samples=10000,
    keep_top=5,
    gt_file_name="object.json",
    verbose=False,
):
    """
    Find the best object candidates for selecting the base part using AID\n

    - requirement_dict: the object dictionary of the requirement\n
    - dataset_dir: the directory of the dataset to search in\n
    - hashbook_path: the path to the hashbook for filtering candidates\n
    - num_states: the number of states to average the metric over\n
    - metric_transform_plucker (optional): whether to use Plucker coordinates to move parts when computing the metric\n
    - metric_compare_handles (optional): whether to compare handles when computing the metric\n
    - metric_iou_include_base (optional): whether to include the base when computing the IoU\n
    - metric_scale_factor (optional): the scale factor to scale the object before computing the metric\n
        - Scaling up the object makes the sampling more well distributed\n
    - metric_num_samples (optional): the number of samples to use when computing the metric\n
    - keep_top (optional): the number of top candidates to keep\n
    - gt_file_name (optional): the name of the ground truth json file, which describes a candidate object\n
    - verbose (optional): whether to print the progress\n

    return:\n
    - a list of best object candidates of the form:
        - {"category", "dir", "score"}
    """
    dataset_dir = os.path.abspath(dataset_dir)

    # Load the hashbook
    with open(hashbook_path, "r") as f:
        hashbook = json.load(f)

    # Resolve paths to directories
    category_specified = False
    requirement_category = ""

    # if the category is specified, only search in that category, otherwise search in all categories
    if "obj_cat" in requirement_dict["meta"]:
        requirement_category = requirement_dict["meta"]["obj_cat"]
        category_specified = True

    category_dirs = (
        [os.path.join(dataset_dir, requirement_category)]
        if category_specified
        else [os.path.join(dataset_dir, category) for category in all_categories]
    )

    # Extract requirement data
    requirement_part_names = []
    requirement_part_bboxes = []
    for part in requirement_dict["diffuse_tree"]:
        requirement_part_names.append(part["name"])
        requirement_part_bboxes.append(
            np.concatenate([part["aabb"]["center"], part["aabb"]["size"]])
        )

    # Compute hash of the requirement graph
    requirement_graph_hash = get_hash(requirement_dict)

    # Prefetch list of ids of candidate objects with the same hash
    if category_specified and requirement_graph_hash in hashbook[requirement_category]:
        same_hash_obj_ids = hashbook[requirement_category][requirement_graph_hash]
    else:
        # Use all categories if category is not specified
        same_hash_obj_ids = []
        for category in all_categories:  
            if requirement_graph_hash in hashbook[category]:
                same_hash_obj_ids += hashbook[category][requirement_graph_hash]

    # Iterate through all candidate objects and keep the top k candidates
    best_obj_candidates = []
    for category_dir in category_dirs:
        obj_ids = os.listdir(category_dir)
        for i, obj_id in enumerate(obj_ids):
            if verbose:
                print(
                    f"\r - Finding candidates from {category_dir.split('/')[-1]}: {i+1}/{len(obj_ids)}",
                    end="",
                )

            # Load the candidate object
            obj_dir = os.path.join(category_dir, obj_id)
            with open(os.path.join(obj_dir, gt_file_name), "r") as f:
                obj_dict = json.load(f)
                if "diffuse_tree" not in obj_dict:  # Rename for compatibility
                    obj_dict["diffuse_tree"] = obj_dict.pop("arti_tree")

            # Compute metric for selecting the base if the hash matches or if there are no objects with the same hash
            if obj_id in same_hash_obj_ids or len(same_hash_obj_ids) == 0:
                scores = IoU_cDist(
                    requirement_dict,
                    obj_dict,
                    num_states=num_states,
                    compare_handles=metric_compare_handles,
                    iou_include_base=metric_iou_include_base,
                    num_samples=metric_num_samples,
                )
                base_score = scores["AS-cDist"]

                # Add the candidate to the list of best candidates and keep the top k candidates
                best_obj_candidates.append(
                    {
                        "category": category_dir.split("/")[-1],
                        "dir": obj_dir,
                        "score": base_score,
                    }
                )
                best_obj_candidates = sorted(
                    best_obj_candidates, key=lambda x: x["score"]
                )[:keep_top]
        if verbose:
            print()

    return best_obj_candidates


def pick_and_rescale_parts(
    requirement_dict,
    obj_candidates,
    dataset_dir,
    gt_file_name="object.json",
    verbose=False,
):
    """
    Pick and rescale parts from the object candidates

    - requirement_dict: the object dictionary of the requirement\n
    - obj_candidates: the list of best object candidates for selecting the base part\n
    - dataset_dir: the directory of the dataset to search in\n
    - gt_file_name (optional): the name of the ground truth file, which describes a candidate object\n
    - verbose (optional): whether to print the progress\n

    return:\n
    - parts_to_render: a list of selected parts for the requirement parts in the form:
        - [{name, dir, files, scale_factor=[sx, sy, sz]}, z_rotate_90]
            - z_rotate_90 is True if the part needs to be rotated by 90 degrees around the z-axis
    """

    # Extract requirement data
    requirement_part_names = []
    requirement_part_bbox_sizes = []
    for part in requirement_dict["diffuse_tree"]:
        requirement_part_names.append(part["name"])
        requirement_part_bbox_sizes.append(part["aabb"]["size"])

    # Collect the unique part names and store the indices of the parts with the same name
    unique_requirement_part_names = {}
    for i, part_name in enumerate(requirement_part_names):
        if part_name not in unique_requirement_part_names:
            unique_requirement_part_names[part_name] = [i]
        else:
            unique_requirement_part_names[part_name].append(i)

    parts_to_render = [{} for _ in range(len(requirement_part_names))]

    # Iterate through the object candidates selected for the base part first
    for candidate in obj_candidates:
        if all(
            [len(part) > 0 for part in parts_to_render]
        ):  # Break if all parts are fulfilled
            break

        # Load the candidate object
        with open(os.path.join(candidate["dir"], gt_file_name), "r") as f:
            candidate_dict = json.load(f)

        # Pick parts from the candidate if the part name matches and the part requirement is not yet fulfilled
        for candidate_part_idx, part in enumerate(candidate_dict["diffuse_tree"]):
            part_needed = part["name"] in unique_requirement_part_names
            if not part_needed:
                continue

            part_not_fulfilled = any(
                [
                    len(parts_to_render[i]) == 0
                    for i in unique_requirement_part_names[part["name"]]
                ]
            )
            if not part_not_fulfilled:
                continue

            # Get the indices of the requirement parts that are fulfilled by this candidate part and their bounding box sizes
            fullfill_part_idxs = unique_requirement_part_names[part["name"]]
            fullfill_part_bbox_sizes = [
                requirement_part_bbox_sizes[i] for i in fullfill_part_idxs
            ]

            # Generate all output part dictionaries at once
            part_dicts = _generate_output_part_dicts(
                candidate_dict,
                candidate_part_idx,
                candidate["dir"],
                fullfill_part_bbox_sizes,
                verbose=verbose,
            )

            # Update the output part dictionaries
            [
                parts_to_render[part_idx].update(part_dicts[part_dict_idx])
                for part_dict_idx, part_idx in enumerate(fullfill_part_idxs)
            ]

    # If there are still parts that are not fulfilled
    if any([len(part) == 0 for part in parts_to_render]):
        # Collect the remaining part names
        remaining_part_names = list(
            set(
                [
                    requirement_part_names[i]
                    for i in range(len(requirement_part_names))
                    if len(parts_to_render[i]) == 0
                ]
            )
        )
        if verbose:
            print(
                f" - Parts {remaining_part_names} are not fulfilled by the selected candidates, searching in the dataset..."
            )

        # If the category is specified, only search in that category, otherwise search in all categories
        requirement_category = requirement_dict["meta"]["obj_cat"]
        category_specified = requirement_category != ""
        if category_specified:
            category_dirs = [os.path.join(dataset_dir, requirement_category)]
        else:
            category_dirs = [
                os.path.join(dataset_dir, category) for category in all_categories
            ]

        # Iterate through all objects
        retry = True  # Retry if the category is specified, but some parts are still not fulfilled (See the end of the while loop)
        while retry:
            for category_dir in category_dirs:
                obj_ids = os.listdir(category_dir)
                random.shuffle(obj_ids)  # Randomize the order of the objects
                for i, obj_id in enumerate(obj_ids):
                    if verbose:
                        print(
                            f"\r - Finding missing parts from {category_dir.split('/')[-1]}: {i+1}/{len(obj_ids)}",
                            end="",
                        )

                    # Load the candidate object
                    obj_dir = os.path.join(category_dir, obj_id)
                    with open(os.path.join(obj_dir, gt_file_name), "r") as f:
                        candidate_dict = json.load(f)

                    # Pick the part from the candidate if the part name matches and the parts that are not fulfilled
                    for candidate_part_idx, part in enumerate(
                        candidate_dict["diffuse_tree"]
                    ):
                        part_needed = part["name"] in remaining_part_names

                        if part_needed:
                            # Get the indices of the requirement parts that are fulfilled by this candidate part and their bounding box sizes
                            fullfill_part_idxs = unique_requirement_part_names[
                                part["name"]
                            ]
                            fullfill_part_bbox_sizes = [
                                requirement_part_bbox_sizes[i]
                                for i in fullfill_part_idxs
                            ]

                            # Generate all output part dictionaries at once
                            part_dicts = _generate_output_part_dicts(
                                candidate_dict,
                                candidate_part_idx,
                                obj_dir,
                                fullfill_part_bbox_sizes,
                                verbose=verbose,
                            )

                            # Update the output part dictionaries
                            [
                                parts_to_render[part_idx].update(
                                    part_dicts[part_dict_idx]
                                )
                                for part_dict_idx, part_idx in enumerate(
                                    fullfill_part_idxs
                                )
                            ]

                    if all([len(part) > 0 for part in parts_to_render]):
                        if verbose:
                            print(" -> Found all missing parts")
                        break
                if all([len(part) > 0 for part in parts_to_render]):
                    retry = False
                    break

            # If the category is specified, but some parts are still not fulfilled, search in all categories
            if category_specified and any([len(part) == 0 for part in parts_to_render]):
                if verbose:
                    print(
                        " - Required category is {requirement_category}, but some parts are still not fulfilled, searching in all categories..."
                    )
                category_specified = False
                retry = True
                category_dirs = [
                    os.path.join(dataset_dir, category)
                    for category in all_categories
                    if category != requirement_category
                ]

    # Raise error if there are still parts that are not fulfilled
    if any([len(part) == 0 for part in parts_to_render]):
        raise RuntimeError(
            "Failed to fulfill all requirements, some parts may not exist in the dataset"
        )

    return parts_to_render

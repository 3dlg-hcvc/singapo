import os
import sys
# make the src directory available for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import trimesh
import argparse
import numpy as np
from retrieval.obj_retrieval import find_obj_candidates, pick_and_rescale_parts

def _retrieve_part_meshes(info_dict, save_dir, gt_data_root):
    mesh_save_dir = os.path.join(save_dir, "plys")
    os.makedirs(mesh_save_dir, exist_ok=True)
    HASHBOOK_PATH = "retrieval/retrieval_hash_no_handles.json"
    # Retrieve meshes for the object
    obj_candidates = find_obj_candidates(
        info_dict,
        gt_data_root,
        HASHBOOK_PATH,
        gt_file_name="object.json",
        num_states=5,
        metric_num_samples=4096,
        keep_top=3,
    )
    retrieved_mesh_specs = pick_and_rescale_parts(
        info_dict, obj_candidates, gt_data_root, gt_file_name="object.json"
    )

    scene = trimesh.Scene()
    for i, mesh_spec in enumerate(retrieved_mesh_specs):
        part_spec = info_dict["diffuse_tree"][i]

        # A part may be composed of multiple meshes
        part_meshes = []
        for file in mesh_spec["files"]:
            mesh = trimesh.load(os.path.join(mesh_spec["dir"], file), force="mesh")
            part_meshes.append(mesh)
        part_mesh = trimesh.util.concatenate(part_meshes)

        # Center the mesh at the origin
        part_mesh.vertices -= part_mesh.bounding_box.centroid

        transformation = trimesh.transformations.compose_matrix(
            scale=mesh_spec["scale_factor"],
            angles=[0, 0, np.radians(90) if mesh_spec["z_rotate_90"] else 0],
            translate=part_spec["aabb"]["center"],
        )
        part_mesh.apply_transform(transformation)
        # Save the part mesh as a PLY file
        part_mesh.export(os.path.join(mesh_save_dir, f"part_{i}.ply"))
        # Update object json
        info_dict["diffuse_tree"][i]["plys"] = [f"plys/part_{i}.ply"]
        # Add the mesh to the scene
        scene.add_geometry(part_mesh)

    # Export the scene as a PLY file
    scene.export(os.path.join(save_dir, "object.ply"))

    del mesh, scene
    return info_dict

def main(args):
    # load the json file
    with open(os.path.join(args.src_dir, args.json_name), "r") as f:
        info_dict = json.load(f)
    # retrieve part meshes and update the json
    updated_json = _retrieve_part_meshes(info_dict, args.src_dir, args.gt_data_root)
    # save the updated json
    with open(os.path.join(args.src_dir, args.json_name), "w") as f:
        json.dump(updated_json, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='path to the directory containing object.json')
    parser.add_argument('--json_name', type=str, default='object.json', help='name of the json file')
    parser.add_argument('--gt_data_root', type=str, default='../data', help='path to the ground truth data')
    args = parser.parse_args()
    main(args)
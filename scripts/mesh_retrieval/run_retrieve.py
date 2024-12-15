import os
import subprocess
import argparse
from tqdm.contrib.concurrent import process_map
from functools import partial

def run_retrieve(src_dir, json_name, data_root):
    fn_call = ['python', 'scripts/mesh_retrieval/retrieve.py', '--src_dir', src_dir, '--json_name', json_name, '--gt_data_root', data_root]
    try:
        subprocess.run(fn_call, check=True,  stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f'Error from run_retrieve: {src_dir}')
        print(f'Error: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="path to the experiment folder")
    parser.add_argument("--json_name", type=str, default="object.json", help="name of the json file")
    parser.add_argument("--gt_data_root", type=str, default="../data", help="path to the ground truth data")
    parser.add_argument("--max_workers", type=int, default=6, help="number of images to render for each object")
    args = parser.parse_args()
    
    assert os.path.exists(args.src), f"Src path does not exist: {args.src}"
    assert os.path.exists(args.gt_data_root), f"GT data root does not exist: {args.gt_data_root}"

    root = args.src
    len_root = len(root)
    print('----------- Retrieve Part Meshes -----------')
    src_dirs = []
    for dirpath, dirname, fnames in os.walk(root):
        for fname in fnames:
            if fname.endswith(args.json_name):
                src_dirs.append(dirpath) # save the relative directory path
    print(f"Found {len(src_dirs)} jsons to retrieve part meshes")

    process_map(partial(run_retrieve, json_name=args.json_name, data_root=args.gt_data_root), src_dirs, max_workers=6, chunksize=1)
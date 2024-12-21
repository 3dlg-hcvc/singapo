
import os
import json
import argparse
import subprocess
from tqdm.contrib.concurrent import process_map


def _prepare_data_list(fpath, data_root):
    data_list = []
    with open(fpath, 'r') as f:
        data = json.load(f)
    # load the data list under each key
    for k in data.keys():
        data_list += f'{data_root}/{data[k]}'


def render_data(data):
    # run the blenderproc script to render the input images (default: 20 images per object)
    fn_call = ['blenderproc', 'run', 'scripts/preprocess/render_script.py', '--n_imgs', '20', '--data']

    command = fn_call + [data]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f'Error from render_script: {data}')
        print(f'Error: {e}')

if __name__ == '__main__':
    '''
    Entry point for rendering the input images for the object data list.
    
    This script requires to install the blenderproc package `pip install blenderproc`.
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='data/data_split.json', help="path to the json file that contains the data list")
    parser.add_argument("--data_root", type=str, default='../data', help="path to the root directory of the data")
    parser.add_argument("--max_workers", type=int, default=6, help="number of images to render for each object")
    args = parser.parse_args()

    assert os.path.exists(args.src), "The src json file does not exist"

    # load the data list from the input json file
    data_list = _prepare_data_list(args.src, args.data_root)
    # multi-processing to render the input images
    process_map(render_data, data_list, max_workers=args.max_workers, chunksize=1)





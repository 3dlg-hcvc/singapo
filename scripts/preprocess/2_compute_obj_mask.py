
import os
import json
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from math import sqrt

def compute_obj_mask(data_root, model_id, resize=256, center_crop=224, n_patches=256):
    img_dir = os.path.join(data_root, model_id, "imgs")
    masks = np.empty((20, n_patches), dtype=bool)
    for i in range(20):
        fname = str(i).zfill(2) + ".png"
        # load alpha channel
        img_path = os.path.join(img_dir, fname)
        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (resize, resize), interpolation=cv.INTER_CUBIC)
        # center crop to 224x224
        center = img.shape
        x = center[1] / 2 - center_crop / 2
        y = center[0] / 2 - center_crop / 2
        crop_img = img[int(y) : int(y + center_crop), int(x) : int(x + center_crop)]
        # get alpha channel
        alpha = crop_img[:, :, 3]
        # patchify the image into 16x16 patches
        n_patches_one_side = int(sqrt(n_patches))
        patch_size = center_crop // int(n_patches_one_side)
        mask = np.mean(
            alpha.reshape(n_patches_one_side, patch_size, n_patches_one_side, patch_size), axis=(1, 3)
        )
        mask = mask > 0.0
        m = mask.reshape(-1)
        masks[i] = m

    return masks

def save_obj_mask(masks, save_path):
    np.save(save_path, masks)


def _prepare_data_list(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    data_list = []
    for k in data.keys():
        data_list += data[k]


if __name__ == "__main__":
    '''
    Script to compute object foreground masks on the image patches (only used for training)

    The file structure:
        <data_root>
        ├── <model_id>
        │   ├── imgs (this folder should be ready)
        │   │   ├── 00.png
        │   │   ├── 01.png
        │   │   ├── ...
        │   ├── features 
        │   │   ├── patch_obj_masks.npy (will save the mask here)

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default="data/data_split.json", help="path to the json file that contains the data list")
    parser.add_argument("--data_root", type=str, default="../data", help="the root directory of the original data")
    args = parser.parse_args()

    assert os.path.exists(args.json_file), "The json file does not exist"
    assert os.path.exists(args.data_root), "The data root does not exist"

    data_list = _prepare_data_list(args.json_file)
    for model_id in tqdm(data_list):
        dst_dir = os.path.join(args.data_root, model_id, "features")
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, "patch_obj_masks.npy")

        mask = compute_obj_mask(args.data_root, model_id)
        save_obj_mask(mask, dst_path)


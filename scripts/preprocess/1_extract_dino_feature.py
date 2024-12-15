
import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# Load the DinoV2 model (with register version)
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda()

# transformation for input images
transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def _make_white_background(src_img):
    src_img.load() # required for png.split()
    background = Image.new("RGB", src_img.size, (255, 255, 255))
    background.paste(src_img, mask=src_img.split()[3]) # 3 is the alpha channel
    return background

def load_images(data_root, model_id):
    """Load images for a given model_id in the image folder"""
    img_dir = f'{data_root}/{model_id}/imgs'
    img_batch = torch.empty(20, 3, 224, 224, dtype=torch.float32)
    for i in range(20):
        fname = str(i).zfill(2)
        img_path = f'{img_dir}/{fname}.png'
        with Image.open(img_path) as img:
            img_ = _make_white_background(img) # default background is black when converting RGBA to RGB
        img = transform(img_) 
        img_batch[i] = img
    return img_batch.cuda()

def extract_patch_features(input):
    with torch.no_grad():
        features = dinov2_vitb14_reg.forward_features(input)["x_norm_patchtokens"]
    return features.cpu().numpy()

def save_patch_features(features, save_path):
    np.save(save_path, features) # (20, 256, 768)

def _prepare_data_list(fpath):
    '''track the data list from the json file'''
    with open(fpath, 'r') as f:
        data = json.load(f)
    data_list = []
    for k in data.keys():
        data_list += data[k]

if __name__ == '__main__':
    '''
    Script to extract image features from DinoV2
    
    The file structure:
        <data_root>
        ├── <model_id>
        │   ├── imgs (this folder should be ready)
        │   │   ├── 00.png
        │   │   ├── 01.png
        │   │   ├── ...
        │   ├── features 
        │   │   ├── dinov2_patch_reg.npy (will save the extracted features here)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='data/data_split.json', help="path to the json file, which contains the data list")
    parser.add_argument("--data_root", type=str, default='../data', help='the root directory of the original data')
    args = parser.parse_args()

    assert os.path.exists(args.json_path), 'The json file does not exist'
    assert os.path.exists(args.data_root), "The data root does not exist"

    
    # Extract features from Dinov2 patch tokens
    data_list = _prepare_data_list(args.json_path)

    for model_id in tqdm(data_list):
        dst_dir = os.path.join(args.data_root, model_id, 'features')
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, 'dinov2_patch_reg.npy')

        input_batch = load_images(args.data_root, model_id)
        features = extract_patch_features(input_batch)
        save_patch_features(features, dst_path)


    
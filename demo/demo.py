import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import torch
import argparse
import numpy as np
from PIL import Image
from utils.plot import viz_graph
from utils.misc import load_config
import torchvision.transforms as T
from diffusers import DDPMScheduler
from models.denoiser import Denoiser
from utils.render import rescale_axis
from utils.refs import joint_ref, sem_ref
from scripts.graph_pred.api import predict_graph
from utils.render import prepare_meshes, draw_boxes_axiss_anim
from data.utils import make_white_background, load_input_from, convert_data_range, parse_tree

def load_img(img_path):
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    with Image.open(img_path) as img:
        if img.mode == 'RGBA':
            img = make_white_background(img)
        img = transform(img)
    img_batch = img.unsqueeze(0).cuda()
    return img_batch

def extract_dino_feature(img_path):
    print('Extracting DINO feature...')
    input_img = load_img(img_path)
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda()
    with torch.no_grad():
        feat = dinov2_vitb14_reg.forward_features(input_img)["x_norm_patchtokens"]
    # release the GPU memory of the model
    torch.cuda.empty_cache()
    return feat

def set_scheduler(n_steps=100):
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear', prediction_type='epsilon')
    scheduler.set_timesteps(n_steps)
    return scheduler

def prepare_model_input(data, cond, feat, n_samples):
    # attention masks
    attr_mask = torch.from_numpy(cond['attr_mask']).unsqueeze(0).repeat(n_samples, 1, 1)
    key_pad_mask = torch.from_numpy(cond['key_pad_mask']).unsqueeze(0).repeat(n_samples, 1, 1)
    graph_mask = torch.from_numpy(cond['adj_mask']).unsqueeze(0).repeat(n_samples, 1, 1)
    # input image feature
    f = feat.repeat(n_samples, 1, 1)
    # input noise
    noise = torch.randn(data.shape, dtype=torch.float32).repeat(n_samples, 1, 1)
    # dummy image feature (used for guided diffusion)
    dummy_feat = torch.from_numpy(np.load('systems/dino_dummy.npy').astype(np.float32))
    dummy_feat = dummy_feat.unsqueeze(0).repeat(n_samples, 1, 1)
    # dummy object category
    cat = torch.zeros(1, dtype=torch.long).repeat(n_samples)
    return {
        "noise": noise.cuda(),
        "attr_mask": attr_mask.cuda(),
        "key_pad_mask": key_pad_mask.cuda(),
        "graph_mask": graph_mask.cuda(),
        "dummy_f": dummy_feat.cuda(),
        'cat': cat.cuda(),
        'f': f.cuda(),  
    }

def save_graph(pred_graph, save_dir):
    print(f'Saving the predicted graph to {save_dir}/pred_graph.json')
    # save the response
    with open(os.path.join(save_dir, "pred_graph.json"), "w") as f:
        json.dump(pred_graph, f, indent=4)
    # Visualize the graph
    img_graph = Image.fromarray(viz_graph(pred_graph))
    img_graph.save(os.path.join(save_dir, "pred_graph.png"))

def forward(model, scheduler, inputs, omega=0.5):
    print('Running inference...')
    noisy_x = inputs['noise']
    for t in scheduler.timesteps:
        timesteps = torch.tensor([t], device=inputs['noise'].device)
        outputs_cond = model(
            x=noisy_x,
            cat=inputs['cat'],
            timesteps=timesteps,
            feat=inputs['f'], 
            key_pad_mask=inputs['key_pad_mask'],
            graph_mask=inputs['graph_mask'],
            attr_mask=inputs['attr_mask'],
            label_free=True,
        ) # take condtional image as input
        if omega != 0:
            outputs_free = model(
                x=noisy_x,
                cat=inputs['cat'],
                timesteps=timesteps,
                feat=inputs['dummy_f'], 
                key_pad_mask=inputs['key_pad_mask'],
                graph_mask=inputs['graph_mask'],
                attr_mask=inputs['attr_mask'],
                label_free=True,
            ) # take the dummy DINO features for the condition-free mode
            noise_pred = (1 + omega) * outputs_cond['noise_pred'] - omega * outputs_free['noise_pred']
        else:
            noise_pred = outputs_cond['noise_pred']
        noisy_x = scheduler.step(noise_pred, t, noisy_x).prev_sample
    return noisy_x

def _convert_json(x, c):
    out = {"meta": {}, "diffuse_tree": []}
    n_nodes = c["n_nodes"]
    par = c["parents"].tolist()
    adj = c["adj"]
    np.fill_diagonal(adj, 0) # remove self-loop for the root node
    if "obj_cat" in c:
        out["meta"]["obj_cat"] = c["obj_cat"]

    # convert the data to original range
    data = convert_data_range(x)
    # parse the tree
    out["diffuse_tree"] = parse_tree(data, n_nodes, par, adj)
    return out

def post_process(output, cond, save_root, gt_data_root, visualize=False):
    print('Post-processing...')
    N = output.shape[0]
    for i in range(N):
        # convert the raw model output to the json format
        out_json = _convert_json(output, cond)
        save_dir = os.path.join(save_root, str(i))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "object.json"), "w") as f:
            json.dump(out_json, f, indent=4)
        

        # retrieve part meshes (call python script)
        print(f"Retrieving part meshes for the object {i}...")
        os.system(f"python scripts/mesh_retrieval/retrieve.py --src_dir {save_dir} --json_name object.json --gt_data_root {gt_data_root}")
        
        if visualize:
            print(f"Visualizing the object {i}...")

            # visualize the object in two states with parts represented in bbox
            vis_meshes = prepare_meshes(out_json)
            vis_img = Image.fromarray(draw_boxes_axiss_anim(
                vis_meshes["bbox_0"], 
                vis_meshes["bbox_1"], 
                vis_meshes["axiss"], 
                mode="graph", 
                resolution=256
            ))

            # save the image
            vis_img.save(os.path.join(save_dir, "vis_img.png"))


    

def load_model(ckpt_path, config):
    print('Loading model from checkpoint...')
    model = Denoiser(config)
    state_dict = torch.load(ckpt_path)['state_dict']
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.cuda()

def run_demo(args):
    # extract DINOV2 feature from the input image
    feat = extract_dino_feature(args.img_path)
    # extract graph from the input image
    if args.use_example_graph: # use the example graph for inference
        with open('demo/example_graph.json', 'r') as f:
            pred_graph = json.load(f)
    else: # use gpt-4o to predict the graph
        pred_graph = predict_graph(args.img_path)
    save_graph(pred_graph, args.save_dir)
    # load input from the predicted graph
    data, cond = load_input_from(pred_graph, K=32)
    # prepare the model input
    inputs = prepare_model_input(data, cond, feat, n_samples=args.n_samples)
    # set the scheduler of the DDPM
    scheduler = set_scheduler(args.n_denoise_steps)
    # load the checkpoint of the model
    model = load_model(args.ckpt_path, args.config.system.model)
    # inference
    with torch.no_grad():
        output = forward(model, scheduler, inputs, omega=args.omega).cpu().numpy()
    # post-process
    post_process(output, cond, args.save_dir, args.gt_data_root, visualize=True)


if __name__ == '__main__':
    '''
    Script for running the inference on an example image input.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default='demo/demo_input.png', help="path to the input image")
    parser.add_argument("--ckpt_path", type=str, default='exps/singapo/final/ckpts/last.ckpt', help="path to the checkpoint of the model")
    parser.add_argument("--config_path", type=str, default='exps/singapo/final/config/parsed.yaml', help="path to the config file")
    parser.add_argument("--use_example_graph", action="store_true", help="if you don't have the openai key yet, turn on to use the example graph for inference")
    parser.add_argument("--save_dir", type=str, default='demo/demo_output', help="path to save the output")
    parser.add_argument("--gt_data_root", type=str, default='../data', help="the root directory of the original data, used for part mesh retrieval")
    parser.add_argument("--n_samples", type=int, default=3, help="number of samples to generate given the input")
    parser.add_argument("--omega", type=float, default=0.5, help="the weight of the condition-free mode in the inference")
    parser.add_argument("--n_denoise_steps", type=int, default=100, help="number of denoising steps")
    args = parser.parse_args()

    assert os.path.exists(args.img_path), "The input image does not exist"
    assert os.path.exists(args.ckpt_path), "The checkpoint does not exist"
    assert os.path.exists(args.config_path), "The config file does not exist"
    os.makedirs(args.save_dir, exist_ok=True)

    config = load_config(args.config_path)
    args.config = config

    run_demo(args)
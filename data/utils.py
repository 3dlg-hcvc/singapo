import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from PIL import Image
from utils.render import rescale_axis
from utils.refs import joint_ref, sem_ref


def make_white_background(src_img):
    '''Make the white background for the input RGBA image.'''
    src_img.load() 
    background = Image.new("RGB", src_img.size, (255, 255, 255))
    background.paste(src_img, mask=src_img.split()[3]) # 3 is the alpha channel
    return background

def build_graph(tree, K=32):
    '''
    Function to build graph from the node list.
    
    Args:
        nodes: list of nodes
        K: the maximum number of nodes in the graph
    Returns:
        adj: adjacency matrix, records the 1-ring relationship (parent+children) between nodes
        edge_list: list of edges, for visualization
    '''
    adj = np.zeros((K, K), dtype=np.float32)
    parents = []
    for node in tree:
        # 1-ring relationship
        if node['parent'] != -1:
            adj[node['id'], node['parent']] = 1
            parents.append(node['parent'])
        else:
            adj[node['id'], node['id']] = 1
            parents.append(-1)
        for child_id in node['children']:
            adj[node['id'], child_id] = 1 

    return {
        'adj': adj,
        'parents': np.array(parents, dtype=np.int8)
    }

def load_input_from(pred_file, K=32):
    '''
    Function to parse input item from a file containing the predicted graph
    '''
    
    cond = {} # conditional information and axillary data
    # prepare node data
    n_nodes = len(pred_file['diffuse_tree'])
    # prepare graph
    pred_graph = build_graph(pred_file['diffuse_tree'], K)

    # attr mask (for Local Attention)
    attr_mask = np.eye(K, K, dtype=bool)
    attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
    cond['attr_mask'] = attr_mask

    # key padding mask (for Global Attention)
    pad_mask = np.zeros((K*5, K*5), dtype=bool)
    pad_mask[:, :n_nodes*5] = 1
    cond['key_pad_mask'] = pad_mask

    # adj mask (for Graph Relation Attention)
    adj_mask = pred_graph['adj'][:].astype(bool)
    adj_mask = adj_mask.repeat(5, axis=0).repeat(5, axis=1)
    adj_mask[n_nodes*5:, :] = 1
    cond['adj_mask'] = adj_mask

    # placeholder
    data = np.zeros((K*5, 6), dtype=bool)
    cond['cat'] = 2

    # axillary info
    cond['adj'] = pred_graph['adj']
    cond['parents'] = np.zeros(K, dtype=np.int8)
    cond['parents'][:n_nodes] = pred_graph['parents']
    cond['n_nodes'] = n_nodes

    return data, cond

def convert_data_range(x):
    '''postprocessing: convert the raw model output to the original range, following CAGE'''
    x = x.reshape(-1, 30)  # (K, 30)
    aabb_max = x[:, 0:3]
    aabb_min = x[:, 3:6]
    center = (aabb_max + aabb_min) / 2.0
    size = (aabb_max - aabb_min).clip(min=5e-3)

    j_type = np.mean(x[:, 6:12], axis=1)
    j_type = ((j_type + 0.5) * 5).clip(min=1.0, max=5.0).round()

    axis_d = x[:, 12:15]
    axis_d = axis_d / (
        np.linalg.norm(axis_d, axis=1, keepdims=True) + np.finfo(float).eps
    )
    axis_o = x[:, 15:18]

    j_range = (x[:, 18:20] + x[:, 20:22] + x[:, 22:24]) / 3
    j_range = j_range.clip(min=-1.0, max=1.0)
    j_range[:, 0] = j_range[:, 0] * 360
    j_range[:, 1] = j_range[:, 1]

    label = np.mean(x[:, 24:30], axis=1)
    label = ((label + 0.8) * 5).clip(min=0.0, max=7.0).round()
    return {
        "center": center,
        "size": size,
        "type": j_type,
        "axis_d": axis_d,
        "axis_o": axis_o,
        "range": j_range,
        "label": label,
    }

def parse_tree(data, n_nodes, par, adj):
    tree = []
    # convert to json format
    for i in range(n_nodes):
        node = {"id": i}
        node["name"] = sem_ref["bwd"][int(data["label"][i].item())]
        node["parent"] = int(par[i])
        node["children"] = [
            int(child) for child in np.where(adj[i] == 1)[0] if child != par[i]
        ]
        node["aabb"] = {}
        node["aabb"]["center"] = data["center"][i].tolist()
        node["aabb"]["size"] = data["size"][i].tolist()
        node["joint"] = {}
        if node['name'] == 'base':
            node["joint"]["type"] = 'fixed'
        else:
            node["joint"]["type"] = joint_ref["bwd"][int(data["type"][i].item())]
        if node["joint"]["type"] == "fixed":
            node["joint"]["range"] = [0.0, 0.0]
        elif node["joint"]["type"] == "revolute":
            node["joint"]["range"] = [0.0, float(data["range"][i][0])]
        elif node["joint"]["type"] == "continuous":
            node["joint"]["range"] = [0.0, 360.0]
        elif (
            node["joint"]["type"] == "prismatic" or node["joint"]["type"] == "screw"
        ):
            node["joint"]["range"] = [0.0, float(data["range"][i][1])]
        node["joint"]["axis"] = {}
        # relocate the axis to visualize well
        axis_o, axis_d = rescale_axis(
            int(data["type"][i].item()),
            data["axis_d"][i],
            data["axis_o"][i],
            data["center"][i],
        )
        node["joint"]["axis"]["direction"] = axis_d
        node["joint"]["axis"]["origin"] = axis_o
        # append node to the tree
        tree.append(node)
    return tree

def convert_json(x, c, prefix=''):
    out = {"meta": {}, "diffuse_tree": []}
    n_nodes = c[f"{prefix}n_nodes"][0].item()
    par = c[f"{prefix}parents"][0].cpu().numpy().tolist()
    adj = c[f"{prefix}adj"][0].cpu().numpy()
    np.fill_diagonal(adj, 0) # remove self-loop for the root node
    if f"{prefix}obj_cat" in c:
        out["meta"]["obj_cat"] = c[f"{prefix}obj_cat"][0]

    # convert the data to original range
    data = convert_data_range(x)
    # parse the tree
    tree = parse_tree(data, n_nodes, par, adj)
    out["diffuse_tree"] = tree
    return out
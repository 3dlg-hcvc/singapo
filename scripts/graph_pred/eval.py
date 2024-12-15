import os
import json
import argparse
import networkx as nx
from tqdm import tqdm

def get_hash(file, key='diffuse_tree'):
    tree = file[key]
    G = nx.DiGraph()
    for node in tree:
        G.add_node(node['id'])
        if node['parent'] != -1:
            G.add_edge(node['id'], node['parent'])
    hashcode = nx.weisfeiler_lehman_graph_hash(G)
    return hashcode

if __name__ == "__main__":
    '''Script to evaluate the accuracy of the generated graphs'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help='path to the experiment directory')
    parser.add_argument('--gt_data_root', type=str, required=True, help='root directory of the ground-truth data')
    parser.add_argument('--gt_json_name', type=str, default='object.json', help='Path to the ground truth data')
    args = parser.parse_args()

    assert os.path.exists(args.exp_dir), "The experiment directory does not exist"
    assert os.path.exists(args.gt_data_root), "The ground-truth data root does not exist"

    exp_dir = args.exp_dir
    gt_data_dir = args.gt_data_root

    acc = 0
    files = os.listdir(exp_dir)
    sorted(files)
    total = len(files)
    wrong_files = []
    for file in tqdm(files):
        tokens = file.split('@')
        gt_dir = f'{gt_data_dir}'
        for token in tokens[:-1]:
            gt_dir = os.path.join(gt_dir, token)
        with open(os.path.join(gt_dir, args.gt_json_name)) as f:
            gt = json.load(f)
        # load json files
        with open(os.path.join(exp_dir, file)) as f:
            pred = json.load(f)
        # get hash for the graph
        pred_hash = get_hash(pred)
        gt_hash = get_hash(gt)
        # compare hash
        if pred_hash == gt_hash:
            acc += 1
        else:
            wrong_files.append(file)


    with open(os.path.join(os.path.dirname(exp_dir), f'acc_{os.path.basename(exp_dir)}.json'), 'w') as f:
        json.dump({'acc': acc/total, 'wrong_files': wrong_files}, f, indent=4)

        

    
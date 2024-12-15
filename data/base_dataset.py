import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from utils.refs import cat_ref, sem_ref, joint_ref
from data.utils import build_graph

class BaseDataset(Dataset):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
    
    def _filter_models(self, models_ids):
        '''
        Filter out models that has more than K nodes.
        '''
        data_root = self.hparams.root
        filtered = []
        for model_id in models_ids:
            path = os.path.join(data_root, model_id, self.json_name)
            with open(path, 'r') as f:
                json_file = json.load(f)
                if len(json_file['diffuse_tree']) <= self.hparams.K:
                    filtered.append(model_id)
        return filtered
    
    def get_acd_mapping(self):
        self.category_mapping = {
            'armoire': 'StorageFurniture',
            'bookcase': 'StorageFurniture',
            'chest_of_drawers': 'StorageFurniture',
            'desk': 'Table',
            'dishwasher': 'Dishwasher',
            'hanging_cabinet': 'StorageFurniture',
            'kitchen_cabinet': 'StorageFurniture',
            'microwave': 'Microwave',
            'nightstand': 'StorageFurniture',
            'oven': 'Oven',
            'refrigerator': 'Refrigerator',
            'sink_cabinet': 'StorageFurniture',
            'tv_stand': 'StorageFurniture',
            'washer': 'WashingMachine',
            'table': 'Table',
            'cabinet': 'StorageFurniture',
            'hanging_cabinet': 'StorageFurniture',
        }

    def _random_permute(self, graph, nodes):
        '''
        Function to randomly permute the nodes and update the graph and node attribute info.

        Args:
            graph: a dictionary containing the adjacency matrix, edge list, and root node
            nodes: a list of nodes
        Returns:
            graph_permuted: a dictionary containing the updated adjacency matrix, edge list, and root node
            nodes_permuted: a list of permuted nodes
        '''
        N = len(nodes)
        order = np.random.permutation(N)
        graph_permuted = self._reorder_nodes(graph, order)
        exchange = [0] * len(order)
        for i in range(len(order)):
            exchange[order[i]] = i
        nodes_permuted = nodes[exchange, :]
        return graph_permuted, nodes_permuted
    
    def _prepare_node_data(self, node):
        # semantic label
        label = np.array([sem_ref['fwd'][node['name']]], dtype=np.float32) / 5. - 0.8 # (1,), range from -0.8 to 0.8
        # joint type
        joint_type = np.array([joint_ref['fwd'][node['joint']['type']] / 5.], dtype=np.float32) - 0.5 # (1,), range from -0.8 to 0.8
        # aabb
        aabb_center = np.array(node['aabb']['center'], dtype=np.float32)  # (3,), range from -1 to 1
        aabb_size = np.array(node['aabb']['size'], dtype=np.float32) # (3,), range from -1 to 1
        aabb_max = aabb_center + aabb_size / 2
        aabb_min = aabb_center - aabb_size / 2
        # joint axis and range
        if node['joint']['type'] == 'fixed':
            axis_dir = np.zeros((3,), dtype=np.float32)
            axis_ori = aabb_center
            joint_range = np.zeros((2,), dtype=np.float32)
        else:
            if node['joint']['type'] == 'revolute' or node['joint']['type'] == 'continuous':
                joint_range = np.array([node['joint']['range'][1]], dtype=np.float32) / 360. 
                joint_range = np.concatenate([joint_range, np.zeros((1,), dtype=np.float32)], axis=0) # (2,) 
            elif node['joint']['type'] == 'prismatic' or node['joint']['type'] == 'screw':
                joint_range = np.array([node['joint']['range'][1]], dtype=np.float32) 
                joint_range = np.concatenate([np.zeros((1,), dtype=np.float32), joint_range], axis=0) # (2,) 
            axis_dir = np.array(node['joint']['axis']['direction'], dtype=np.float32) * 0.7 # (3,), range from -0.7 to 0.7
            # make sure the axis is pointing to the positive direction
            if np.sum(axis_dir > 0) < np.sum(-axis_dir > 0): 
                axis_dir = -axis_dir 
                joint_range = -joint_range
            axis_ori = np.array(node['joint']['axis']['origin'], dtype=np.float32) # (3,), range from -1 to 1
            if (node['joint']['type'] == 'prismatic' or node['joint']['type'] == 'screw') and node['name'] != 'door':
                axis_ori = aabb_center
        node_data = np.concatenate([aabb_max, aabb_min, joint_type.repeat(6), axis_dir, axis_ori, joint_range.repeat(3), label.repeat(6)], axis=0)
        return node_data


    def _reorder_nodes(self, graph, order):
        '''
        Function to reorder nodes in the graph and 
        update the adjacency matrix, edge list, and root node.

        Args:
            graph: a dictionary containing the adjacency matrix, edge list, and root node
            order: a list of indices for reordering
        Returns:
            new_graph: a dictionary containing the updated adjacency matrix, edge list, and root node
        '''
        N = len(order)
        mapping = {i: order[i] for i in range(N)}
        mapping.update({i: i for i in range(N, self.hparams.K)})
        G = nx.from_numpy_array(graph['adj'], create_using=nx.Graph)
        G_ = nx.relabel_nodes(G, mapping)
        new_adj = nx.adjacency_matrix(G_, G.nodes).todense()
        return {
            'adj': new_adj.astype(np.float32),
            'parents': graph['parents'][order]
        }


    def _prepare_input_GT(self, file, model_id):
        '''
        Function to parse input item from a json file for the CAGE training.
        '''
        tree = file['diffuse_tree']
        K = self.hparams.K # max number of nodes
        cond = {} # conditional information and axillary data
        cond['parents'] = np.zeros(K, dtype=np.int8)

        # prepare node data
        nodes = []
        for node in tree:
            node_data = self._prepare_node_data(node) # (30,)     
            nodes.append(node_data) 
        nodes = np.array(nodes, dtype=np.float32)
        n_nodes = len(nodes)

        # prepare graph
        graph = build_graph(tree, self.hparams.K)
        
        # augment node order
        if self.mode == 'train': # perturb the node order for training
            graph, nodes = self._random_permute(graph, nodes)

        # pad the nodes to K with empty nodes
        if n_nodes < K:
            empty_node = np.zeros((nodes[0].shape[0],))
            data = np.concatenate([nodes, [empty_node] * (K - n_nodes)], axis=0, dtype=np.float32) # (K, 30)
        else:
            data = nodes
        data = data.reshape(K*5, 6) # (K * n_attr, 6)

        # attr mask (for Local Attention)
        attr_mask = np.eye(K, K, dtype=bool)
        attr_mask = attr_mask.repeat(5, axis=0).repeat(5, axis=1)
        cond['attr_mask'] = attr_mask

        # key padding mask (for Global Attention)
        pad_mask = np.zeros((K*5, K*5), dtype=bool)
        pad_mask[:, :n_nodes*5] = 1
        cond['key_pad_mask'] = pad_mask

        # adj mask (for Graph Relation Attention)
        adj_mask = graph['adj'][:].astype(bool)
        adj_mask = adj_mask.repeat(5, axis=0).repeat(5, axis=1)
        adj_mask[n_nodes*5:, :] = 1
        cond['adj_mask'] = adj_mask

        # object category
        cond['cat'] = cat_ref[file['meta']['obj_cat']]

        # axillary info
        cond['name'] = model_id
        cond['adj'] = graph['adj']
        cond['parents'][:n_nodes] = graph['parents']
        cond['n_nodes'] = n_nodes
        cond['obj_cat'] = file['meta']['obj_cat']
        
        return data, cond

    def _prepare_input(self, model_id, pred_file, gt_file=None):
        '''
        Function to parse input item from pred_file, and parse GT from gt_file if available.
        '''
        K = self.hparams.K # max number of nodes
        cond = {} # conditional information and axillary data
        # prepare node data
        n_nodes = len(pred_file['diffuse_tree'])
        # prepare graph
        pred_graph = build_graph(pred_file['diffuse_tree'], K)
        # dummy GT data
        data = np.zeros((K*5, 6), dtype=np.float32)
        
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

        # placeholder category, won't be used if category is given (below)
        cond['cat'] = cat_ref['StorageFurniture']
        cond['obj_cat'] = 'StorageFurniture'
        # if object category is given as input
        if not self.hparams.get('test_label_free', False):
            assert 'meta' in pred_file, 'meta not found in the json file.'
            assert 'obj_cat' in pred_file['meta'], 'obj_cat not found in the metadata of the json file.'
            category = pred_file['meta']['obj_cat']
            if self.map_cat:  # for ACD dataset
                category = self.category_mapping[category]
            cond['cat'] = cat_ref[category]
            cond['obj_cat'] = category

        # axillary info
        cond['name'] = model_id
        cond['adj'] = pred_graph['adj']
        cond['parents'] = np.zeros(K, dtype=np.int8)
        cond['parents'][:n_nodes] = pred_graph['parents']
        cond['n_nodes'] = n_nodes

        if gt_file is not None:
            # prepare node data
            gt_n_nodes = len(gt_file['diffuse_tree'])
            gt_nodes = []
            for gt_node in gt_file['diffuse_tree']:
                gt_node_data = self._prepare_node_data(gt_node) # (30,)     
                gt_nodes.append(gt_node_data) 
            gt_nodes = np.array(gt_nodes, dtype=np.float32)
            # pad the nodes to K with empty nodes
            if gt_n_nodes < K:
                empty_node = np.zeros((gt_nodes[0].shape[0],))
                data = np.concatenate([gt_nodes, [empty_node] * (K - gt_n_nodes)], axis=0, dtype=np.float32) # (K, 30)
            else:
                data = gt_nodes
            data = data.reshape(K*5, 6) # (K * n_attr, 6)
            gt_graph = build_graph(gt_file['diffuse_tree'], K)
            cond['gt_parents'] = np.zeros(K, dtype=np.int8)
            cond['gt_parents'][:gt_n_nodes] = gt_graph['parents']
            cond['gt_n_nodes'] = gt_n_nodes
            cond['gt_adj'] = gt_graph['adj']

            if not self.hparams.get('test_label_free', False):
                category = gt_file['meta']['obj_cat']
                if self.map_cat:  # for ACD dataset
                    category = self.category_mapping[category]
                cond['gt_obj_cat'] = category

        return data, cond

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError


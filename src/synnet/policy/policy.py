import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from synnet.utils.data_utils import Reaction, ReactionSet, SyntheticTreeSet, Skeleton, SkeletonSet, Program
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import os
from pathlib import Path
import networkx as nx
from copy import deepcopy



class RxnPolicy(nn.Module):
    """
    Simple neural network policy for solving the hill climbing task.
    Consists of one common dense layer for both policy and value estimate and
    another dense layer for each.
    """

    def __init__(self, n_obs, n_hidden, n_actions, sk, hash_dir, rxns):
        super(RxnPolicy, self).__init__()
        self.n_obs = n_obs
        if self.n_obs != 4278:
            breakpoint()
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.dense1 = nn.Linear(n_obs, n_hidden)
        self.dense_p = nn.Linear(n_hidden, n_actions)
        self.dense_v = nn.Linear(n_hidden, 1)
        self.tree = sk.tree
        self.hash_dir = hash_dir
        self.entries = self.compute_entries(sk)
        self.rxns = rxns

    
    def compute_entries(self, sk):
        sk.mask = [sk.tree_root]
        rxn_graph, node_map, reverse_node_map = sk.rxn_graph()
        for n in rxn_graph:
            rxn_graph.nodes[n]['rxn_id'] = -1
        self.rxn_graph = rxn_graph     
        self.node_map = node_map     
        self.reverse_node_map = reverse_node_map
        nums = {}
        top_rxn = node_map[list(sk.tree.successors(sk.tree_root))[0]]        
        Skeleton.num_rxns(rxn_graph, top_rxn, nums)
        entries = {}        
        for n in sorted(nums, key=lambda k: -nums[k]):            
            if nums[n] in [1, 2]:
                entries[n] = self.hash_dir            
        self.rxn_to_nodes = np.array(sorted(self.node_map))
        self.parents = {n: list(rxn_graph.predecessors(n)) for n in rxn_graph}
        return entries # entry nodes
    
    def action_mask(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        if len(obs.shape) == 2:
            return torch.stack([self.action_mask(obs[b]) for b in range(obs.shape[0])], dim=0)
        rxns_mask = obs[-self.n_actions:].reshape(-1, 91)
        mask = rxns_mask.sum(axis=-1, keepdims=True) == 0 # unfilled nodes        
        # TODO: for each node whose parent is not in, mask it out
        for i in mask[:, 0].argwhere().flatten():
            i = int(i)
            if self.parents[i] and mask[self.parents[i][0], 0]:
                mask[i, 0] = False
        mask = torch.tile(mask, (1,91))
        # use which nodes are filled to get Program

        rxn_graph_copy = deepcopy(self.rxn_graph)
        bool_mask = rxns_mask.sum(axis=-1) > 0 # filled
        if bool_mask.sum().item() == 0:            
            assert len(mask.sum(axis=-1).argwhere().flatten()) == 1
            parent = int(mask.sum(axis=-1).argwhere().flatten()[0])
            num_reactant = len(list(self.tree.successors(self.rxn_to_nodes[parent])))
            assert parent == self.node_map[self.rxn_to_nodes[parent]]
            r_mask = []
            for r in range(91):
                if Reaction(self.rxns[r]).num_reactant != num_reactant:
                    r_mask.append(False)
                else:
                    rxn_graph_copy.nodes[parent]['rxn_id'] = r
                    p = Program(rxn_graph_copy)
                    path = os.path.join(self.hash_dir, p.get_path())
                    r_mask.append(os.path.exists(path))
            mask[parent] = torch.from_numpy(np.array(r_mask))
            return mask.flatten()
        filled_nodes = self.rxn_to_nodes[bool_mask]
        rxn_ids = rxns_mask[bool_mask].argmax(axis=-1).cpu().numpy()

        entries = deepcopy(self.entries) # root
        for filled, rxn_id in zip(filled_nodes, rxn_ids):
            node = self.node_map[filled]
            rxn_graph_copy.nodes[node]['rxn_id'] = int(rxn_id)            
        for filled in filled_nodes:
            node = self.node_map[filled]
            pred = self.parents[node]
            p = Program(rxn_graph_copy)
            entries[node] = os.path.join(self.hash_dir, p.get_path()) # TODO: only works for size-2 programs

        # for each node whose parent is in, get mask
        for unfilled in (~bool_mask).argwhere().flatten():
            unfilled = int(unfilled)  
            # mask out bi-mol rxns if unfilled is uni-mol, vice-versa            
            num_reactant = len(list(self.tree.successors(self.reverse_node_map[unfilled])))
            r_mask = [num_reactant == Reaction(self.rxns[r]).num_reactant for r in range(91)]
            if bool_mask[self.parents[unfilled]].item():
                pred = self.parents[unfilled][0]                
                for r in range(91):
                    if not r_mask[r]:
                        continue
                    rxn_graph_copy.nodes[unfilled]['rxn_id'] = r
                    p = Program(rxn_graph_copy)
                    ppath = Path(entries[pred])
                    path = os.path.join(ppath.parent, ppath.stem, p.get_path())
                    r_mask[r] = os.path.exists(path)
                rxn_graph_copy.nodes[unfilled]['rxn_id'] = -1
                mask[unfilled] = torch.from_numpy(np.array(r_mask))
        mask = mask.reshape(self.n_actions)
        
        return mask
        

    def forward(self, obs):
        obs_one_hot = obs.float()
        h_relu = F.relu(self.dense1(obs_one_hot))
        logits = self.dense_p(h_relu)           
        mask = self.action_mask(obs_one_hot)
        if mask.sum().item() == 0:
            breakpoint()
        logits[~mask] = float("-inf")
        policy = F.softmax(logits, dim=1)
        value = self.dense_v(h_relu).view(-1)
        return logits, policy, value

    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        obs = torch.from_numpy(obs)
        _, pi, v = self.forward(obs)
        return pi.detach().numpy(), v.detach().numpy()

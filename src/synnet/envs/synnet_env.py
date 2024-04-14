import numpy as np
import gym
from functools import partial
from utils import cprint
from static_env import StaticEnv
from synnet.utils.predict_utils import mol_fp, tanimoto_similarity
from synnet.utils.data_utils import Reaction, ReactionSet, SyntheticTreeSet, Skeleton, SkeletonSet, Program
from synnet.config import DELIM
from synnet.models.mlp import nn_search_list
from synnet.models.gnn import PtrDataset
import os
from copy import deepcopy
import torch
from torch_geometric.data import Data
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import json
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3



def initial_state(sk, smiles):
    return mol_fp(smiles).tolist() + ([0] * 91) * sk.rxns.sum()


def get_obs_for_states(sk, states):
    return np.array(states)


def next_state(sk, state, action):    
    new_state = deepcopy(state)
    new_state[-sk.rxns.sum()*91+action] = 1    
    return new_state


def state_to_program(sk, state):
    sk.clear_tree()
    rxn_graph, node_map, reverse_node_map = sk.rxn_graph()
    for i in range(sk.rxns.sum()):
        rxn_id = np.argmax(state[-sk.rxns.sum()*91:][91*i:91*i+91])
        n = np.argwhere(sk.rxns).flatten()[i]
        sk.tree.nodes[n]['rxn_id'] = rxn_id
        sk.mask = [n]
        rxn_graph.nodes[node_map[n]]['rxn_id'] = int(rxn_id)
    return rxn_graph, node_map, reverse_node_map


def is_done_state(sk, state, policy): 
    if sum(state[-sk.rxns.sum()*91:]) == sk.rxns.sum():
        return True
    return policy.action_mask(np.array(state)).sum().item() == 0


def prepare_input(sk, model_bb):
    edges = sk.tree_edges
    tree_edges = np.concatenate((edges, edges[::-1]), axis=-1)
    edge_input = torch.tensor(tree_edges, dtype=torch.int64)      
    _, X, y = sk.get_state(rxn_target_down_bb=True, rxn_target_down=True)
    if model_bb.layers[0].in_channels != X.shape[1]:
        pe = PtrDataset.positionalencoding1d(32, len(X))
        x_input_bb = np.concatenate((X, pe), axis=-1)            
    else:
        x_input_bb = X        
    data_bb = Data(edge_index=edge_input, x=torch.Tensor(x_input_bb))
    return data_bb


def get_return(sk, state, model, hash_dir, rxns, bbs, bb_emb):
    """
    bbs: all bbs
    emb_bb: embedding of all bbs
    """    
    if sum(state[-sk.rxns.sum()*91:]) < 2:
        return 0.
    rxn_graph, node_map, reverse_node_map = state_to_program(sk, state)
    # run surrogate to get bbs
    model.eval()    
    data_bb = prepare_input(sk, model)
    logits_bb = model(data_bb)    

    # get program
    assert len(rxn_graph.edges) == 1
    src, dest = list(rxn_graph.edges)[0]
    rxn_id = rxn_graph.nodes[dest]['rxn_id']
    rxn_graph.nodes[dest]['rxn_id'] = -1
    parent_path = Program(rxn_graph).get_path()
    rxn_graph.nodes[dest]['rxn_id'] = rxn_id
    child_path = Program(rxn_graph).get_path()
    parent_path = os.path.join(hash_dir, parent_path)
    child_path = os.path.join(hash_dir, Path(parent_path).stem, child_path)

    parent_dic = json.load(open(parent_path))
    child_dic = json.load(open(child_path))
    lookup = {src: parent_dic, dest: child_dic}
        
    # reconstruct target
    for i in np.argwhere(sk.leaves).flatten():
        emb_bb = logits_bb[i][:256]
        parent = list(sk.tree.predecessors(i))[0]
        child = sk.tree.nodes[i]['child'] == 'right'
        term = node_map[parent]
        if 'bbs' not in lookup[term]:
            breakpoint()
        bb_lookup = lookup[term]['bbs']
        if str(term) in bb_lookup:
            key = str(term)       
            indices = [bbs.index(smi) for smi in bb_lookup[key][child]]             
        else:
            key = f"{str(term)}{DELIM}{int(child)}"
            assert key in bb_lookup
            assert len(bb_lookup[key]) == 1
            indices = [bbs.index(smi) for smi in bb_lookup[key][0]]
        
        bb_ind = nn_search_list(emb_bb, bb_emb[indices], top_k=1).item()
        smiles = bbs[indices[bb_ind]]
        sk.modify_tree(i, smiles=smiles)

    sk.reconstruct(rxns)
    
    interms = []
    for n in sk.tree:
        if 'smiles' in sk.tree.nodes[n]:
            interms.append(sk.tree.nodes[n]['smiles'])
    sims = tanimoto_similarity(state[:4096], interms)
    return max(sims)


def get_mask(sk, policy, state):
    return policy.action_mask(np.array(state))


# class SkeletonEnv(gym.Env):
#     """
#     Simple gym environment with the goal of populating a Skeleton object.
#     """

#     vis_dir = None
#     sk = None
#     model = None
    
#     def initial_state(self):
#         breakpoint()

#     def render(self):
#         self.sk.visualize(os.path.join(self.vis_dir, f"{self.step_idx}.png"))

def constructor(self, policy):
    self.policy = policy
    self.actions = []
    self.states = []  
    self.sk_progress = [] 


def step(self, action):
    self.actions.append(action)
    next_state = self.next_state(self.states[-1], action)    
    self.states.append(next_state)    
    done = self.is_done_state(next_state, self.policy)    
    n = np.argwhere(self.sk.rxns).flatten()[action//91]     
    self.sk.modify_tree(n, rxn_id=action%91)
    self.sk_progress.append(deepcopy(self.sk))
    if done:
        reward = self.get_return(next_state, self.model, self.hash_dir, self.rxns, self.bbs, self.bb_emb)
        self.sk_progress.append(deepcopy(self.sk))
    else:           
        reward = 0.    
    return next_state, reward, done, None


def reset(self, smiles):
    self.sk.clear_tree()
    self.sk.reset([self.sk.tree_root])
    self.sk.tree.nodes[self.sk.tree_root]['smiles'] = smiles
    init_state = initial_state(self.sk, smiles)
    self.states.append(init_state)
    self.sk_progress.append(deepcopy(self.sk))
    return init_state, 0., False, None


def render(self, path):
    fig = plt.Figure()
    n = len(self.sk_progress)
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        self.sk_progress[i].visualize(path, ax=ax)
    fig.savefig(path)
    print(os.path.abspath(path))



def make_skeleton_class(idx, vis_dir, sk, model, hash_dir, rxns, bbs, bb_emb):
    attrs = {
        'vis_dir': vis_dir,
        'sk': sk,
        'model': model,
        'n_actions': sk.rxns.sum()*91,
        'hash_dir': hash_dir,
        'rxns': rxns,
        'bbs': bbs,
        'bb_emb': bb_emb,
    }
    attrs.update({
        # static methods
        'initial_state': partial(initial_state, sk),
        'get_obs_for_states': partial(get_obs_for_states, sk),
        'next_state': partial(next_state, sk),
        'is_done_state': partial(is_done_state, sk),
        'get_return': partial(get_return, sk),
        'get_mask': partial(get_mask, sk),
        # constructor
        "__init__": constructor,
        # instance methods
        'step': step,
        'reset': reset,
        'render': render,
    })
    return type(f"SkeletonEnv_{idx}", (gym.Env,), attrs)


if __name__ == '__main__':  
    syntree_set_all = [st for v in skeletons.values() for st in v]
    index = 44
    sk = skeletons[list(skeletons)[index]][0]
    sk = Skeleton(sk, index)
    env = SkeletonEnv(sk, './vis/')
    env.render()
    for n in [20, 19, 17, 16, 15]:
        env.step((n, 0))
        env.render()

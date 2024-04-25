"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from synnet.data_generation.preprocessing import BuildingBlockFileHandler, ReactionTemplateFileHandler
from synnet.policy import RxnPolicy, Trainer, ReplayMemory, execute_episode
from synnet.envs.synnet_env import make_skeleton_class
from synnet.utils.data_utils import Skeleton, SkeletonSet, ReactionSet, Reaction
from synnet.utils.analysis_utils import reorder_syntrees
from synnet.models.common import load_gnn_from_ckpt
from synnet.config import DELIM
import torch
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import rdkit.Chem as Chem
random.seed(0)

import pickle

# for split in ['train','valid','test']:
#     skeletons = pickle.load(open(f'results/viz/top_1000/skeletons-top-1000-{split}.pkl', 'rb'))
#     for i, st in enumerate(skeletons):
#         pickle.dump(skeletons[st], open(f'results/viz/top_1000/class/{i}-{split}.pkl', 'wb+'))
#         print(i)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument("--log_path", default='./env_vis/')
    parser.add_argument("--hash-dir")
    parser.add_argument("--building_blocks_file")
    parser.add_argument("--rxn_templates_file")
    parser.add_argument("--rxns_collection_file")
    parser.add_argument("--embeddings_knn_file")
    parser.add_argument("--skeleton_class", default=[3], nargs='+', type=int)
    parser.add_argument("--surrogate")
    parser.add_argument("--split", choices=['train','valid','test'])
    parser.add_argument("--method", choices=['mcts', 'forward'], default='mcts')
    parser.add_argument("--test-iters", default=5, type=int)
    parser.add_argument("--num-simulations", default=20, type=int)
    parser.add_argument("--ncpu", default=50, type=int)
    parser.add_argument("--test_size", default=0.2, type=float)
    # Trainer
    parser.add_argument("--lr", default=0.1, type=float)
    return parser.parse_args()



def log(test_env, iteration, step_idx, total_rew):
    """
    Logs one step in a testing episode.
    :param test_env: Test environment that should be rendered.
    :param iteration: Number of training iterations so far.
    :param step_idx: Index of the step in the episode.
    :param total_rew: Total reward collected so far.
    """
    time.sleep(0.3)
    print()
    print(f"Training Episodes: {iteration}")
    test_env.render()
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")


def test_agent(iteration, sk, skeleton_class, network):  
    sk.reset([sk.tree_root])      
    test_env = skeleton_class
    total_rew = 0
    state, reward, done, _ = test_env.reset()
    step_idx = 0
    while not done:
        log(test_env, iteration, step_idx, total_rew)
        p, _ = network.step(np.array([state]))
        # print(p)
        action = np.argmax(p)
        state, reward, done, _ = test_env.step(action)
        step_idx+=1
        total_rew += reward
    log(test_env, iteration, step_idx, total_rew)    


def plot_rewards(rewards, path):
    def movingaverage(interval, window_size=10):
        window= np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')    

    fig = plt.Figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.plot(movingaverage(rewards))    
    ax.plot(rewards)
    fig.savefig(path)
    print(os.path.abspath(path))


def forward_search(st, i, index, test_env, p):
    sk = Skeleton(st, index)
    sk.reset([sk.tree_root])
    smiles = st.chemicals[-1].smiles # key for shared data        
    total_rew = 0
    state, reward, done, _ = test_env.reset(i, smiles)
    done = False
    step_idx = 0    
    while not done:
        # p, _ = network.step(np.array([state]))
        # print(p)            
        if not network.action_mask(np.array(state))[p[step_idx]].item(): 
            break
        action = p[step_idx]
        state, reward, done, _ = test_env.step(i, action)
        step_idx += 1
        total_rew += reward
    print(total_rew)
    # if total_rew > globals()['best_reward']:
    #     globals()['best_reward'] = total_rew
    #     print(total_rew)
    # folder = os.path.join(test_env.vis_dir, f"{iter}")   
    # os.makedirs(folder, exist_ok=True)
    # path = os.path.join(f"{folder}/{i}_eval.png")
    # test_env.render(path)    
    return total_rew



def evaluate(st, i, index, test_env):
    sk = Skeleton(st, index)
    sk.reset([sk.tree_root])
    total_rew = 0
    smiles = st.chemicals[-1].smiles # key for shared data
    state, reward, done, _ = test_env.reset(i, smiles)        
    done = False
    step_idx = 0
    while not done:
        p, _ = network.step(np.array([state]))
        # print(p)
        action = np.argmax(p)
        state, reward, done, _ = test_env.step(i, action)
        step_idx += 1
        total_rew += reward     
    # folder = os.path.join(test_env.vis_dir, f"{iter}")   
    # os.makedirs(folder, exist_ok=True)
    # path = os.path.join(f"{folder}/{i}_eval.png")
    # test_env.render(path)
    return total_rew


def mp_evaluate(args, sts, index, skeleton_class, network):
    def get_rxns(st):
        sk = Skeleton(st, -1)
        rxn_nodes = np.argwhere(sk.rxns).flatten()
        rxn_ids = [sk.tree.nodes[n]['rxn_id'] for n in rxn_nodes]
        for i in range(1, len(rxn_ids)):
            rxn_ids[i] += i*91        
        return rxn_ids[::-1]
        
    print(len(sts), "test sts")
    test_env = skeleton_class(network)
    globals()['network'] = network   
          
    # with mp.Pool(50) as p:
    #     total_rews = p.starmap(evaluate, tqdm([(st, i, index) for i, st in enumerate(sts)]))
    tasks = [(st, i, index, test_env) for i, st in tqdm(enumerate(sts))]
    if args.method == 'mcts':
        with mp.pool.ThreadPool(args.ncpu) as p:
            total_rews = p.starmap(evaluate, tqdm(tasks))
    else:
        globals()['best_reward'] = 0.
        total_rews = []
        # poss = []
        # for a1 in range(91, 182):
        #     for a2 in range(91):
        #         poss.append((a1, a2))         
        tasks = [(st, i, index, test_env, get_rxns(st)) for i, st in enumerate(sts)]
        if args.ncpu > 1:
            with mp.pool.ThreadPool(args.ncpu) as p:
                all_rews = p.starmap(forward_search, tqdm(tasks))
        else:
            all_rews = [forward_search(*task) for task in tasks]
        total_rews += [max(rews) for rews in all_rews]
    total_rews = list(total_rews)
    return total_rews

def mp_execute(args, sts, idx, skeleton_class, network, mems):
    num_simulations = args.num_simulations
    skeleton_class.network = network
    if args.ncpu == 1:
        res = [execute_episode(num_simulations, skeleton_class, i, st.chemicals[-1].smiles) for i, st in enumerate(sts)]
    else:        
        tasks = [(num_simulations, skeleton_class, i, st.chemicals[-1].smiles) for i, st in enumerate(sts)]
        with mp.pool.ThreadPool(args.ncpu) as p:
            res = p.starmap(execute_episode, tqdm(tasks))
    for r in res:
        obs, pis, returns, total_reward, done_state = r
        mems[idx].add_all({"ob": obs, "pi": pis, "return": returns})        



def main(args):
    trainers = {}
    networks = {}
    mems = {}
    bbs = BuildingBlockFileHandler().load(args.building_blocks_file)
    bb_emb = np.load(args.embeddings_knn_file)
    bb_emb = torch.as_tensor(bb_emb, dtype=torch.float32)    
    rxns = [Reaction(smirks) for smirks in ReactionTemplateFileHandler().load(args.rxn_templates_file)]
    for idx in args.skeleton_class:
        split = args.split
        sts = pickle.load(open(f'results/viz/top_1000/class/{idx}-{split}.pkl', 'rb'))
        # Train-test split

        # test
        # for st in tqdm(sts_test, desc="testing"):
        #     sk_test = Skeleton(st, idx)
        #     sk_test.visualize(f'results/viz/top_1000/class/debug/{idx}_gold.png')
        #     gold = sk_test.tree.nodes[sk_test.tree_root]['smiles']
        #     sk_test.reconstruct(rxns)
        #     sk_test.visualize(f'results/viz/top_1000/class/debug/{idx}.png')
        #     pred = sk_test.tree.nodes[sk_test.tree_root]['smiles']
        #     gold = Chem.CanonSmiles(gold)
        #     preds = [Chem.CanonSmiles(p) for p in pred.split(DELIM)]
        #     if gold not in preds:
        #         breakpoint()
 
        sk = Skeleton(sts[0], idx)        
        trainer = Trainer(lambda: RxnPolicy(4096+sk.rxns.sum()*91, 20, sk.rxns.sum()*91, sk, args.hash_dir, rxns), learning_rate=args.lr)
        trainers[idx] = trainer
        networks[idx] = trainer.step_model        
        mems[idx] = ReplayMemory(200,
                       { "ob": np.uint8,
                         "pi": np.float32,
                         "return": np.float32},
                       { "ob": [4096+sk.rxns.sum()*91],
                         "pi": [sk.rxns.sum()*91],
                         "return": []})
        model = load_gnn_from_ckpt(args.surrogate)
        vis_dir = os.path.join(args.log_path, f"{idx}/")
        os.makedirs(vis_dir, exist_ok=True)        
        skeleton_class = make_skeleton_class(idx, vis_dir, sk, model, args.hash_dir, rxns, bbs, bb_emb)
        value_losses = []
        policy_losses = []
        reward_history = []
        for iter in range(1000):
            if iter % args.test_iters == 0:
                fig = plt.Figure()
                ax = fig.add_subplot(1,1,1)
                ax.plot(value_losses, label="value loss")
                ax.plot(policy_losses, label="policy loss")
                ax.legend()
                fig.savefig(os.path.join(args.log_path, f'{iter}_loss.png'))
                print(os.path.abspath(os.path.join(args.log_path, f'{iter}_loss.png')))
                total_rewards = mp_evaluate(args, sts, idx, skeleton_class, networks[idx])
                reward_history.append(np.mean(total_rewards))
                plot_rewards(reward_history, os.path.join(args.log_path, f'{iter}_rewards.png'))
                print(os.path.abspath(os.path.join(args.log_path, f'{iter}_rewards.png')))

            mp_execute(args, sts, idx, skeleton_class, networks[idx], mems)                
            batch = mems[idx].get_minibatch()
            vl, pl = trainers[idx].train(batch["ob"], batch["pi"], batch["return"])            
            value_losses.append(vl)
            policy_losses.append(pl)



if __name__ == '__main__':    
    args = get_args()
    main(args)
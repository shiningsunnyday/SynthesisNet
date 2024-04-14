"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.policy import RxnPolicy, Trainer, ReplayMemory, execute_episode
from synnet.envs.synnet_env import SkeletonEnv, make_skeleton_class
from synnet.utils.data_utils import Skeleton, SkeletonSet, ReactionSet
from synnet.models.common import load_gnn_from_ckpt
import torch
import random
random.seed(0)

import pickle

# skeletons = pickle.load(open('results/viz/top_1000/skeletons-top-1000.pkl', 'rb'))
# for i, st in enumerate(skeletons):
#     pickle.dump(skeletons[st], open(f'results/viz/top_1000/class/{i}.pkl', 'wb+'))
#     print(i)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument("--log_path", default='./env_vis/')
    parser.add_argument("--hash-dir")
    parser.add_argument("--building_blocks_file")
    parser.add_argument("--rxns_collection_file")
    parser.add_argument("--embeddings_knn_file")
    parser.add_argument("--skeleton_class", default=[3], nargs='+', type=int)
    parser.add_argument("--surrogate")
    parser.add_argument("--test-iters", default=1)
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
    sk.reset()      
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
    ax.plot(movingaverage(rewards))    
    fig.savefig(path)
    print(os.path.abspath(path))



def main(args):
    trainers = {}
    networks = {}
    mems = {}
    bbs = BuildingBlockFileHandler().load(args.building_blocks_file)
    bb_emb = np.load(args.embeddings_knn_file)
    bb_emb = bb_emb/np.linalg.norm(bb_emb, axis=-1, keepdims=True)
    bb_emb = torch.as_tensor(bb_emb, dtype=torch.float32)    
    rxns = ReactionSet().load(args.rxns_collection_file).rxns
    for idx in args.skeleton_class:
        sts = pickle.load(open(f'results/viz/top_1000/class/{idx}.pkl', 'rb'))
        random.shuffle(sts)
        sk = Skeleton(sts[0], idx)        
        trainer = Trainer(lambda: RxnPolicy(4096+sk.rxns.sum()*91, 20, sk.rxns.sum()*91, sk, args.hash_dir, rxns))
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
        skeleton_class = make_skeleton_class(idx, vis_dir, sk, model, rxns, bbs, bb_emb)
        value_losses = []
        policy_losses = []        
        total_rewards = []
        for i, st in enumerate(sts):
            # if i % args.test_iters == 0:
            #     sk = Skeleton(st, i)
            #     test_agent(i, sk, skeleton_class, networks[idx])
            #     fig = plt.Figure()
            #     ax = fig.add_subplot(1,1,1)
            #     ax.plot(value_losses, label="value loss")
            #     ax.plot(policy_losses, label="policy loss")
            #     ax.legend()
            #     fig.savefig(os.path.join(args.log_path, f'{i}.png'))
            #     print(os.path.join(args.log_path, f'{i}.png'))
                

            obs, pis, returns, total_reward, done_state = execute_episode(networks[idx], 20, skeleton_class, smiles=st.chemicals[-1].smiles)
            mems[idx].add_all({"ob": obs, "pi": pis, "return": returns})
            total_rewards.append(total_reward)
            batch = mems[idx].get_minibatch()
            vl, pl = trainers[idx].train(batch["ob"], batch["pi"], batch["return"])            
            value_losses.append(vl)
            policy_losses.append(pl)
            plot_rewards(total_rewards, os.path.join(args.log_path, f'{i}.png'))



if __name__ == '__main__':    
    args = get_args()
    breakpoint()
    main(args)
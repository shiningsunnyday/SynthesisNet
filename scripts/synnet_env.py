import numpy as np
import gym

from utils import cprint
from static_env import StaticEnv
import os


from synnet.utils.data_utils import Skeleton, SkeletonSet
import pickle
skeletons = pickle.load(open('results/viz/top_1000/skeletons-top-1000-train.pkl', 'rb'))
skeleton_set = SkeletonSet().load_skeletons(skeletons)  

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class SkeletonEnv(gym.Env):
    """
    Simple gym environment with the goal of populating a Skeleton object.
    """

    def __init__(self, sk, vis_dir):        
        self.sk = sk
        self.vis_dir = vis_dir
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.sk.reset()
        self.sk.mask = [self.sk.tree_root]
        state = self.sk.mask
        return state, 0, False, None

    def step(self, action):
        self.step_idx += 1
        n, rxn_id = action        
        self.sk.modify_tree(n, rxn_id=rxn_id)
        if self.sk.mask.all():
            done = True
            breakpoint()
        else:
            done = False
            reward = 0.
        state = self.sk.mask
        return state, reward, done, None

    def render(self):
        self.sk.visualize(os.path.join(self.vis_dir, f"{self.step_idx}.png"))


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

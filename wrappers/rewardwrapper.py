import numpy as np
from typing import List, Tuple
from gymnasium import Env, spaces
import gymnasium as gym
from pokegym import ram_map

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.time = env.time
        self.env_name = env.env_name
        self.seen_maps = set()
        self.tasks_done = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, done, info = self.env.step(action)
        _, _, z = ram_map.position()
        self.seen_maps.add((z))

        self.seen_coords = self.env.seen_coords
        return obs, reward, done, done, info
    
    def reward(self, task):
        reward = 0
        return reward

    def explore(self):
        exploration = len(self.seen_coords)
        return exploration
    
    def pewter(self):
        reward = 0
        return reward

    def cerulean(self):
        reward = 0
        return reward
    
    def vermilion(self):
        reward = 0
        return reward
    
    def celadon(self):
        reward = 0
        return reward

    def fuchsia(self):
        reward = 0
        return reward

    def saffron(self):
        reward = 0
        return reward
    
    def cinnabar(self):
        reward = 0
        return reward
    
    def viridian(self):
        reward = 0
        return reward
    
    def indigo(self):
        reward = 0
        return reward

    def mt_moon(self):
        reward = 0
        return reward
    
    def ss_anne(self):
        reward = 0
        return reward
    
    def 
    
    def victory_road(self):
        reward = 0
        return reward








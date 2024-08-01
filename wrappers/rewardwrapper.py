import numpy as np
from typing import List, Tuple
from gymnasium import Env, spaces
import gymnasium as gym

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.time = env.time
        self.env_name = env.env_name

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, done, info = self.env.step(action)

        return obs, reward, done, done, info
    
    def pick_task(self, task):
        if task == 'task1':
            return self.reward

        return reward

    def task1(self):
        return self.reward








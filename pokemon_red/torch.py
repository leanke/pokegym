from functools import partial
from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces
import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)
    
class Policy(nn.Module):
    def __init__(self, env, *args, framestack=4, flat_size=64*5*6, input_size=512, hidden_size=512, output_size=512, channels_last=True, downsample=1, **kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)
        self.extra_obs = env.extra_obs

        self.screen= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            # pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            # nn.ReLU(),
        )
        self.embedding = torch.nn.Embedding(250, 4, dtype=torch.float32)
        self.linear= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),)

    def encode_observations(self, observations):
        observation = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        screens = torch.cat([
            observation['screen'], 
            observation['fixed_window'],
            ], dim=-1)
        if self.extra_obs:
            cat = torch.cat(
            (
                self.screen(screen.float() / 255.0).squeeze(1),
                self.embedding(observations["map_n"].long()).squeeze(1),
                observations["flute"].float(),
                observations["bike"].float(),
                observations["hideout"].float(),
                observations["tower"].float(),
                observations["silphco"].float(),
                observations["snorlax_12"].float(),
                observations["snorlax_16"].float(),
            ),
            dim=-1,
        )
        else:
            cat = self.screen(screen.float() / 255.0),

        
        if self.channels_last:
            screen = screens.permute(0, 3, 1, 2)
        if self.downsample > 1:
            screen = screens[:, :, ::self.downsample, ::self.downsample]
        return self.linear(cat), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
    
    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value
    
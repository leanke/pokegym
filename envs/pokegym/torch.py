from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces


class Policy(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.encoder = nn.Linear(1941, self.hidden_size)
        self.decoder = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        self.screen= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(2, 32, 8, stride=4)), # 2 channels for screen and fixed_window
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.map_embedding = torch.nn.Embedding(248, 4, dtype=torch.float32)
        self.event_fc = nn.Sequential(pufferlib.pytorch.layer_init(nn.Linear(16, 16)),nn.ReLU(),)
        self.position_fc = nn.Sequential(pufferlib.pytorch.layer_init(nn.Linear(7, 4)),nn.ReLU(),)





    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        observation = pufferlib.pytorch.nativize_tensor(observations, self.dtype)



        screens = torch.cat([observation['screen'], observation['fixed_window'],], dim=-1)
        screen = screens.permute(0, 3, 1, 2)
        cnn = self.screen(screen.float() / 255.0) # screen and fixed_window stacked
        map = self.map_embedding(observation["map_n"].long()).squeeze(1) # map_id embedding
        x = observation["x"].float() / 255.0
        y = observation["y"].float() / 255.0
        direction = observation["direction"].float() / 4.0
        pos_cat = torch.cat((map, x, y, direction), dim=-1) # map embedding, x, y, direction
        pos = self.position_fc(pos_cat) # position fc
        event = self.event_fc(observation["events"].float()) # events: badges and bike, hideout, tower, silphco, snorlax_12, snorlax_16, got_flute
        full_cat = torch.cat((cnn, event, pos, observation["in_battle"].float()), dim=-1) # final cat also includes in_battle





        return self.encoder(full_cat)

    def decode_actions(self, hidden):
        logits = self.decoder(hidden)
        values = self.value(hidden)
        return logits, values
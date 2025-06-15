from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces
import pufferlib.models

class Recurrent(nn.Module):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if 'layer_norm' in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        #self.pre_layernorm = nn.LayerNorm(hidden_size)
        #self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward_eval(self, observations, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        hidden = self.policy.encode_observations(observations, state=state)
        h = state['lstm_h']
        c = state['lstm_c']

        # TODO: Don't break compile
        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0], 'LSTM state must be (h, c)'
            lstm_state = (h, c)
        else:
            lstm_state = None

        #hidden = self.pre_layernorm(hidden)
        hidden, c = self.cell(hidden, lstm_state)
        #hidden = self.post_layernorm(hidden)
        state['hidden'] = hidden
        state['lstm_h'] = hidden
        state['lstm_c'] = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x = observations
        lstm_h = state['lstm_h']
        lstm_c = state['lstm_c']

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if lstm_h is not None:
            assert lstm_h.shape[1] == lstm_c.shape[1] == B, 'LSTM state must be (h, c)'
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        x = x.reshape(B*TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)

        hidden = hidden.transpose(0, 1)
        #hidden = self.pre_layernorm(hidden)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        #hidden = self.post_layernorm(hidden)
        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(B*TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        #state.batch_logits = logits.reshape(B, TT, -1)
        state['hidden'] = hidden
        state['lstm_h'] = lstm_h.detach()
        state['lstm_c'] = lstm_c.detach()
        return logits, values


class Policy(nn.Module):
    def __init__(self, env, hidden_size=512):
        super().__init__()
        self.is_continuous = False
        self.hidden_size = hidden_size
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.encoder = nn.Linear(1940, self.hidden_size)
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

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

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
        full_cat = torch.cat((cnn, event, pos, ), dim=-1) # final cat also includes in_battle observation["in_battle"].float()


        return self.encoder(full_cat)

    def decode_actions(self, hidden):
        logits = self.decoder(hidden)
        values = self.value(hidden)
        return logits, values

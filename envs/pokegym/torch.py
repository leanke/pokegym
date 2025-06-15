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
        self.encoder = nn.Linear(2081, self.hidden_size)
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

        self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
        self.move_fc_relu = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)
        self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)
        self.poke_fc_relu = nn.Sequential(
            nn.Linear(63, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.poke_party_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))
        self.poke_opp_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))
        self.item_ids_embedding = nn.Embedding(256, 16, padding_idx=0)  # (20, 16)
        self.item_ids_fc_relu = nn.Sequential(
            nn.Linear(17, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        self.event_ids_embedding = nn.Embedding(2570, 16, padding_idx=0)  # (20, )
        self.event_ids_fc_relu = nn.Sequential(
            nn.Linear(17, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        self._features_dim = 406
        self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
        self.move_fc_relu = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)
        self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)
        self.poke_fc_relu = nn.Sequential(
            nn.Linear(63, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.poke_party_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))
        self.poke_opp_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))
        self.item_ids_embedding = nn.Embedding(256, 16, padding_idx=0)  # (20, 16)
        self.item_ids_fc_relu = nn.Sequential(
            nn.Linear(17, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        self.event_ids_embedding = nn.Embedding(2570, 16, padding_idx=0)  # (20, )
        self.event_ids_fc_relu = nn.Sequential(
            nn.Linear(17, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        self._features_dim = 406

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
        embedded_poke_move_ids = self.poke_move_ids_embedding(observation['poke_move_ids'].to(torch.int))
        poke_move_pps = observation['poke_move_pps']
        poke_moves = torch.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
        poke_moves = self.move_fc_relu(poke_moves)
        poke_moves = self.move_max_pool(poke_moves).squeeze(-2)
        embedded_poke_type_ids = self.poke_type_ids_embedding(observation['poke_type_ids'].to(torch.int))
        poke_types = torch.sum(embedded_poke_type_ids, dim=-2)
        embedded_poke_ids = self.poke_ids_embedding(observation['poke_ids'].to(torch.int))
        poke_ids = embedded_poke_ids
        poke_stats = observation['poke_all']
        pokemon_concat = torch.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)
        pokemon_features = self.poke_fc_relu(pokemon_concat)
        party_pokemon_features = pokemon_features[..., :6, :]
        poke_party_head = self.poke_party_head(party_pokemon_features)
        poke_party_head = self.poke_party_head_max_pool(poke_party_head).squeeze(-2)
        opp_pokemon_features = pokemon_features[..., 6:, :]
        poke_opp_head = self.poke_opp_head(opp_pokemon_features)
        poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)
        embedded_item_ids = self.item_ids_embedding(observation['item_ids'].to(torch.int))
        item_quantity = observation['item_quantity']
        item_concat = torch.cat([embedded_item_ids, item_quantity], dim=-1)
        item_features = self.item_ids_fc_relu(item_concat)
        item_features = self.item_ids_max_pool(item_features).squeeze(-2)
        embedded_event_ids = self.event_ids_embedding(observation['event_ids'].to(torch.int))
        event_step_since = observation['event_step_since']
        event_concat = torch.cat([embedded_event_ids, event_step_since], dim=-1)
        event_features = self.event_ids_fc_relu(event_concat)
        event_features = self.event_ids_max_pool(event_features).squeeze(-2)
        vector = observation['vector']
    # 'bike': Box(0, 1, (1,), uint8), 
    # 'event_ids': Box(0, 2570, (10,), uint32), 
    # 'event_step_since': Box(-1.0, 1.0, (10, 1), float32), 
    # 'fixed_window': Box(0, 255, (72, 80, 1), uint8), 
    # 'flute': Box(0, 1, (1,), uint8), 
    # 'hideout': Box(0, 1, (1,), uint8), 
    # 'item_ids': Box(0, 255, (20,), uint8), 
    # 'item_quantity': Box(-1.0, 1.0, (20, 1), float32), 
    # 'map_ids': Box(0, 255, (1,), uint8), 
    # 'map_n': Box(0, 250, (1,), uint8), 
    # 'poke_all': Box(-1.0, 1.0, (12, 23), float32), 
    # 'poke_ids': Box(0, 255, (12,), uint8), 
    # 'poke_move_ids': Box(0, 255, (12, 4), uint8), 
    # 'poke_move_pps': Box(-1.0, 1.0, (12, 4, 2), float32), 
    # 'poke_type_ids': Box(0, 255, (12, 2), uint8), 
    # 'screen': Box(0, 255, (72, 80, 1), uint8), 
    # 'silphco': Box(0, 1, (1,), uint8), 
    # 'snorlax_12': Box(0, 1, (1,), uint8), 
    # 'snorlax_16': Box(0, 1, (1,), uint8), 
    # 'tower': Box(0, 1, (1,), uint8), 
    # 'vector': Box(-1.0, 1.0, (54,), float32)
        cat = torch.cat((
            self.screen(screen.float() / 255.0),
            self.map_embedding(observation["map_n"].long()).squeeze(1),
            observation["flute"].float(),
            observation["bike"].float(),
            observation["hideout"].float(),
            observation["tower"].float(),
            observation["silphco"].float(),
            observation["snorlax_12"].float(),
            observation["snorlax_16"].float(),
            poke_party_head, 
            poke_opp_head, 
            item_features, 
            event_features, 
            vector
        ),dim=-1,)

        return self.encoder(cat)

    def decode_actions(self, hidden):
        logits = self.decoder(hidden)
        values = self.value(hidden)
        return logits, values
    

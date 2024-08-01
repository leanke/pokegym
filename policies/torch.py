from pdb import set_trace as T

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces
import pufferlib.models
from pokegym import data
# pufferlib.pytorch.nativize_tensor = torch.compiler.disable(pufferlib.pytorch.nativize_tensor)


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)
    
class Policy(nn.Module):
    def __init__(self, env, *args, framestack=4, flat_size=64*5*6, input_size=512, hidden_size=512, output_size=512, channels_last=True, downsample=1, **kwargs): #64*6*6+90
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.flat_size = flat_size
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)
        self.extra_obs = env.unwrapped.env.extra_obs # env.unwrapped is GymnasiumPufferEnv
        if self.extra_obs:
            self.flat_size = self.flat_size + 11
        self.add_boey_obs = env.unwrapped.env.add_boey_obs
        if self.add_boey_obs:
            self.boey_nets()
            self.flat_size = self.flat_size + 150

        self.screen= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.embedding = torch.nn.Embedding(250, 4, dtype=torch.float32) # 6? or 4?
        
        self.linear= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.flat_size, hidden_size)),
            nn.ReLU(),)

    def encode_observations(self, observations):
        observation = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        screens = torch.cat([
            observation['screen'], 
            observation['fixed_window'],
            ], dim=-1)
        
        if self.channels_last:
            screen = screens.permute(0, 3, 1, 2)
        if self.downsample > 1:
            screen = screens[:, :, ::self.downsample, ::self.downsample]

        if self.extra_obs:
            cat = torch.cat(
            (
                self.screen(screen.float() / 255.0).squeeze(1),
                self.embedding(observation["map_n"].long()).squeeze(1),
                observation["flute"].float(),
                observation["bike"].float(),
                observation["hideout"].float(),
                observation["tower"].float(),
                observation["silphco"].float(),
                observation["snorlax_12"].float(),
                observation["snorlax_16"].float(),
            ),
            dim=-1,
        )
        else:
            cat = self.screen(screen.float() / 255.0),

        if self.add_boey_obs:
                boey_obs = self.boey_obs(observation)
                cat = torch.cat([cat, boey_obs], dim=-1)

        return self.linear(cat), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
    
    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value
    
    def boey_obs(self, observation):
        if self.add_boey_obs:
            # Pokemon
            # Moves
            embedded_poke_move_ids = self.poke_move_ids_embedding(observation['poke_move_ids'].to(torch.int))
            poke_move_pps = observation['poke_move_pps']
            poke_moves = torch.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
            poke_moves = self.move_fc_relu(poke_moves)
            poke_moves = self.move_max_pool(poke_moves).squeeze(-2)  # (12, 16)
            # Types
            embedded_poke_type_ids = self.poke_type_ids_embedding(observation['poke_type_ids'].to(torch.int))
            poke_types = torch.sum(embedded_poke_type_ids, dim=-2)  # (12, 8)
            # Pokemon ID
            embedded_poke_ids = self.poke_ids_embedding(observation['poke_ids'].to(torch.int))
            poke_ids = embedded_poke_ids  # (12, 8)
            # Pokemon stats (12, 23)
            poke_stats = observation['poke_all']
            # All pokemon features
            pokemon_concat = torch.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)  # (12, 63)
            pokemon_features = self.poke_fc_relu(pokemon_concat)  # (12, 32)

            # Pokemon party head
            party_pokemon_features = pokemon_features[..., :6, :]  # (6, 32), ... for batch dim
            poke_party_head = self.poke_party_head(party_pokemon_features)  # (6, 32)
            poke_party_head = self.poke_party_head_max_pool(poke_party_head).squeeze(-2)  # (6, 32) -> (32, )

            # Pokemon opp head
            opp_pokemon_features = pokemon_features[..., 6:, :]  # (6, 32), ... for batch dim
            poke_opp_head = self.poke_opp_head(opp_pokemon_features)  # (6, 32)
            poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)  # (6, 32) -> (32, )

            # Items
            embedded_item_ids = self.item_ids_embedding(observation['item_ids'].to(torch.int))  # (20, 16)
            # item_quantity
            item_quantity = observation['item_quantity']  # (20, 1)
            item_concat = torch.cat([embedded_item_ids, item_quantity], dim=-1)  # (20, 17)
            item_features = self.item_ids_fc_relu(item_concat)  # (20, 16)
            item_features = self.item_ids_max_pool(item_features).squeeze(-2)  # (20, 16) -> (16, )

            # Events
            embedded_event_ids = self.event_ids_embedding(observation['event_ids'].to(torch.int))
            # event_step_since
            event_step_since = observation['event_step_since']  # (20, 1)
            event_concat = torch.cat([embedded_event_ids, event_step_since], dim=-1)  # (20, 17)
            event_features = self.event_ids_fc_relu(event_concat)
            event_features = self.event_ids_max_pool(event_features).squeeze(-2)  # (20, 16) -> (16, )

            # Raw vector
            vector = observation['vector']  # (54, )

            # Concat all features
            all_features = torch.cat([poke_party_head, poke_opp_head, item_features, event_features, vector], dim=-1)  # (406, )

        return all_features
    
    def boey_nets(self):
        # observation_space.spaces.items()

        # poke_move_ids (12, 4) -> (12, 4, 8)
        self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
        # concat with poke_move_pps (12, 4, 2)
        # input (12, 4, 10) for fc relu
        self.move_fc_relu = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        # max pool
        self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        # output (12, 1, 16), sqeeze(-2) -> (12, 16)

        # poke_type_ids (12, 2) -> (12, 2, 8)
        self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)
        # (12, 2, 8) -> (12, 8) by sum(dim=-2)

        # poke_ids (12, ) -> (12, 8)
        self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)
        
        # pokemon fc relu
        self.poke_fc_relu = nn.Sequential(
            nn.Linear(63, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # pokemon party head
        self.poke_party_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        # get the first 6 pokemon and do max pool
        self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

        # pokemon opp head
        self.poke_opp_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        # get the last 6 pokemon and do max pool
        self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

        # item_ids embedding
        self.item_ids_embedding = nn.Embedding(256, 16, padding_idx=0)  # (20, 16)
        # item_ids fc relu
        self.item_ids_fc_relu = nn.Sequential(
            nn.Linear(17, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        # item_ids max pool
        self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))

        # event_ids embedding
        self.event_ids_embedding = nn.Embedding(2570, 16, padding_idx=0)  # (20, )
        # event_ids fc relu
        self.event_ids_fc_relu = nn.Sequential(
            nn.Linear(17, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        # event_ids max pool
        self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))


        self._features_dim = 406
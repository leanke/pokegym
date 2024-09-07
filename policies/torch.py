import json
import os
from pathlib import Path
from pdb import set_trace as T
import uuid

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces
import pufferlib.models
from pokegym.data import poke_and_type_dict, map_dict

# torch._dynamo.config.capture_scalar_outputs = True
UNIQ_RUN = Path(f'{str(uuid.uuid4())[:4]}')

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)
        
    def get_embeds(self):
        return self.policy.get_embeds()
    
    def get_activations(self, observations):
        return self.policy.get_activations(observations)
    
    def plot_activations(self, activations):
        return self.policy.plot_activations(activations)

class Policy(nn.Module):
    def __init__(self, env, *args, framestack=4, flat_size=64*5*6, input_size=512, hidden_size=512, output_size=512, channels_last=True, downsample=1, **kwargs): #64*6*6+90
        super().__init__()
        self.save_table = True
        self.channels_last = channels_last
        self.downsample = downsample
        self.flat_size = flat_size
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)
        self.extra_obs = env.unwrapped.env.extra_obs # env.unwrapped is GymnasiumPufferEnv
        if self.extra_obs:
            self.flat_size = self.flat_size + 11 #+ 144
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
        self.single_screen= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )
        self.map_embedding = torch.nn.Embedding(248, 4, dtype=torch.float32)
        self.poke_id = nn.Embedding(190, 6, dtype=torch.float32)
        self.poke_type = nn.Embedding(15, 6, dtype=torch.float32)
        self.pokemon_embedding = nn.Linear(in_features=38, out_features=16) # input: id, status, type1, type2, stats_level # 8+8+8+8+6 # output: 16?
        self.activations = []
        self.linear= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.flat_size, hidden_size)),
            nn.ReLU(),)
        self.counter = 0

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
                self.map_embedding(observation["map_n"].long()).squeeze(1),
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
        self.counter += 1
        
        return self.linear(cat), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
    
    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value
    
    def pokemon_observation(self, observation):
        poke_obs_cat_list = []
        for i in range(6):
            ppoke = observation[f'ppoke{i+1}'].long()
            ptype = observation[f'ptype{i+1}'].long()
            opoke = observation[f'opoke{i+1}'].long()
            otype = observation[f'otype{i+1}'].long()
            ppoke_embed = self.poke_id(ppoke).squeeze(1)
            ptype_embed = self.poke_type(ptype).squeeze(1)
            opoke_embed = self.poke_id(opoke).squeeze(1)
            otype_embed = self.poke_type(otype).squeeze(1)
            poke_obs_cat_list.append(torch.cat([ppoke_embed, ptype_embed, opoke_embed, otype_embed], dim=-1))
        return torch.cat(poke_obs_cat_list, dim=-1)

    def get_embeds(self):
        poke_ids = [v['name'] for v in poke_and_type_dict.values()]
        type_id = ['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Fire', 'Water', 'Grass', 'Electric', 'psycic', 'ice', 'dragon']
        map_ids = [v for v in map_dict.values()]
        id_embeddings = self.poke_id.weight
        type_embeddings = self.poke_type.weight
        map_embeddings = self.map_embedding.weight

        id_list = map_ids # poke_ids + type_id
        shit = map_embeddings # torch.cat([id_embeddings, type_embeddings], dim=0)
        embed_list = shit.tolist()
        return id_list, embed_list
    
    def get_activations(self, observations):
        # x = observations['screen']
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        self.activations = []
        def hook_fn(module, input, output):
            self.activations.append(output)
        hooks = []
        for layer in self.single_screen:
            hooks.append(layer.register_forward_hook(hook_fn))
            input_tensor = observations['screen'].permute(0, 3, 1, 2)
            # input_tensor = input_tensor[2:3]
        _ = self.single_screen(input_tensor.float())
        for hook in hooks:
            hook.remove()
        return self.activations

    def plot_activations(self, activations, path):
        layer_counter = 0
        for i, activation in enumerate(activations):
            num_filters = activation.shape[1]
            grid_size = int(np.ceil(np.sqrt(num_filters)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
            axes = axes.flatten()
            for filter_idx in range(num_filters):
                axes[filter_idx].imshow(activation[0, filter_idx].detach().cpu().numpy(), cmap='viridis')
                axes[filter_idx].axis('off') 

            for filter_idx in range(num_filters, grid_size * grid_size):
                axes[filter_idx].axis('off')
            layer_counter += 1
            dpi = 75 # ~ Gameboy dpi
            folder = f'{path}/activations'
            if not os.path.exists(folder):
                os.makedirs(folder)
            plot_path = f'{folder}/step{self.counter}_layer{layer_counter}'
            # plt.show()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()

    def boey_obs(self, observation):
        if self.add_boey_obs:
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

            all_features = torch.cat([poke_party_head, poke_opp_head, item_features, event_features, vector], dim=-1)

        return all_features
    
    def boey_nets(self):
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
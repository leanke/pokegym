import json
import os
from pathlib import Path
import uuid
from matplotlib import pyplot as plt
import numpy as np
from pdb import set_trace as T

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces
import pufferlib.models
from pokegym.data import poke_and_type_dict, map_dict
# torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True

# import logging
# import torch._dynamo.config as dcfg
# dcfg.verbose=True
# import torch._functorch.config as fcfg
# import torch._inductor.config as icfg


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
    def __init__(self, env, *args, framestack=2, flat_size=64*5*6 + 25 + 192, input_size=512, hidden_size=512, output_size=512, channels_last=True, downsample=1, **kwargs): #64*6*6+90
        super().__init__()
        self.save_table = True
        self.channels_last = channels_last
        self.downsample = downsample
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

        self.screen= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.map_embedding = torch.nn.Embedding(248, 4, dtype=torch.float32)
        self.event_fc = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(16, 16)),
            nn.ReLU(),
         )
        self.position_fc = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(7, 4)),
            nn.ReLU(),
        )
        self.poke_id = nn.Embedding(192, 4, dtype=torch.float32)
        self.poke_type = nn.Embedding(15, 4, dtype=torch.float32)
        self.move = nn.Embedding(166, 4, dtype=torch.float32)
        self.status = nn.Embedding(7, 4, dtype=torch.float32)
        self.stat_fc = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(4, 4)),
            nn.ReLU(),
        )
        self.pokemon_fc = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(38, 32)),
            nn.ReLU(),
        )
        self.party_fc = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(192, 192)),
            nn.ReLU(),
        )
        self.enc_lin = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
         )

    def encode_observations(self, observations):
        observation = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        pokemon = observation["pokemon"].contiguous() # observation["pokemon"] is (32,6,17) (batch, mon, features)
        mon_data = []
        for i in range(6):
            id_embed = self.poke_id(pokemon[:, i, 0].unsqueeze(-1).long()).squeeze(1)
            poke_type_1 = self.poke_type(pokemon[:, i, 1].unsqueeze(-1).long()).squeeze(1)
            hp = pokemon[:, i, 2].unsqueeze(-1).float() / 100.0
            status_embed = self.status(pokemon[:, i, 3].unsqueeze(-1).long()).squeeze(1)
            move_embed_1 = self.move(pokemon[:, i, 4].unsqueeze(-1).long())
            move_embed_2 = self.move(pokemon[:, i, 5].unsqueeze(-1).long())
            move_embed_3 = self.move(pokemon[:, i, 6].unsqueeze(-1).long())
            move_embed_4 = self.move(pokemon[:, i, 7].unsqueeze(-1).long())
            level = pokemon[:, i, 12].unsqueeze(-1).float() / 100.0
            moves = torch.cat([move_embed_1, move_embed_2, move_embed_3, move_embed_4], dim=-1)
            move_pp = torch.cat([pokemon[:, i, 8].unsqueeze(-1).float(), 
                                 pokemon[:, i, 9].unsqueeze(-1).float(), 
                                 pokemon[:, i, 10].unsqueeze(-1).float(), 
                                 pokemon[:, i, 11].unsqueeze(-1).float()], dim=-1)
            pp_norm = move_pp / 100.0
            stats = torch.cat([pokemon[:, i, 13].unsqueeze(-1).float(), 
                               pokemon[:, i, 14].unsqueeze(-1).float(), 
                               pokemon[:, i, 15].unsqueeze(-1).float(), 
                               pokemon[:, i, 16].unsqueeze(-1).float()], dim=-1)
            stats_out = self.stat_fc(stats.float() / 716.0)
            mon = torch.cat([id_embed, poke_type_1, moves.squeeze(1), level, status_embed, hp, pp_norm, stats_out], dim=-1)
            mon_out = self.pokemon_fc(mon) # 38 per mon?
            mon_data.append(mon_out)
        mon_cat = torch.cat(mon_data, dim=-1)
        party_out = self.party_fc(mon_cat)


        screens = torch.cat([observation['screen'], observation['fixed_window'],], dim=-1)
        screen = screens.permute(0, 3, 1, 2)
        cnn = self.screen(screen.float() / 255.0)
        map = self.map_embedding(observation["map_n"].long()).squeeze(1)
        x = observation["x"].float() / 255.0
        y = observation["y"].float() / 255.0
        direction = observation["direction"].float() / 4.0
        pos_cat = torch.cat((map, x, y, direction), dim=-1)
        pos = self.position_fc(pos_cat)
        event = self.event_fc(observation["events"].float())
        full_cat = torch.cat((cnn, map, event, pos, party_out, observation["in_battle"].float()), dim=-1)
        final_out = self.enc_lin(full_cat)

        return final_out, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
    
    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value


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
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        x = torch.cat([observations['screen'], observations['fixed_window'],], dim=-1)
        self.activations = []
        def hook_fn(module, input, output):
            self.activations.append(output)
        hooks = []
        for layer in self.screen:
            hooks.append(layer.register_forward_hook(hook_fn))
            input_tensor = x.permute(0, 3, 1, 2)
        _ = self.screen(input_tensor.float())
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

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

# torch compile debugging
import logging
import torch._dynamo.config as dcfg
dcfg.verbose=True
# dcfg.log_level = logging.DEBUG
# dcfg.print_graph_breaks = True
# dcfg.output_code = True
import torch._functorch.config as fcfg
# fcfg.debug_graphs = True
# fcfg.log_level = logging.DEBUG
import torch._inductor.config as icfg
# icfg.debug = True
# icfg.trace.enabled = True
# import torch._C._jit_tree_views as jit_tree_views
# jit_tree_views.debug = True


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
    def __init__(self, env, *args, framestack=2, flat_size=64*5*6 + 25, input_size=512, hidden_size=512, output_size=512, channels_last=True, downsample=1, **kwargs): #64*6*6+90
        super().__init__()
        self.save_table = True
        self.channels_last = channels_last
        self.downsample = downsample
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)
        self.screen = nn.Sequential(
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
        self.enc_lin = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
         )

    def encode_observations(self, observations):
        observation = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        
        screens = torch.cat([observation['screen'], observation['fixed_window'],], dim=-1)
        screen = screens.permute(0, 3, 1, 2)
        cnn = self.screen(screen.float() / 255.0) # screen and fixed_window stacked
        map = self.map_embedding(observation["map_n"].long()).squeeze(1) # map_id embedding
        pos_cat = torch.cat((map, observation["x"].float(), observation["y"].float(), observation["direction"].float()), dim=-1) # map embedding, x, y, direction
        pos = self.position_fc(pos_cat) # position fc
        event = self.event_fc(observation["events"].float()) # events: badges and bike, hideout, tower, silphco, snorlax_12, snorlax_16, got_flute
        full_cat = torch.cat((cnn, map, event, pos, observation["in_battle"].float()), dim=-1) # final cat also includes in_battle
        final_out = self.enc_lin(full_cat) # final linear layer

        return final_out, None

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

import multiprocessing
from pathlib import Path
from pdb import set_trace as T
import sqlite3
from typing import Any, Optional, List, Tuple
import uuid
from gymnasium import Env, spaces
import numpy as np
import time
from collections import defaultdict, deque
import io, os
import random

from pathlib import Path
import mediapy as media

from .utils.pyboy_binding import (
    ACTIONS,
    make_env,
    open_state_file,
    load_pyboy_state,
    run_action_on_emulator,
)
from .utils import ram_map
from .utils.data import required_events
from .utils.gym_manager import Gym
from .utils.story_manager import Story
from .utils.events import (EventFlags, EVENTS)


STATE_PATH = __file__.rstrip("Pokegym.py") + "States/"
CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]
db_name = Path(f'{str(uuid.uuid4())[:4]}')

class Pokegym:
    counter_lock = multiprocessing.Lock()
    counter = multiprocessing.Value('i', 0)
    def __init__(self, env_config, rom_path="pokemon_red.gb", state_path=None, headless=True, quiet=False, verbose=False, **kwargs,):
        with Pokegym.counter_lock:
            env_id = Pokegym.counter.value
            Pokegym.counter.value += 1
        # Initialize emulator
        if rom_path is None or not os.path.exists(rom_path):
            raise FileNotFoundError("No ROM file found in the specified directory.")
        if state_path is None:
            state_path = STATE_PATH +  "Bulbasaur.state" # STATE_PATH + "has_pokedex_nballs.state"
        self.game, self.screen = make_env(rom_path, headless, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.headless = headless
        self.verbose = verbose
        


        # Configs
        self.swarming = env_config['swarming']
        self.extra_obs = env_config['extra_obs']
        self.add_boey_obs = env_config['add_boey_obs']
        self.full_resets = env_config['full_resets']
        self.anneal = env_config['anneal']
        self.manual_reset = env_config['manual_reset']
        self.max_episode_steps = env_config['max_episode_steps']
        self.rew_reset = env_config['rew_reset']
        self.reward_scale = env_config['reward_scale']
        self.expl_scale = env_config['expl_scale']
        self.reset_mem = env_config['reset_mem']
        self.countdown = env_config['countdown']
        self.inf_money = env_config['inf_money']
        self.save_video = env_config['save_video']
        self.new_events = env_config['new_events']
        self.thatguys_cnn = env_config['thatguys_cnn']
        self.db_path = Path(f"{env_config['db_path']}")
        
        self.n_pokemon_features = 23
        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A]
        self.output_vector_shape = (54, )
        self.visited_pokecenter_list = []
        self.last_10_map_ids = np.zeros(10, dtype=np.uint8)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.init_caches()
        self.past_events_string = ''
        self.last_10_event_ids = np.zeros((10, 2), dtype=np.float32)

        self.obs_size = (72, 80, 1) # 72, 80, 1
        self.screen_memory = defaultdict(lambda: np.zeros((255, 255, 1), dtype=np.uint8))
        self.observation_space = spaces.Dict({})
        self.obs_space()
        self.action_space = spaces.Discrete(len(ACTIONS))
        load_pyboy_state(self.game, self.load_last_state())
        self.env_id = env_id
        self.s_path = Path(f"videos/{self.env_id}")
        
        # Misc
        self.last_reward = None
        self.is_dead = False
        self.time = 0
        self.used_cut = 0
        self.death_count = 0
        self.reset_count = 0
        self.full_reset_count = 0
        self.swarm_count = 0
        self.events = EventFlags(self.game)
        self.gym = Gym(self.events)
        # self.story = Story(self.game)
        self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.poketower = [142, 143, 144, 145, 146, 147, 148]
        self.pokehideout = [199, 200, 201, 202, 203, 135] # , 135
        self.silphco = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        self.safari = [156, 217, 218, 219, 220, 221, 222, 223, 224, 225] # 156 - safari gate2, 21: 'Safari Zone (Rest house 1)', 222: 'Safari Zone (Prize house)', 223: 'Safari Zone (Rest house 2)', 224: 'Safari Zone (Rest house 3)', 225: 'Safari Zone (Rest house 4)'
        r, c, map_n = ram_map.position(self.game)
        self.coords = (c, r, map_n)
        self.hm_count = 0
        self.cut = 0

    def get_fixed_window(self, arr, y, x, window_size):
        height, width, _ = arr.shape
        h_w = window_size[0] // 2
        w_w = window_size[1] // 2

        y_min = max(0, y - h_w)
        y_max = min(height, y + h_w + (window_size[0] % 2))
        x_min = max(0, x - w_w)
        x_max = min(width, x + w_w + (window_size[1] % 2))

        window = arr[y_min:y_max, x_min:x_max]

        pad_top = h_w - (y - y_min)
        pad_bottom = h_w + (window_size[0] % 2) - 1 - (y_max - y - 1)
        pad_left = w_w - (x - x_min)
        pad_right = w_w + (window_size[1] % 2) - 1 - (x_max - x - 1)

        return np.pad(
            window,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

    def render(self):
        screen = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)
        screen = screen[::2, ::2]
        return screen
    
    def obs_space(self):
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(low=0, high=255, shape=self.obs_size, dtype=np.uint8),
                "fixed_window": spaces.Box(low=0, high=255, shape=(72,80,1), dtype=np.uint8),
                "flute": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "bike": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "hideout": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "tower": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "silphco": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "snorlax_12": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "snorlax_16": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "map_n": spaces.Box(low=0, high=250, shape=(1,), dtype=np.uint8),
                'vector': spaces.Box(low=-1, high=1, shape=self.output_vector_shape, dtype=np.float32),
                'map_ids': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                'item_ids': spaces.Box(low=0, high=255, shape=(20,), dtype=np.uint8),
                'item_quantity': spaces.Box(low=-1, high=1, shape=(20, 1), dtype=np.float32),
                'poke_ids': spaces.Box(low=0, high=255, shape=(12,), dtype=np.uint8),
                'poke_type_ids': spaces.Box(low=0, high=255, shape=(12, 2), dtype=np.uint8),
                'poke_move_ids': spaces.Box(low=0, high=255, shape=(12, 4), dtype=np.uint8),
                'poke_move_pps': spaces.Box(low=-1, high=1, shape=(12, 4, 2), dtype=np.float32),
                'poke_all': spaces.Box(low=-1, high=1, shape=(12, self.n_pokemon_features), dtype=np.float32),
                'event_ids': spaces.Box(low=0, high=2570, shape=(10,), dtype=np.uint32),
                'event_step_since': spaces.Box(low=-1, high=1, shape=(10, 1), dtype=np.float32),
            })

    def _get_obs(self):
        c, r, map_n = self.coords
        mmap = self.screen_memory[map_n]
        if 0 <= r <= 254 and 0 <= c <= 254:
            mmap[r, c] = 255
        return {
            "screen": self.render(),
            "fixed_window": self.get_fixed_window(mmap, r, c, self.observation_space['screen'].shape),
            "flute": np.array([ram_map.read_bit(self.game, 0xD76C, 0)], dtype=np.uint8),
            "bike": np.array([ram_map.read_bit(self.game, 0xD75F, 0)], dtype=np.uint8),
            "hideout": np.array([ram_map.read_bit(self.game, 0xD81B, 7)], dtype=np.uint8),
            "tower": np.array([ram_map.read_bit(self.game, 0xD7E0, 7)], dtype=np.uint8),
            "silphco": np.array([ram_map.read_bit(self.game, 0xD838, 7)], dtype=np.uint8),
            "snorlax_12": np.array([ram_map.read_bit(self.game, 0xD7D8, 7)], dtype=np.uint8),
            "snorlax_16": np.array([ram_map.read_bit(self.game, 0xD7E0, 1)], dtype=np.uint8),
            "map_n": np.array([map_n], dtype=np.uint8),
            'vector': self.get_all_raw_obs(),
            'map_ids': self.get_last_map_id_obs(),
            'item_ids': self.get_all_item_ids_obs(),
            'item_quantity': self.get_items_quantity_obs(),
            'poke_ids': self.get_all_pokemon_ids_obs(),
            'poke_type_ids': self.get_all_pokemon_types_obs(),
            'poke_move_ids': self.get_all_move_ids_obs(),
            'poke_move_pps': self.get_all_move_pps_obs(),
            'poke_all': self.get_all_pokemon_obs(),
            'event_ids': self.get_all_event_ids_obs(),
            'event_step_since': self.get_all_event_step_since_obs(),
        }

    def reset(self, seed=None, options=None):
        self.reset_count += 1
        self.reset_state()
        self.reset_var()
        self.visited_pokecenter_list = []
        self.last_10_map_ids = np.zeros(10, dtype=np.uint8)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.init_caches()
        self.past_events_string = ''
        self.last_10_event_ids = np.zeros((10, 2), dtype=np.float32)
        options = options or {}
        info = {}

        if options.get("state", None) is not None:
            self.game.load_state(io.BytesIO(options["state"]))
            self.swarm_count += 1

        if self.swarming:
            self.required_events = self.get_req_events()
            info |= {
                    "state": { tuple(sorted(list(self.required_events))): self.swarming_state()}, # .read()
                    "required_count": len(self.required_events),
                    "env_id": self.env_id,
                    }

        return self._get_obs(), info

    def step(self, action):
        run_action_on_emulator(self.game, ACTIONS[action])
        self.time += 1
        self.events = EventFlags(self.game)
        self.update_last_center()
        self.update_past_events()
        self.init_caches()
        self.past_events_string = self.all_events_string
        # print(f"step: {self.time}")
        # if self.manual_reset:
        #     self.manual_reset_rew()
        if self.save_video:
            self.add_video_frame()

        self.update_pokedex()
        self.update_moves_obtained()
        self.hm_rew()
        self.cut_rew()
        reward = self.reward_scale * self.reward_sum()

        if self.last_reward is None:
            reward = 0
            self.last_reward = 0
        else:
            nxt_reward = reward
            reward -= self.last_reward
            self.last_reward = nxt_reward

        info = {}
        done = self.time >= self.max_episode_steps
        
        if done:
            # print(f"Event Reward: {self.event_reward}")
            if self.save_video:
                self.full_frame_writer.close()
            info = self.infos_dict()

        
        return self._get_obs(), reward, done, done, info
    
#################################################################################################################################################

    def save_to_database(self):
        db_dir = self.db_path
        conn = sqlite3.connect(f'{db_dir}/{db_name}.db')
        cursor = conn.cursor()

        cursor.execute("CREATE TABLE IF NOT EXISTS Pokegym (env_id TEXT PRIMARY KEY,hm_count INTEGER,cut INTEGER)")
        cursor.execute("INSERT OR REPLACE INTO Pokegym VALUES (?, ?, ?)", (str(self.env_id), self.hm_count, self.cut))

        conn.commit()
        conn.close()

    def read_database(self):
        db_dir = self.db_path
        conn = sqlite3.connect(f'{db_dir}/{db_name}.db')
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM Pokegym WHERE cut = 1")
        count_cut_1 = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Pokegym")
        total_instances = cursor.fetchone()[0]
        percentage = (count_cut_1 / total_instances) * 100
        # print(f"Percentage of instances with hm_count = 1: {percentage:.2f}%")
        conn.close()
        return percentage

    def hm_rew(self):
        # HM reward
        hm_count = ram_map.get_hm_count(self.game)
        if hm_count >= 1 and self.hm_count == 0:
            self.hm_count = 1
        # hm_reward = hm_count * 10

    def reset_var(self):
        if self.save_video:
            base_dir = self.s_path
            base_dir.mkdir(parents=True, exist_ok=True)
            full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()

        if self.inf_money:
            r, c, map_n = ram_map.position(self.game)
            if map_n == 7:
                ram_map.write_mem(self.game, 0xD347, 0x09)

        self.screen_memory = defaultdict(lambda: np.zeros((255, 255, 1), dtype=np.uint8))
        self.time = 0
        self.cut_reward = 0
        self.event_reward = 0   
        self.seen_pokemon_reward = 0
        self.caught_pokemon_reward = 0
        self.moves_obtained_reward = 0
        self.used_cut_rew = 0
        self.cut_coords_reward = 0
        self.cut_tiles_reward = 0
        self.max_level_sum = 0
        self.seen_coords = set()
        self.total_healing = 0
        self.hm_count = 0
        self.cut = 0
        self.cut_coords = {}
        self.cut_tiles = {}
        self.cut_state = deque(maxlen=3)
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = {}
        self.cut_counter = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.reward_sum_calc = 0
        self.events = EventFlags(self.game)
        r, c, map_n = ram_map.position(self.game)
        self.coords = (c, r, map_n)

    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.game.memory[i + 0xD2F7]
            seen_mem = self.game.memory[i + 0xD30A]
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0  
    
    def update_moves_obtained(self):
        # Scan party
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
            if self.game.memory[i] != 0:
                for j in range(4):
                    move_id = self.game.memory[i + j + 8]
                    if move_id != 0:
                        if move_id == 15:
                            self.cut = 1
                            self.moves_obtained[move_id] = 9
                        else:
                            self.moves_obtained[move_id] = 1
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.game.memory[0xda80]):
            offset = i*box_struct_length + 0xda96
            if self.game.memory[offset] != 0:
                for j in range(4):
                    move_id = self.game.memory[offset + j + 8]
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1

    def video(self):
        video = self.screen.screen_ndarray()
        return video
            
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.video())

    def save_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.game.save_state(state)
        self.initial_states.append(state)

    def swarming_state(self):
        state = io.BytesIO()
        self.game.save_state(state)
        state.seek(0)
        return state.read()
    
    def get_req_events(self):
        events_return = []
        for event in required_events:
            events_done = ram_map.read_bit(self.game, event[0], event[1])
            if events_done:
                events_return.append(1)
        return(events_return)
    
    def update_state(self, state: bytes):
        self.reset(seed=random.randint(0, 10), options={"state": state})
    
    def load_last_state(self):
        return self.initial_states[len(self.initial_states) - 1]
    
    def load_first_state(self):
        return self.initial_states[0]
    
    def load_random_state(self):
        rand_idx = random.randint(0, len(self.initial_states) - 1)
        return self.initial_states[rand_idx]
    
    def reset_state(self):
        self.countdown -= 1
        if self.full_resets:
            if self.countdown == 0:
                self.reset_mem += 1
                self.countdown = self.reset_mem
                load_pyboy_state(self.game, self.load_last_state())
                self.full_reset_count += 1
        if self.anneal:
            if self.countdown == 0:
                self.countdown = 10
                self.max_episode_steps += 2048
            load_pyboy_state(self.game, self.load_last_state())

    def expl_rew(self):
        r, c, map_n = ram_map.position(self.game) # this is [y, x, z]
        self.seen_coords.add((r, c, map_n))
        self.coords = (c, r, map_n)
        # # high_gym_maps, low_gym_maps = self.gym.maps()

        # # if map_n in high_gym_maps:
        # #     exploration_reward = (0.03 * len(self.seen_coords)) 
        # else:
        if not self.events.get_event('EVENT_FOUND_ROCKET_HIDEOUT'):
            if map_n in self.poketower:
                exploration_reward = 0
            elif map_n == 135:
                exploration_reward = (0.03 * len(self.seen_coords)) 
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif not self.events.get_event('EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI') and self.events.get_event('EVENT_FOUND_ROCKET_HIDEOUT'):
            if map_n in self.pokehideout:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif not self.events.get_event('EVENT_RESCUED_MR_FUJI') and self.events.get_event('EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI'):
            if map_n in self.poketower:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif not self.events.get_event('EVENT_GOT_POKE_FLUTE') and self.events.get_event('EVENT_RESCUED_MR_FUJI'):
            if map_n == 149:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif not self.events.get_event('EVENT_BEAT_SILPH_CO_GIOVANNI') and self.events.get_event('EVENT_GOT_POKE_FLUTE'):
            if map_n in self.silphco:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif not self.events.get_event('EVENT_GOT_HM03') or not self.events.get_event('EVENT_GAVE_GOLD_TEETH'):
            if map_n in self.safari or map_n == 7 or map_n == 155:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords)) 
        else:
            exploration_reward = (0.02 * len(self.seen_coords))
        
        
        # # Story
        # self.story.update()
        # high_story_maps, low_story_maps = self.story.maps()
        # # print(f'Low Story: {low_story_maps}\n High Story: {high_story_maps}')

        # # New Exploration
        # self.expl_high_map = high_gym_maps + high_story_maps
        # self.expl_low_map = low_gym_maps + low_story_maps
        # r, c, map_n = ram_map.position(self.game) # this is [y, x, z]
        # self.seen_coords.add((r, c, map_n))
        # if map_n in self.expl_high_map:
        #     self.exploration_reward = (0.03 * len(self.seen_coords))
        # elif map_n in self.expl_low_map:
        #     self.exploration_reward = (0.01 * len(self.seen_coords))
        # else:
        #     self.exploration_reward = (0.02 * len(self.seen_coords))

        return exploration_reward

    def cut_rew(self):
        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        c, r, map_n = self.coords # this is [x, y, z]
        if ram_map.mem_val(self.game, 0xD057) == 0: # is_in_battle if 1
            if self.cut == 1:
                player_direction = self.game.memory[0xC109]
                if player_direction == 0:  # down
                    coords = (c, r + 1, map_n)
                if player_direction == 4:
                    coords = (c, r - 1, map_n)
                if player_direction == 8:
                    coords = (c - 1, r, map_n)
                if player_direction == 0xC:
                    coords = (c + 1, r, map_n)
                self.cut_state.append(
                    (
                        self.game.memory[0xCFC6],
                        self.game.memory[0xCFCB],
                        self.game.memory[0xCD6A],
                        self.game.memory[0xD367],
                        self.game.memory[0xD125],
                        self.game.memory[0xCD3D],
                    )
                )
                if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                    self.cut_coords[coords] = 5 # from 14
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif self.cut_state == CUT_GRASS_SEQ:
                    self.cut_coords[coords] = 0.001
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
                    self.cut_coords[coords] = 0.001
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                if int(ram_map.read_bit(self.game, 0xD803, 0)):
                    if ram_map.check_if_in_start_menu(self.game):
                        self.seen_start_menu = 1
                    if ram_map.check_if_in_pokemon_menu(self.game):
                        self.seen_pokemon_menu = 1
                    if ram_map.check_if_in_stats_menu(self.game):
                        self.seen_stats_menu = 1
                    if ram_map.check_if_in_bag_menu(self.game):
                        self.seen_bag_menu = 1

        if ram_map.used_cut(self.game) == 61 and self.cut_counter >= 5:
            ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write resets tile check
            self.used_cut += 1
            self.cut_counter += 1

    def reward_sum(self):
        exploration_reward = self.expl_rew()
        level_reward = self.level_rew()
        healing_reward = self.heal_rew()
        if self.new_events:
            if self.time % 2 == 0:
                events = [self.events.get_event(event) for event in EVENTS]
                self.event_reward = sum(events)*3
        else:
            if self.time % 2 == 0:
                silph = ram_map.silph_co(self.game)
                rock_tunnel = ram_map.rock_tunnel(self.game)
                ssanne = ram_map.ssanne(self.game)
                mtmoon = ram_map.mtmoon(self.game)
                routes = ram_map.routes(self.game)
                misc = ram_map.misc(self.game)
                snorlax = ram_map.snorlax(self.game)
                hmtm = ram_map.hmtm(self.game)
                bill = ram_map.bill(self.game)
                oak = ram_map.oak(self.game)
                towns = ram_map.towns(self.game)
                lab = ram_map.lab(self.game)
                mansion = ram_map.mansion(self.game)
                safari = ram_map.safari(self.game)
                dojo = ram_map.dojo(self.game)
                hideout = ram_map.hideout(self.game)
                tower = ram_map.poke_tower(self.game)
                gym1 = ram_map.gym1(self.game)
                gym2 = ram_map.gym2(self.game)
                gym3 = ram_map.gym3(self.game)
                gym4 = ram_map.gym4(self.game)
                gym5 = ram_map.gym5(self.game)
                gym6 = ram_map.gym6(self.game)
                gym7 = ram_map.gym7(self.game)
                gym8 = ram_map.gym8(self.game)
                rival = ram_map.rival(self.game)
                self.event_reward = sum([silph, rock_tunnel, ssanne, mtmoon, routes, misc, snorlax, hmtm, bill, oak, towns, lab, mansion, safari, dojo, hideout, tower, gym1, gym2, gym3, gym4, gym5, gym6, gym7, gym8, rival])
            # print(f"Event Reward: {self.event_reward}")

        self.cut_reward = self.cut * 10
        self.seen_pokemon_reward = sum(self.seen_pokemon) * self.reward_scale
        self.caught_pokemon_reward = sum(self.caught_pokemon) * self.reward_scale
        self.moves_obtained_reward = sum(self.moves_obtained.values()) * self.reward_scale
        self.used_cut_rew = self.used_cut * 0.1
        self.cut_coords_reward = sum(self.cut_coords.values())
        self.cut_tiles_reward = len(self.cut_tiles)
        start_menu = self.seen_start_menu * 0.01
        pokemon_menu = self.seen_pokemon_menu * 0.1
        stats_menu = self.seen_stats_menu * 0.1
        bag_menu = self.seen_bag_menu * 0.1
        that_guy = (start_menu + pokemon_menu + stats_menu + bag_menu ) / 2
        self.reward_sum_calc = (
            + level_reward
            + healing_reward
            + exploration_reward 
            + self.cut_reward
            + self.event_reward     
            + self.seen_pokemon_reward
            + self.caught_pokemon_reward
            + self.moves_obtained_reward
            + self.used_cut_rew
            + self.cut_coords_reward
            + self.cut_tiles_reward
            + that_guy
        )
        return self.reward_sum_calc
    
    def infos_dict(self):
        info = {
            "Data": {
                "brock": self.events.get_event('EVENT_BEAT_BROCK'),
                "misty": self.events.get_event('EVENT_BEAT_MISTY'),
                "surge": self.events.get_event('EVENT_BEAT_LT_SURGE'),
                "erika": self.events.get_event('EVENT_BEAT_ERIKA'),
                "koga": self.events.get_event('EVENT_BEAT_KOGA'),
                "sabrina": self.events.get_event('EVENT_BEAT_SABRINA'),
                "blaine": self.events.get_event('EVENT_BEAT_BLAINE'),
                "giovanni": self.events.get_event('EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI'),
                "got_bike": self.events.get_event('EVENT_GOT_BICYCLE'),
                "beat_hideout": self.events.get_event('EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI'),
                "saved_fuji": self.events.get_event('EVENT_RESCUED_MR_FUJI'),
                "got_flute": self.events.get_event('EVENT_GOT_POKE_FLUTE'),
                "beat_silphco": self.events.get_event('EVENT_BEAT_SILPH_CO_GIOVANNI'),
                "beat_snorlax_12": self.events.get_event('EVENT_BEAT_ROUTE12_SNORLAX'),
                "beat_snorlax_16": self.events.get_event('EVENT_BEAT_ROUTE16_SNORLAX'),
            },
            # "Events": self.events.event_rewards(),
            # "Rewards": {
            #     "Reward_Sum": self.reward_sum(),
            #     "Exploration": self.expl_rew(),
            #     "Level": self.level_rew(),
            #     "Healing": self.heal_rew(),
            #     "Event_Sum": self.event_reward,
            #     "Cut": self.cut_reward,    
            #     "Seen_Poke": self.seen_pokemon_reward,
            #     "Caught_Poke": self.caught_pokemon_reward,
            #     "Moves_Obtained": self.moves_obtained_reward,
            #     "Used_Cut": self.used_cut_rew,
            #     "Cut_Coords": self.cut_coords_reward,
            #     "Cut_Tiles": self.cut_tiles_reward,
            #     "Start_Menu": self.seen_start_menu * 0.01,
            #     "Poke_Menu": self.seen_pokemon_menu * 0.1,
            #     "Stats_Menu": self.seen_stats_menu * 0.1,
            #     "Bag_Menu": self.seen_bag_menu * 0.1,
            # },
            # "Misc": {
            #     "cut": self.cut,
            #     "deaths": self.death_count,
            #     "local_expl_rew": len(self.seen_coords)/self.max_episode_steps,
            # },
        }
        if self.swarming:
            required_events = self.get_req_events()
            new_required_events = sum(required_events) - sum(self.required_events)
            if new_required_events:
                info |= {
                    "state": { tuple(sorted(list(required_events))): self.swarming_state()},
                    "required_count": len(required_events),
                    "env_id": self.env_id,
                }
                self.required_events = required_events
        return info
    
    def level_rew(self):
        party_size, party_levels = ram_map.party(self.game)
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        # level_reward = sum(party_levels)/600 # test line
        if self.max_level_sum < 15:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 15 + (self.max_level_sum - 15) / 4
        return level_reward

    def heal_rew(self):
        party_size, party_levels = ram_map.party(self.game)
        # Healing and death rewards
        hp = ram_map.hp(self.game)
        hp_delta = hp - self.last_hp
        party_size_constant = party_size == self.last_party_size
        if hp_delta > 0.5 and party_size_constant and not self.is_dead:
            self.total_healing += hp_delta
        if hp <= 0 and self.last_hp > 0:
            self.death_count += 1
            self.is_dead = True
        elif hp > 0.01:  # TODO: Check if this matters
            self.is_dead = False
        self.last_hp = hp
        self.last_party_size = party_size
        death_reward = 0 # -0.08 * self.death_count  # -0.05
        healing_reward = self.total_healing
        return healing_reward


    def update_last_center(self):
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)

    def multi_hot_encoding(self, cnt, max_n):
        return [1 if cnt < i else 0 for i in range(max_n)]
    
    def one_hot_encoding(self, cnt, max_n, start_zero=False):
        if start_zero:
            return [1 if cnt == i else 0 for i in range(max_n)]
        else:
            return [1 if cnt == i+1 else 0 for i in range(max_n)]
    
    def scaled_encoding(self, cnt, max_n: float):
        max_n = float(max_n)
        if isinstance(cnt, list):
            return [min(1.0, c / max_n) for c in cnt]
        elif isinstance(cnt, np.ndarray):
            return np.clip(cnt / max_n, 0, 1)
        else:
            return min(1.0, cnt / max_n)
    
    def get_badges_obs(self):
        return self.multi_hot_encoding(self.get_badges(), 12)

    def get_money_obs(self):
        return [self.scaled_encoding(self.read_money(), 100_000)]
    
    def read_swap_mon_pos(self):
        is_in_swap_mon_party_menu = self.read_m(0xd07d) == 0x04
        if is_in_swap_mon_party_menu:
            chosen_mon = self.read_m(0xcc35)
            if chosen_mon == 0:
                print(f'\nsomething went wrong, chosen_mon is 0')
            else:
                return chosen_mon - 1
        return -1
    
    def get_last_pokecenter_obs(self):
        return self.get_last_pokecenter_list()

    def get_visited_pokecenter_obs(self):
        result = [0] * len(self.pokecenter_ids)
        for i in self.visited_pokecenter_list:
            result[i] = 1
        return result
    
    def get_hm_move_obs(self):
        hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        result = [0] * len(hm_moves)
        all_moves = self.get_party_moves()
        for i, hm_move in enumerate(hm_moves):
            if hm_move in all_moves:
                result[i] = 1
                continue
        return result
    
    def get_hm_obs(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.get_items_in_bag()
        result = [0] * len(hm_ids)
        for i, hm_id in enumerate(hm_ids):
            if hm_id in items:
                result[i] = 1
                continue
        return result
    
    def get_items_in_bag(self, one_indexed=0):
        first_item = 0xD31E
        # total 20 items
        # item1, quantity1, item2, quantity2, ...
        item_ids = []
        for i in range(0, 20, 2):
            item_id = self.read_m(first_item + i)
            if item_id == 0 or item_id == 0xff:
                break
            item_ids.append(item_id + one_indexed)
        return item_ids
    
    def get_items_obs(self):
        # items from self.get_items_in_bag()
        # add 0s to make it 20 items
        items = self.get_items_in_bag(one_indexed=1)
        items.extend([0] * (20 - len(items)))
        return items

    def get_items_quantity_obs(self):
        # items from self.get_items_quantity_in_bag()
        # add 0s to make it 20 items
        items = self.get_items_quantity_in_bag()
        items = self.scaled_encoding(items, 20)
        items.extend([0] * (20 - len(items)))
        return np.array(items, dtype=np.float32).reshape(-1, 1)

    def get_bag_full_obs(self):
        # D31D
        return [1 if self.read_m(0xD31D) >= 20 else 0]
    
    def get_last_10_map_ids_obs(self):
        return self.last_10_map_ids
    
    def get_last_10_coords_obs(self):
        # 10, 2
        # scale x with 45, y with 72
        result = []
        for coord in self.last_10_coords:
            result.append(min(coord[0] / 45, 1))
            result.append(min(coord[1] / 72, 1))
        return result
    
    def get_pokemon_ids_obs(self):
        return self.read_party(one_indexed=1)
    
    def read_party(self, one_indexed=0):
        parties = [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
        return [p + one_indexed if p != 0xff and p != 0 else 0 for p in parties]
    
    def get_battle_pokemon_ids_obs(self):
        battle_pkmns = [self.read_m(addr) for addr in [0xcfe5, 0xd014]]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in battle_pkmns]
    
    def get_party_types_obs(self):
        # 6 pokemon, 2 types each
        # start from D170 type1, D171 type2
        # next pokemon will be + 44
        # 0xff is no pokemon
        result = []
        for i in range(0, 44*6, 44):
            # 2 types per pokemon
            type1 = self.read_m(0xD170 + i)
            type2 = self.read_m(0xD171 + i)
            result.append(type1)
            result.append(type2)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_opp_types_obs(self):
        # 6 pokemon, 2 types each
        # start from D8A9 type1, D8AA type2
        # next pokemon will be + 44
        # 0xff is no pokemon
        result = []
        for i in range(0, 44*6, 44):
            # 2 types per pokemon
            type1 = self.read_m(0xD8A9 + i)
            type2 = self.read_m(0xD8AA + i)
            result.append(type1)
            result.append(type2)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_battle_types_obs(self):
        # CFEA type1, CFEB type2
        # d019 type1, d01a type2
        result = [self.read_m(0xcfea), self.read_m(0xCFEB), self.read_m(0xD019), self.read_m(0xD01A)]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_party_move_ids_obs(self):
        # D173 move1, D174 move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD173, 0xD174, 0xD175, 0xD176]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_opp_move_ids_obs(self):
        # D8AC move1, D8AD move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD8AC, 0xD8AD, 0xD8AE, 0xD8AF]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_battle_move_ids_obs(self):
        # CFED move1, CFEE move2
        # second pokemon starts from D003
        result = []
        for addr in [0xCFED, 0xD003]:
            moves = [self.read_m(addr + i) for i in range(4)]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_party_move_pps_obs(self):
        # D188 pp1, D189 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD188, 0xD189, 0xD18A, 0xD18B]]
            result.extend(pps)
        return result
    
    def get_opp_move_pps_obs(self):
        # D8C1 pp1, D8C2 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD8C1, 0xD8C2, 0xD8C3, 0xD8C4]]
            result.extend(pps)
        return result
    
    def get_battle_move_pps_obs(self):
        # CFFE pp1, CFFF pp2
        # second pokemon starts from D02D
        result = []
        for addr in [0xCFFE, 0xD02D]:
            pps = [self.read_m(addr + i) for i in range(4)]
            result.extend(pps)
        return result
    
    def get_party_level_obs(self):
        # D18C level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD18C + i)
            result.append(level)
        return result
    
    def get_opp_level_obs(self):
        # D8C5 level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD8C5 + i)
            result.append(level)
        return result
    
    def get_battle_level_obs(self):
        # CFF3 level
        # second pokemon starts from D037
        result = []
        for addr in [0xCFF3, 0xD022]:
            level = self.read_m(addr)
            result.append(level)
        return result
    
    def get_all_level_obs(self):
        result = []
        result.extend(self.get_party_level_obs())
        result.extend(self.get_opp_level_obs())
        result.extend(self.get_battle_level_obs())
        result = np.array(result, dtype=np.float32) / 100
        # every elemenet max is 1
        result = np.clip(result, 0, 1)
        return result
    
    def get_party_hp_obs(self):
        # D16C hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.read_hp(0xD16C + i)
            max_hp = self.read_hp(0xD18D + i)
            result.extend([hp, max_hp])
        return result
    
    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    def get_opp_hp_obs(self):
        # D8A5 hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.read_hp(0xD8A5 + i)
            max_hp = self.read_hp(0xD8C6 + i)
            result.extend([hp, max_hp])
        return result
    
    def get_battle_hp_obs(self):
        # CFE6 hp
        # second pokemon starts from CFFC
        result = []
        for addr in [0xCFE6, 0xCFF4, 0xCFFC, 0xD00A]:
            hp = self.read_hp(addr)
            result.append(hp)
        return result
    
    def get_all_hp_obs(self):
        result = []
        result.extend(self.get_party_hp_obs())
        result.extend(self.get_opp_hp_obs())
        result.extend(self.get_battle_hp_obs())
        result = np.array(result, dtype=np.float32)
        # every elemenet max is 1
        result = np.clip(result, 0, 600) / 600
        return result
    
    def get_all_hp_pct_obs(self):
        hps = []
        hps.extend(self.get_party_hp_obs())
        hps.extend(self.get_opp_hp_obs())
        hps.extend(self.get_battle_hp_obs())
        # divide every hp by max hp
        hps = np.array(hps, dtype=np.float32)
        hps = hps.reshape(-1, 2)
        hps = hps[:, 0] / (hps[:, 1] + 0.00001)
        # every elemenet max is 1
        return hps
    
    def get_all_pokemon_dead_obs(self):
        # 1 if dead, 0 if alive
        hp_pct = self.get_all_hp_pct_obs()
        return [1 if hp <= 0 else 0 for hp in hp_pct]
    
    def get_battle_status_obs(self):
        # D057
        # 0 not in battle return 0, 0
        # 1 wild battle return 1, 0
        # 2 trainer battle return 0, 1
        # -1 lost battle return 0, 0
        result = []
        status = self.battle_type
        if status == 1:
            result = [1, 0]
        elif status == 2:
            result = [0, 1]
        else:
            result = [0, 0]
        return result
    
    def fix_pokemon_type(self, ptype: int) -> int:
        if ptype < 9:
            return ptype
        elif ptype < 27:
            return ptype - 11
        else:
            print(f'invalid pokemon type: {ptype}')
            return 16
        
    def get_pokemon_types(self, start_addr):
        return [self.fix_pokemon_type(self.read_m(start_addr + i)) + 1 for i in range(2)]
        
    def get_all_pokemon_types_obs(self):
        # 6 party pokemon types start from D170
        # 6 enemy pokemon types start from D8A9
        party_type_addr = 0xD170
        enemy_type_addr = 0xD8A9
        result = []
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            # 2 types per pokemon
            ptypes = self.get_pokemon_types(party_type_addr + i * 44)
            result.append(ptypes)
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append([0, 0])
        if self.is_in_battle():
            # zero padding if not in battle, reduce dimension
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    # 2 types per pokemon
                    ptypes = self.get_pokemon_types(enemy_type_addr + i * 44)
                    result.append(ptypes)
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0, 0])
            else:
                wild_ptypes = self.get_pokemon_types(0xCFEA)  # 2 ptypes only, add padding for remaining 5
                result.append(wild_ptypes)
                result.extend([[0, 0]] * 5)
        else:
            result.extend([[0, 0]] * 6)
        result = np.array(result, dtype=np.uint8)  # shape (24,)
        assert result.shape == (12, 2), f'invalid ptypes shape: {result.shape}'  # set PYTHONOPTIMIZE=1 to disable assert
        return result
    
    def get_pokemon_status(self, addr):
        # status
        # bit 0 - 6
        # one byte has 8 bits, bit unused: 7
        statuses = [self.read_bit(addr, i) for i in range(7)]
        return statuses  # shape (7,)
    
    def get_one_pokemon_obs(self, start_addr, team, position, is_wild=False):
        # team 0 = my team, 1 = opp team
        # 1 pokemon, address start from start_addr
        # +0 = id
        # +5 = type1 (15 types) (physical 0 to 8 and special 20 to 26)  + 1 to be 1 indexed, 0 is no pokemon/padding
        # +6 = type2 (15 types)
        # +33 = level
        # +4 = status (bit 0-6)
        # +1 = current hp (2 bytes)
        # +34 = max hp (2 bytes)
        # +36 = attack (2 bytes)
        # +38 = defense (2 bytes)
        # +40 = speed (2 bytes)
        # +42 = special (2 bytes)
        # exclude id, type1, type2
        result = []
        # status
        status = self.get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.scaled_encoding(self.read_m(start_addr + 33), 100)
        result.append(level)
        # hp
        hp = self.scaled_encoding(self.read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.scaled_encoding(self.read_double(start_addr + 34), 250)
        result.append(max_hp)
        # attack
        attack = self.scaled_encoding(self.read_double(start_addr + 36), 134)
        result.append(attack)
        # defense
        defense = self.scaled_encoding(self.read_double(start_addr + 38), 180)
        result.append(defense)
        # speed
        speed = self.scaled_encoding(self.read_double(start_addr + 40), 140)
        result.append(speed)
        # special
        special = self.scaled_encoding(self.read_double(start_addr + 42), 154)
        result.append(special)
        # is alive
        is_alive = 1 if hp > 0 else 0
        result.append(is_alive)
        # is in battle, check position 0 indexed against the following addr
        if is_wild:
            in_battle = 1
        else:
            if self.is_in_battle():
                if team == 0:
                    in_battle = 1 if position == self.read_m(0xCC35) else 0
                else:
                    in_battle = 1 if position == self.read_m(0xCFE8) else 0
            else:
                in_battle = 0
        result.append(in_battle)
        # my team 0 / opp team 1
        result.append(team)
        # position 0 to 5, one hot, 5 elements, first pokemon is all 0
        result.extend(self.one_hot_encoding(position, 5))
        # is swapping this pokemon
        if team == 0:
            swap_mon_pos = self.read_swap_mon_pos()
            if swap_mon_pos != -1:
                is_swapping = 1 if position == swap_mon_pos else 0
            else:
                is_swapping = 0
        else:
            is_swapping = 0
        result.append(is_swapping)
        return result

    def get_party_pokemon_obs(self):
        # 6 party pokemons start from D16B
        # 2d array, 6 pokemons, N features
        result = np.zeros((6, self.n_pokemon_features), dtype=np.float32)
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result[i] = self.get_one_pokemon_obs(0xD16B + i * 44, 0, i)
        for i in range(pokemon_count, 6):
            result[i] = np.zeros(self.n_pokemon_features, dtype=np.float32)
        return result

    def read_opp_pokemon_num(self):
        return self.read_m(0xD89C)
    
    def get_battle_base_pokemon_obs(self, start_addr, team):
        # CFE5
        result = []
        # status
        status = self.get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.scaled_encoding(self.read_m(start_addr + 14), 100)
        result.append(level)
        # hp
        hp = self.scaled_encoding(self.read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.scaled_encoding(self.read_double(start_addr + 15), 250)
        result.append(max_hp)
        # attack
        attack = self.scaled_encoding(self.read_double(start_addr + 17), 134)
        result.append(attack)
        # defense
        defense = self.scaled_encoding(self.read_double(start_addr + 19), 180)
        result.append(defense)
        # speed
        speed = self.scaled_encoding(self.read_double(start_addr + 21), 140)
        result.append(speed)
        # special
        special = self.scaled_encoding(self.read_double(start_addr + 23), 154)
        result.append(special)
        # is alive
        is_alive = 1 if hp > 0 else 0
        result.append(is_alive)
        # is in battle, check position 0 indexed against the following addr
        in_battle = 1
        result.append(in_battle)
        # my team 0 / opp team 1
        result.append(team)
        # position 0 to 5, one hot, 5 elements, first pokemon is all 0
        result.extend(self.one_hot_encoding(0, 5))
        return result
    
    def get_wild_pokemon_obs(self):
        start_addr = 0xCFE5
        return self.get_battle_base_pokemon_obs(start_addr, team=1)

    def get_opp_pokemon_obs(self):
        # 6 enemy pokemons start from D8A4
        # 2d array, 6 pokemons, N features
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.get_one_pokemon_obs(0xD8A4 + i * 44, 1, i))
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0] * self.n_pokemon_features)
            else:
                # wild battle, take the battle pokemon
                result.append(self.get_wild_pokemon_obs())
                for i in range(5):
                    result.append([0] * self.n_pokemon_features)
        else:
            return np.zeros((6, self.n_pokemon_features), dtype=np.float32)
        result = np.array(result, dtype=np.float32)
    
    def get_all_pokemon_obs(self):
        # 6 party pokemons start from D16B
        # 6 enemy pokemons start from D8A4
        # gap between each pokemon is 44
        party = self.get_party_pokemon_obs()
        opp = self.get_opp_pokemon_obs()
        # print(f'party shape: {party.shape}, opp shape: {opp.shape}')
        result = np.concatenate([party, opp], axis=0)
        return result  # shape (12, 22)
    
    def get_party_pokemon_ids_obs(self):
        # 6 party pokemons start from D16B
        # 1d array, 6 pokemons, 1 id
        result = []
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result.append(self.read_m(0xD16B + i * 44) + 1)
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append(0)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_opp_pokemon_ids_obs(self):
        # 6 enemy pokemons start from D8A4
        # 1d array, 6 pokemons, 1 id
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.read_m(0xD8A4 + i * 44) + 1)
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append(0)
            else:
                # wild battle, take the battle pokemon
                result.append(self.read_m(0xCFE5) + 1)
                for i in range(5):
                    result.append(0)
        else:
            return np.zeros(6, dtype=np.uint8)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_all_pokemon_ids_obs(self):
        # 6 party pokemons start from D16B
        # 6 enemy pokemons start from D8A4
        # gap between each pokemon is 44
        party = self.get_party_pokemon_ids_obs()
        opp = self.get_opp_pokemon_ids_obs()
        result = np.concatenate((party, opp), axis=0)
        return result
    
    def get_one_pokemon_move_ids_obs(self, start_addr):
        # 4 moves
        return [self.read_m(start_addr + i) for i in range(4)]
    
    def get_party_pokemon_move_ids_obs(self):
        # 6 party pokemons start from D173
        # 2d array, 6 pokemons, 4 moves
        result = []
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result.append(self.get_one_pokemon_move_ids_obs(0xD173 + (i * 44)))
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append([0] * 4)
        result = np.array(result, dtype=np.uint8)
        return result

    def get_opp_pokemon_move_ids_obs(self):
        # 6 enemy pokemons start from D8AC
        # 2d array, 6 pokemons, 4 moves
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.get_one_pokemon_move_ids_obs(0xD8AC + (i * 44)))
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0] * 4)
            else:
                # wild battle, take the battle pokemon
                result.append(self.get_one_pokemon_move_ids_obs(0xCFED))
                for i in range(5):
                    result.append([0] * 4)
        else:
            return np.zeros((6, 4), dtype=np.uint8)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_all_move_ids_obs(self):
        # 6 party pokemons start from D173
        # 6 enemy pokemons start from D8AC
        # gap between each pokemon is 44
        party = self.get_party_pokemon_move_ids_obs()
        opp = self.get_opp_pokemon_move_ids_obs()
        result = np.concatenate((party, opp), axis=0)
        return result  # shape (12, 4)
    
    def get_one_pokemon_move_pps_obs(self, start_addr):
        # 4 moves
        result = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            pp = self.scaled_encoding(self.read_m(start_addr + i), 30)
            have_pp = 1 if pp > 0 else 0
            result[i] = [pp, have_pp]
        return result
    
    def get_party_pokemon_move_pps_obs(self):
        # 6 party pokemons start from D188
        # 2d array, 6 pokemons, 8 features
        # features: pp, have pp
        result = np.zeros((6, 4, 2), dtype=np.float32)
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result[i] = self.get_one_pokemon_move_pps_obs(0xD188 + (i * 44))
        for i in range(pokemon_count, 6):
            result[i] = np.zeros((4, 2), dtype=np.float32)
        return result
    
    def get_opp_pokemon_move_pps_obs(self):
        # 6 enemy pokemons start from D8C1
        # 2d array, 6 pokemons, 8 features
        # features: pp, have pp
        result = np.zeros((6, 4, 2), dtype=np.float32)
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result[i] = self.get_one_pokemon_move_pps_obs(0xD8C1 + (i * 44))
                for i in range(pokemon_count, 6):
                    result[i] = np.zeros((4, 2), dtype=np.float32)
            else:
                # wild battle, take the battle pokemon
                result.append(self.get_one_pokemon_move_pps_obs(0xCFFE))
                for i in range(5):
                    result.append(np.zeros((4, 2), dtype=np.float32))
        else:
            return np.zeros((6, 4, 2), dtype=np.float32)
        return result
    
    def get_all_move_pps_obs(self):
        # 6 party pokemons start from D188
        # 6 enemy pokemons start from D8C1
        party = self.get_party_pokemon_move_pps_obs()
        opp = self.get_opp_pokemon_move_pps_obs()
        result = np.concatenate((party, opp), axis=0)
        return result
    
    def get_all_item_ids_obs(self):
        # max 85
        return np.array(self.get_items_obs(), dtype=np.uint8)
    
    def get_all_event_ids_obs(self):
        # max 249
        # padding_idx = 0
        # change dtype to uint8 to save space
        return np.array(self.last_10_event_ids[:, 0] + 1, dtype=np.uint8)
    
    def get_all_event_step_since_obs(self):
        step_gotten = self.last_10_event_ids[:, 1]  # shape (10,)
        step_since = self.time - step_gotten
        # step_count - step_since and scaled_encoding
        return self.scaled_encoding(step_since, 1000).reshape(-1, 1)  # shape (10,)
    
    def get_last_coords_obs(self):
        # 2 elements
        coord = self.last_10_coords[0]
        return [self.scaled_encoding(coord[0], 45), self.scaled_encoding(coord[1], 72)]
    
    def get_num_turn_in_battle_obs(self):
        if self.is_in_battle:
            return self.scaled_encoding(self.read_m(0xCCD5), 30)
        else:
            return 0
    
    def get_all_raw_obs(self):
        obs = []
        obs.extend(self.get_badges_obs())
        obs.extend(self.get_money_obs())
        obs.extend(self.get_last_pokecenter_obs())
        obs.extend(self.get_visited_pokecenter_obs())
        obs.extend(self.get_hm_move_obs())
        obs.extend(self.get_hm_obs())
        obs.extend(self.get_battle_status_obs())
        pokemon_count = self.read_num_poke()
        obs.extend([self.scaled_encoding(pokemon_count, 6)])  # number of pokemon
        obs.extend([1 if pokemon_count == 6 else 0])  # party full
        obs.extend([self.scaled_encoding(self.read_m(0xD31D), 20)])  # bag num items
        obs.extend(self.get_bag_full_obs())  # bag full
        obs.extend(self.get_last_coords_obs())  # last coords x, y
        obs.extend([self.get_num_turn_in_battle_obs()])  # num turn in battle
        # obs.extend(self.get_reward_check_obs())  # reward check
        return np.array(obs, dtype=np.float32)

    def get_last_map_id_obs(self):
        return np.array([self.last_10_map_ids[0]], dtype=np.uint8)
    
    def get_in_battle_mask_obs(self):
        return np.array([self.is_in_battle()], dtype=np.float32)

    def update_past_events(self):
        if self.past_events_string and self.past_events_string != self.all_events_string:
            self.last_10_event_ids = np.roll(self.last_10_event_ids, 1, axis=0)
            self.last_10_event_ids[0] = [self.get_first_diff_index(self.past_events_string, self.all_events_string), self.time]

    def read_num_poke(self):
        return self.read_m(0xD163)
    
    def get_items_quantity_in_bag(self):
        first_quantity = 0xD31F
        # total 20 items
        # quantity1, item2, quantity2, ...
        item_quantities = []
        for i in range(1, 20, 2):
            item_quantity = self.read_m(first_quantity + i)
            if item_quantity == 0 or item_quantity == 0xff:
                break
            item_quantities.append(item_quantity)
        return item_quantities
    
    def is_in_battle(self):
        # D057
        # 0 not in battle
        # 1 wild battle
        # 2 trainer battle
        # -1 lost battle
        return self.battle_type > 0
    
    @property
    def battle_type(self):
        if not self._battle_type:
            result = self.read_m(0xD057)
            if result == -1:
                return 0
            return result
        return self._battle_type
    
    def is_wild_battle(self):
        return self.battle_type == 1
    
    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))
    
    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))
    
    def get_last_pokecenter_list(self):
        pc_list = [0, ] * len(self.pokecenter_ids)
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1:
            pc_list[last_pokecenter_id] = 1
        return pc_list
    
    def get_last_pokecenter_id(self):
        last_pokecenter = self.read_m(0xD719)
        # will throw error if last_pokecenter not in pokecenter_ids, intended
        if last_pokecenter == 0:
            # no pokecenter visited yet
            return -1
        return self.pokecenter_ids.index(last_pokecenter)
    
    def get_party_moves(self):
        # first pokemon moves at D173
        # 4 moves per pokemon
        # next pokemon moves is 44 bytes away
        first_move = 0xD173
        moves = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            move = [self.read_m(first_move + i + j) for j in range(4)]
            moves.extend(move)
        return moves
    
    def read_m(self, addr):
        return self.game.memory[addr]

    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def read_double(self, start_add):
        return 256*self.read_m(start_add) + self.read_m(start_add+1)
    
    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    @property
    def all_events_string(self):
        # cache all events string to improve performance
        if not self._all_events_string:
            event_flags_start = 0xD747
            event_flags_end = 0xD886
            result = ''
            for i in range(event_flags_start, event_flags_end):
                result += bin(self.read_m(i))[2:]  # .zfill(8)
            self._all_events_string = result
        return self._all_events_string
    
    def init_caches(self):
        # for cached properties
        self._all_events_string = ''
        self._battle_type = -999

    def get_first_diff_index(self, arr1, arr2):
        for i in range(len(arr1)):
            if arr1[i] != arr2[i]:
                return i
        return -1


    def close(self):
        self.game.stop(False)

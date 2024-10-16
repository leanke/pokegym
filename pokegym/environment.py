import multiprocessing
from pathlib import Path
from pdb import set_trace as T
import sqlite3
from typing import Any, Optional
import uuid
from gymnasium import Env, spaces
import numpy as np
import time
from collections import defaultdict, deque
import io, os
import random

from pathlib import Path
import mediapy as media

from pokegym.pyboy_binding import (
    ACTIONS,
    make_env,
    open_state_file,
    load_pyboy_state,
    run_action_on_emulator,
)
from pokegym import data, ram_map
from .classes.gym_manager import Gym
from .classes.story_manager import Story
from .classes.events import (EventFlags, EVENTS)


STATE_PATH = __file__.rstrip("environment.py") + "States/"
GLITCH = __file__.rstrip("environment.py") + "glitch/"
CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]
db_name = Path(f'{str(uuid.uuid4())[:4]}')

class Environment:
    counter_lock = multiprocessing.Lock()
    counter = multiprocessing.Value('i', 0)
    def __init__(self, env_config, rom_path="pokemon_red.gb", state_path=None, headless=True, quiet=False, verbose=False, **kwargs,):
        with Environment.counter_lock:
            env_id = Environment.counter.value
            Environment.counter.value += 1
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
        h_w, w_w = window_size[0], window_size[1]
        h_w, w_w = window_size[0] // 2, window_size[1] // 2

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
        if self.extra_obs:
            self.observation_space = spaces.Dict(
                {
                    "screen": spaces.Box(low=0, high=255, shape=self.obs_size, dtype=np.uint8),
                    "fixed_window": spaces.Box(low=0, high=255, shape=(72,80,1), dtype=np.uint8),
                    "map_n": spaces.Box(low=0, high=250, shape=(1,), dtype=np.uint8),
                    "x": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                    "y": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                    "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
                    "flute": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                    "bike": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                    "hideout": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                    "tower": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                    "silphco": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                    "snorlax_12": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                    "snorlax_16": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                })
        else:
            self.observation_space = spaces.Dict(
                {
                    "screen": spaces.Box(low=0, high=255, shape=self.obs_size, dtype=np.uint8),
                    "fixed_window": spaces.Box(low=0, high=255, shape=(72,80,1), dtype=np.uint8),
                })

    def _get_obs(self):
        c, r, map_n = self.coords
        mmap = self.screen_memory[map_n]
        if 0 <= r <= 254 and 0 <= c <= 254:
            mmap[r, c] = 255
        if self.extra_obs:
            return {
                "screen": self.render(),
                "fixed_window": self.get_fixed_window(mmap, r, c, (72,80,1).shape),
                "map_n": np.array(map_n, dtype=np.uint8),
                "x": np.array(c, dtype=np.uint8),
                "y": np.array(r, dtype=np.uint8),
                "direction": np.array(self.game.memory[0xC109] // 4, dtype=np.uint8),
                "flute": np.array(ram_map.read_bit(self.game, 0xD76C, 0), dtype=np.uint8),
                "bike": np.array(ram_map.read_bit(self.game, 0xD75F, 0), dtype=np.uint8),
                "hideout": np.array(ram_map.read_bit(self.game, 0xD81B, 7), dtype=np.uint8),
                "tower": np.array(ram_map.read_bit(self.game, 0xD7E0, 7), dtype=np.uint8),
                "silphco": np.array(ram_map.read_bit(self.game, 0xD838, 7), dtype=np.uint8),
                "snorlax_12": np.array(ram_map.read_bit(self.game, 0xD7D8, 7), dtype=np.uint8),
                "snorlax_16": np.array(ram_map.read_bit(self.game, 0xD7E0, 1), dtype=np.uint8),
            }
        else:
            return {
                "screen": self.render(),
                "fixed_window": self.get_fixed_window(mmap, r, c, self.observation_space['screen'].shape),
            }

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        self.reset_count += 1
        self.reset_state()
        self.reset_var()
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

        cursor.execute("CREATE TABLE IF NOT EXISTS environment (env_id TEXT PRIMARY KEY,hm_count INTEGER,cut INTEGER)")
        cursor.execute("INSERT OR REPLACE INTO environment VALUES (?, ?, ?)", (str(self.env_id), self.hm_count, self.cut))

        conn.commit()
        conn.close()

    def read_database(self):
        db_dir = self.db_path
        conn = sqlite3.connect(f'{db_dir}/{db_name}.db')
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM environment WHERE cut = 1")
        count_cut_1 = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM environment")
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

    # def update_pokedex(self):
    #     for i in range(0xD30A - 0xD2F7):
    #         caught_mem = self.game.memory[i + 0xD2F7]
    #         seen_mem = self.game.memory[i + 0xD30A]
    #         for j in range(8):
    #             self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
    #             self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0  

    def update_pokedex(self):
        num_entries = 0xD30A - 0xD2F7
        caught_mem = np.array(self.game.memory[0xD2F7:0xD2F7 + num_entries], dtype=np.uint8)
        seen_mem = np.array(self.game.memory[0xD30A:0xD30A + num_entries], dtype=np.uint8)
        self.caught_pokemon = np.unpackbits(caught_mem).astype(np.uint8)
        self.seen_pokemon = np.unpackbits(seen_mem).astype(np.uint8)
    
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
        for event in data.required_events:
            events_done = ram_map.read_bit(self.game, event[0], event[1])
            if events_done:
                events_return.append(1)
        return(events_return)
    
    def update_state(self, state: bytes):
        self.reset(seed=random.randint(0, 10), options={"state": state})

    def glitch_state(self):
        saved = open(f"{GLITCH}glitch_{self.reset_count}_{self.env_id}.state", "wb")
        self.game.save_state(saved)
    
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
        # healing_reward = self.heal_rew()
        if self.time % 2 == 0:
            if self.new_events:
                events = [self.events.get_event(event) for event in EVENTS]
                self.event_reward = sum(events)*3
            else:
                ram_events = [
                    ram_map.silph_co(self.game), ram_map.rock_tunnel(self.game), ram_map.ssanne(self.game), 
                    ram_map.mtmoon(self.game), ram_map.routes(self.game), ram_map.misc(self.game), 
                    ram_map.snorlax(self.game), ram_map.hmtm(self.game), ram_map.bill(self.game), 
                    ram_map.oak(self.game), ram_map.towns(self.game), ram_map.lab(self.game), 
                    ram_map.mansion(self.game), ram_map.safari(self.game), ram_map.dojo(self.game), 
                    ram_map.hideout(self.game), ram_map.poke_tower(self.game), ram_map.gym1(self.game), 
                    ram_map.gym2(self.game), ram_map.gym3(self.game), ram_map.gym4(self.game), 
                    ram_map.gym5(self.game), ram_map.gym6(self.game), ram_map.gym7(self.game), 
                    ram_map.gym8(self.game), ram_map.rival(self.game)
                ]
                self.event_reward = sum(ram_events)

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
            # + healing_reward
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
    
    # def reward_sum(self):
    #     exploration_reward = self.expl_rew()
    #     level_reward = self.level_rew()
    #     # healing_reward = self.heal_rew()
    #     if self.time % 2 == 0:
    #         if self.new_events:
    #             events = (self.events.get_event(event) for event in EVENTS)
    #             self.event_reward = sum(events) * 3
    #         else:
    #             ram_events = [
    #                 ram_map.silph_co(self.game), ram_map.rock_tunnel(self.game), ram_map.ssanne(self.game), 
    #                 ram_map.mtmoon(self.game), ram_map.routes(self.game), ram_map.misc(self.game), 
    #                 ram_map.snorlax(self.game), ram_map.hmtm(self.game), ram_map.bill(self.game), 
    #                 ram_map.oak(self.game), ram_map.towns(self.game), ram_map.lab(self.game), 
    #                 ram_map.mansion(self.game), ram_map.safari(self.game), ram_map.dojo(self.game), 
    #                 ram_map.hideout(self.game), ram_map.poke_tower(self.game), ram_map.gym1(self.game), 
    #                 ram_map.gym2(self.game), ram_map.gym3(self.game), ram_map.gym4(self.game), 
    #                 ram_map.gym5(self.game), ram_map.gym6(self.game), ram_map.gym7(self.game), 
    #                 ram_map.gym8(self.game), ram_map.rival(self.game)
    #             ]
    #             self.event_reward = sum(ram_events)

    #     self.cut_reward = self.cut * 10
    #     sum_pokemon_reward = [self.seen_pokemon, self.caught_pokemon,]
    #     self.sum_pokemon_reward = sum(sum_pokemon_reward) * self.reward_scale
    #     cut_menu_rew = [self.seen_pokemon_menu, self.seen_stats_menu, self.seen_bag_menu]
    #     self.cut_menu_rew = sum(cut_menu_rew) * 0.1
    #     self.used_cut_rew = self.used_cut * 0.1
    #     moves_obtained_rew = sum(self.moves_obtained.values()) * self.reward_scale
    #     self.cut_coords_reward = sum(self.cut_coords.values())
    #     self.cut_tiles_reward = len(self.cut_tiles)
    #     start_menu = self.seen_start_menu * 0.01
    #     that_guy = (start_menu + self.cut_menu_rew ) / 2
    #     self.reward_sum_calc = (
    #         moves_obtained_rew
    #         + level_reward
    #         # + healing_reward
    #         + exploration_reward 
    #         + self.cut_reward
    #         + self.event_reward     
    #         + self.sum_pokemon_reward
    #         + self.used_cut_rew
    #         + self.cut_coords_reward
    #         + self.cut_tiles_reward
    #         + that_guy
    #     )
    #     return self.reward_sum_calc
    
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

    def close(self):
        self.game.stop(False)

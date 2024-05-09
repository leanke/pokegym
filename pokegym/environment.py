import multiprocessing
from pathlib import Path
from pdb import set_trace as T
import sqlite3
import uuid
from gymnasium import spaces
import numpy as np

from collections import defaultdict, deque
import io, os
import random

import matplotlib.pyplot as plt
from pathlib import Path
import mediapy as media

from pokegym.pyboy_binding import (
    ACTIONS,
    make_env,
    open_state_file,
    load_pyboy_state,
    run_action_on_emulator,
)
from pokegym import ram_map, data
from .classes.gym_manager import Gym
from .classes.story_manager import Story
from .classes.event_manager import Event


STATE_PATH = __file__.rstrip("environment.py") + "States/bulba/"
GLITCH = __file__.rstrip("environment.py") + "glitch/"
CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]
def get_random_state():
    state_files = [f for f in os.listdir(STATE_PATH) if f.endswith(".state")]
    if not state_files:
        raise FileNotFoundError("No State files found in the specified directory.")
    return random.choice(state_files)
state_file = get_random_state()
randstate = os.path.join(STATE_PATH, state_file)
db_name = Path(f'{str(uuid.uuid4())[:4]}')

class Base:
    counter_lock = multiprocessing.Lock()
    counter = multiprocessing.Value('i', 1)

    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None,
        headless=True,
        save_video=False,
        quiet=False,
        **kwargs,
    ):
        with Base.counter_lock:
            env_id = Base.counter.value
            Base.counter.value += 1
        self.state_file = get_random_state()
        self.randstate = os.path.join(STATE_PATH, self.state_file)
        """Creates a PokemonRed environment"""
        if state_path is None:
            state_path = STATE_PATH + "saffron.state" # STATE_PATH + "has_pokedex_nballs.state"
                # Make the environment
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=True, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.save_video = save_video
        self.headless = headless
        self.mem_padding = 2
        self.memory_shape = 80
        self.use_screen_memory = True
        self.screenshot_counter = 0
        self.env_id = env_id
        self.first = True
        self.reset_count = 0               
        self.explore_hidden_obj_weight = 1

        R, C = self.screen.raw_screen_buffer_dims()
        self.obs_size = (R // 2, C // 2) # 72, 80, 3

        if self.use_screen_memory:
            self.screen_memory = defaultdict(
                lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            )
            self.obs_size += (4,)
        else:
            self.obs_size += (3,)
        self.observation_space = spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=self.obs_size
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

    def save_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.game.save_state(state)
        self.initial_states.append(state)

    def glitch_state(self):
        saved = open(f"{GLITCH}glitch_{self.reset_count}_{self.env_id}.state", "wb")
        self.game.save_state(saved)
        party = data.logs(self.game)
        with open(f"{GLITCH}log_{self.reset_count}_{self.env_id}.txt", 'w') as log:
            log.write(party)
    
    def load_last_state(self):
        return self.initial_states[len(self.initial_states) - 1]
    
    def load_first_state(self):
        return self.initial_states[0]
    
    def load_random_state(self):
        rand_idx = random.randint(0, len(self.initial_states) - 1)
        return self.initial_states[rand_idx]

    def reset(self, seed=None, options=None):
        """Resets the game. Seeding is NOT supported"""
        return self.screen.screen_ndarray(), {}

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
        if self.use_screen_memory:
            r, c, map_n = ram_map.position(self.game)
            # Update tile map
            mmap = self.screen_memory[map_n]
            if 0 <= r <= 254 and 0 <= c <= 254:
                mmap[r, c] = 255

            # Downsamples the screen and retrieves a fixed window from mmap,
            # then concatenates along the 3rd-dimensional axis (image channel)
            return np.concatenate(
                (
                    self.screen.screen_ndarray()[::2, ::2],
                    self.get_fixed_window(mmap, r, c, self.observation_space.shape),
                ),
                axis=2,
            )
        else:
            return self.screen.screen_ndarray()[::2, ::2]

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        return self.render(), 0, False, False, {}
        
    def video(self):
        video = self.screen.screen_ndarray()
        return video

    def close(self):
        self.game.stop(False)

class Environment(Base):
    def __init__(self,rom_path="pokemon_red.gb",state_path=None,headless=True,save_video=False,quiet=False,verbose=False,**kwargs,):
        super().__init__(rom_path, state_path, headless, save_video, quiet, **kwargs)
        load_pyboy_state(self.game, self.load_last_state())

        self.counts_map = np.zeros((444, 436))
        self.verbose = verbose
        self.is_dead = False
        self.gym = Gym(self.game)
        self.story = Story(self.game)
        self.event = Event(self.game)

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

    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.game.get_memory_value(i + 0xD2F7)
            seen_mem = self.game.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0  
    
    def update_moves_obtained(self):
        # Scan party
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
            if self.game.get_memory_value(i) != 0:
                for j in range(4):
                    move_id = self.game.get_memory_value(i + j + 8)
                    if move_id != 0:
                        if move_id != 0:
                            self.moves_obtained[move_id] = 1
                        if move_id == 15:
                            self.cut = 1
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.game.get_memory_value(0xda80)):
            offset = i*box_struct_length + 0xda96
            if self.game.get_memory_value(offset) != 0:
                for j in range(4):
                    move_id = self.game.get_memory_value(offset + j + 8)
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1
            
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.video())

    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
        """Resets the game. Seeding is NOT supported"""
        self.reset_count += 1
        
        if self.save_video:
            base_dir = self.s_path
            base_dir.mkdir(parents=True, exist_ok=True)
            full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()

        if self.use_screen_memory:
            self.screen_memory = defaultdict(
                lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            )

        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.last_reward = None

        self.max_level_sum = 0
        self.seen_coords = set()
        self.total_healing = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.hm_count = 0
        self.cut = 0
        self.cut_coords = {}
        self.cut_tiles = {} # set([])
        self.cut_state = deque(maxlen=3)
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = {} # np.zeros(255, dtype=np.uint8)
        self.used_cut = 0
        self.death_count = 0
        self.expl_high_map = set()
        self.expl_low_map = set()

        return self.render(), {}

    def step(self, action, fast_video=True):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
        self.time += 1

        if self.save_video:
            self.add_video_frame()
        
        # Gym
        self.gym.update()
        high_gym_maps, low_gym_maps = self.gym.maps()
        gym_rew = self.gym.rew_sum
        print(f'Low Gym: {low_gym_maps}\n High Gym: {high_gym_maps}')
        print(f'Gym Rew: {gym_rew}')

        # Event
        self.event.update()
        event_rew = self.event.rew_sum
        print(f'Events Rew: {event_rew}')
        
        # Story
        self.story.update()
        high_story_maps, low_story_maps = self.story.maps()
        print(f'Low Story: {low_story_maps}\n High Story: {high_story_maps}')

        # New Exploration
        self.expl_high_map = high_gym_maps + high_story_maps
        self.expl_low_map = low_gym_maps + low_story_maps
        r, c, map_n = ram_map.position(self.game) # this is [y, x, z]
        self.seen_coords.add((r, c, map_n))
        if map_n in self.expl_high_map:
            self.exploration_reward = (0.03 * len(self.seen_coords))
        elif map_n in self.expl_low_map:
            self.exploration_reward = (0.01 * len(self.seen_coords))
        else:
            self.exploration_reward = (0.02 * len(self.seen_coords))

        # Level reward
        party_size, party_levels = ram_map.party(self.game)
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 15 + (self.max_level_sum - 15) / 4
            
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
        
        # HM reward
        hm_count = ram_map.get_hm_count(self.game)
        if hm_count >= 1 and self.hm_count == 0:
            self.hm_count = 1
        # hm_reward = hm_count * 10
        

        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        if ram_map.mem_val(self.game, 0xD057) == 0: # is_in_battle if 1
            if self.cut == 1:
                player_direction = self.game.get_memory_value(0xC109)
                y, x, map_id = ram_map.position(self.game) # this is [y, x, z]  # x, y, map_id
                if player_direction == 0:  # down
                    coords = (x, y + 1, map_id)
                if player_direction == 4:
                    coords = (x, y - 1, map_id)
                if player_direction == 8:
                    coords = (x - 1, y, map_id)
                if player_direction == 0xC:
                    coords = (x + 1, y, map_id)
                self.cut_state.append(
                    (
                        self.game.get_memory_value(0xCFC6),
                        self.game.get_memory_value(0xCFCB),
                        self.game.get_memory_value(0xCD6A),
                        self.game.get_memory_value(0xD367),
                        self.game.get_memory_value(0xD125),
                        self.game.get_memory_value(0xCD3D),
                    )
                )
                if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                    self.cut_coords[coords] = 10 # from 14
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

        if ram_map.used_cut(self.game):
            self.used_cut += 1

        # Misc
        badges = ram_map.badges(self.game)
        self.update_pokedex()
        self.update_moves_obtained()
        

        exploration_reward = self.exploration_reward
         
        seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon)
        caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon)
        moves_obtained_reward = self.reward_scale * sum(self.moves_obtained)

        cut_rew = self.cut * 10 
        used_cut_rew = self.used_cut * 0.02
        cut_coords = sum(self.cut_coords.values()) * 1.0
        cut_tiles = len(self.cut_tiles) * 1.0

        start_menu = self.seen_start_menu * 0.01
        pokemon_menu = self.seen_pokemon_menu * 0.1
        stats_menu = self.seen_stats_menu * 0.1
        bag_menu = self.seen_bag_menu * 0.1
        that_guy = (start_menu + pokemon_menu + stats_menu + bag_menu ) / 2
    
        reward = self.reward_scale * (
            + level_reward
            + healing_reward
            + exploration_reward 
            + cut_rew
            + event_rew  
            + gym_rew   
            + seen_pokemon_reward
            + caught_pokemon_reward
            + moves_obtained_reward
            + used_cut_rew
            + cut_coords 
            + cut_tiles
            + that_guy
        )

        # Subtract previous reward
        # TODO: Don't record large cumulative rewards in the first place
        if self.last_reward is None:
            reward = 0
            self.last_reward = 0
        else:
            nxt_reward = reward
            reward -= self.last_reward
            self.last_reward = nxt_reward

        info = {}
        done = self.time >= self.max_episode_steps
        if self.save_video and done:
            self.full_frame_writer.close()
        if done:
            # self.save_to_database()
            poke = self.game.get_memory_value(0xD16B)
            level = self.game.get_memory_value(0xD18C)
            if poke == 57 and level == 0:
                self.glitch_state()
            info = {
                "Events": {
                    # "silph": silph,
                    # "rock_tunnel": rock_tunnel,
                    # "ssanne": ssanne,
                    # "mtmoon": mtmoon,
                    # "routes": routes,
                    # "misc": misc,
                    # "snorlax": snorlax,
                    # "hmtm": hmtm,
                    # "bill": bill,
                    # "oak": oak,
                    # "towns": towns,
                    # "lab": lab,
                    # "mansion": mansion,
                    # "safari": safari,
                    # "dojo": dojo,
                    # "hideout": hideout,
                    # "tower": tower,
                    # "gym1": gym1,
                    # "gym2": gym2,
                    # "gym3": gym3,
                    # "gym4": gym4,
                    # "gym5": gym5,
                    # "gym6": gym6,
                    # "gym7": gym7,
                    # "gym8": gym8,
                    # "rival": rival,
                    "lift_key": int(ram_map.read_bit(self.game, 0xD81B, 6)),
                    "beat_hideout": int(ram_map.read_bit(self.game, 0xD81B, 7)),
                    "beat_marowak": int(ram_map.read_bit(self.game, 0xD768, 7)),
                    "got_pokeflute": int(ram_map.read_bit(self.game, 0xD76C, 0)),
                    "Got_Bicycle": int(ram_map.read_bit(self.game, 0xD75F, 0)),
                    "route_12_snorlax": int(ram_map.read_bit(self.game, 0xD7D8, 7)),
                    "route_16_snorlax": int(ram_map.read_bit(self.game, 0xD7E0, 1)),
                    "Beat_Silph_Co_Giovanni": int(ram_map.read_bit(self.game, 0xD838, 7)),
                },
                "BET": {
                    "Reward_Delta": reward,
                    "Seen_Poke": seen_pokemon_reward,
                    "Caught_Poke": caught_pokemon_reward,
                    "Moves_Obtain": moves_obtained_reward,
                    # "Get_HM": hm_reward,
                    "Level": level_reward,
                    "Death": death_reward,
                    "Healing": healing_reward,
                    "Exploration": exploration_reward,
                    "Taught_Cut": cut_rew,
                    "Menuing": that_guy,
                    "Used_Cut": used_cut_rew,
                    "Cut_Coords": cut_coords,
                    "Cut_Tiles": cut_tiles,
                    # "Bulba_Check": bulba_check,
                    # "Respawn": respawn_reward
                },
                "hm_count": hm_count,
                "cut_taught": self.cut,
                "badge_1": float(badges >= 1),
                "badge_2": float(badges >= 2),
                "badge_3": float(badges >= 3),
                "badge_4": float(badges >= 4),
                "badge_5": float(badges >= 5),
                "badge_6": float(badges >= 6),
                "badge_7": float(badges >= 7),
                "badge_8": float(badges >= 8),
                "badges": float(badges),
                "party_size": party_size,
                "moves_obtained": sum(self.moves_obtained),
                # "deaths": self.death_count,
                'cut_coords': cut_coords,
                'cut_tiles': cut_tiles,
                'bag_menu': bag_menu,
                'stats_menu': stats_menu,
                'pokemon_menu': pokemon_menu,
                'start_menu': start_menu,
                'used_cut': self.used_cut,
                # "respawn_coord_len": len(self.respawn)
            }
        
        return self.render(), reward, done, done, info

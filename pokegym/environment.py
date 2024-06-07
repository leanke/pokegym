from pathlib import Path
from pdb import set_trace as T
import uuid
from gymnasium import Env, spaces
import numpy as np

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


STATE_PATH = __file__.rstrip("environment.py") + "States/"
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

class Base:
    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None,
        headless=True,
        save_video=False,
        quiet=False,
        **kwargs,
    ):
        if rom_path is None or not os.path.exists(rom_path):
            raise FileNotFoundError("No ROM file found in the specified directory.")

        self.state_file = get_random_state()
        self.randstate = os.path.join(STATE_PATH, self.state_file)
        """Creates a PokemonRed environment"""
        if state_path is None:
            state_path = STATE_PATH + "Bulbasaur.state" # STATE_PATH + "has_pokedex_nballs.state"
                # Make the environment
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=True, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.save_video = save_video
        self.headless = headless
        self.mem_padding = 2
        self.memory_shape = 80
        self.use_screen_memory = True
        self.screenshot_counter = 0
        self.env_id = Path(f'{str(uuid.uuid4())[:4]}')
        self.s_path = Path(f"videos/{self.env_id}")
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
        self.counts_map = np.zeros((444, 436))
        self.death_count = 0
        self.verbose = verbose
        self.include_conditions = []
        self.seen_maps_difference = set()
        self.current_maps = []
        self.is_dead = False
        self.last_map = -1
        self.log = True
        self.used_cut = 0
        # self.seen_coords = set()
        self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.poketower = [142, 143, 144, 145, 146, 147, 148]
        self.pokehideout = [199, 200, 201, 202, 203, 135]
        self.silphco = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        load_pyboy_state(self.game, self.load_last_state())
        
    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.game.get_memory_value(i + 0xD2F7)
            seen_mem = self.game.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0  

    def town_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.game.save_state(state)
        self.initial_states.append(state)
        return 
    
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

        self.prev_map_n = None
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0
        self.seen_coords = set()
        self.seen_maps = set()
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
        self.seen_cancel_bag_menu = 0
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = {} # np.zeros(255, dtype=np.uint8)

        return self.render(), {}

    def step(self, action, fast_video=True):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
        self.time += 1

        if self.save_video:
            self.add_video_frame()
        
        # Exploration
        r, c, map_n = ram_map.position(self.game) # this is [y, x, z]
        # Exploration reward
        self.seen_coords.add((r, c, map_n))
        if int(ram_map.read_bit(self.game, 0xD81B, 7)) == 0: # pre hideout
            if map_n in self.poketower:
                exploration_reward = 0
            elif map_n in self.pokehideout:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD7E0, 7)) == 0 and int(ram_map.read_bit(self.game, 0xD81B, 7)) == 1: # hideout done poketower not done
            if map_n in self.poketower:
                exploration_reward = (0.03 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD76C, 0)) == 0 and int(ram_map.read_bit(self.game, 0xD7E0, 7)) == 1: # tower done no flute
            if map_n == 149:
                exploration_reward = (0.03 * len(self.seen_coords))
            elif map_n in self.poketower:
                exploration_reward = (0.01 * len(self.seen_coords))
            elif map_n in self.pokehideout:
                exploration_reward = (0.01 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD838, 7)) == 0 and int(ram_map.read_bit(self.game, 0xD76C, 0)) == 1: # flute gotten pre silphco
            if map_n in self.silphco:
                exploration_reward = (0.03 * len(self.seen_coords))
            elif map_n in self.poketower:
                exploration_reward = (0.01 * len(self.seen_coords))
            elif map_n in self.pokehideout:
                exploration_reward = (0.01 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD838, 7)) == 1 and int(ram_map.read_bit(self.game, 0xD76C, 0)) == 1: # flute gotten post silphco
            if map_n in self.silphco:
                exploration_reward = (0.01 * len(self.seen_coords))
            elif map_n in self.poketower:
                exploration_reward = (0.01 * len(self.seen_coords))
            elif map_n in self.pokehideout:
                exploration_reward = (0.01 * len(self.seen_coords))
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        else:
            exploration_reward = (0.02 * len(self.seen_coords))

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
                        self.game.get_memory_value(0xCFC6),
                        self.game.get_memory_value(0xCFCB),
                        self.game.get_memory_value(0xCD6A),
                        self.game.get_memory_value(0xD367),
                        self.game.get_memory_value(0xD125),
                        self.game.get_memory_value(0xCD3D),
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


        if ram_map.used_cut(self.game) == 61:
            ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write resets tile check
            self.used_cut += 1

        # Misc
        badges = ram_map.badges(self.game)
        self.update_pokedex()
        self.update_moves_obtained()
        
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

        cut_rew = self.cut * 10    
        event_reward = sum([silph, rock_tunnel, ssanne, mtmoon, routes, misc, snorlax, hmtm, bill, oak, towns, lab, mansion, safari, dojo, hideout, tower, gym1, gym2, gym3, gym4, gym5, gym6, gym7, gym8, rival])
        seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon)
        caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon)
        moves_obtained_reward = self.reward_scale * sum(self.moves_obtained)
        used_cut_rew = self.used_cut * 0.1
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
            + event_reward     
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
            poke = self.game.get_memory_value(0xD16B)
            level = self.game.get_memory_value(0xD18C)
            if poke == 57 and level == 0:
                self.glitch_state()
            info = {
                "Events": {
                    "silph": silph,
                    "rock_tunnel": rock_tunnel,
                    "ssanne": ssanne,
                    "mtmoon": mtmoon,
                    "routes": routes,
                    "misc": misc,
                    "snorlax": snorlax,
                    "hmtm": hmtm,
                    "bill": bill,
                    "oak": oak,
                    "towns": towns,
                    "lab": lab,
                    "mansion": mansion,
                    "safari": safari,
                    "dojo": dojo,
                    "hideout": hideout,
                    "tower": tower,
                    "gym1": gym1,
                    "gym2": gym2,
                    "gym3": gym3,
                    "gym4": gym4,
                    "gym5": gym5,
                    "gym6": gym6,
                    "gym7": gym7,
                    "gym8": gym8,
                    "rival": rival,
                },
                "Rewards": {
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
                "maps_explored": np.sum(self.seen_maps),
                "party_size": party_size,
                "moves_obtained": sum(self.moves_obtained),
                "deaths": self.death_count,
                'cut_coords': cut_coords,
                'cut_tiles': cut_tiles,
                'bag_menu': bag_menu,
                'stats_menu': stats_menu,
                'pokemon_menu': pokemon_menu,
                'start_menu': start_menu,
                'used_cut': self.used_cut,
            }
        
        return self.render(), reward, done, done, info

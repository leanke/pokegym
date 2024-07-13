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

class Environment:
    def __init__(self, rom_path="pokemon_red.gb", state_path=None, headless=True, save_video=False, quiet=False, verbose=False, **kwargs,):

        # Initialize emulator
        if rom_path is None or not os.path.exists(rom_path):
            raise FileNotFoundError("No ROM file found in the specified directory.")
        if state_path is None:
            state_path = STATE_PATH + "Bulbasaur.state" # STATE_PATH + "has_pokedex_nballs.state"
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=True, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.save_video = save_video
        self.headless = headless
        self.verbose = verbose

        # Policy things
        self.extra_obs = True
        R, C = self.screen.raw_screen_buffer_dims()
        self.obs_size = (R // 2, C // 2, 3) # 72, 80, 3
        self.screen_memory = defaultdict(lambda: np.zeros((255, 255, 1), dtype=np.uint8))
        self.observation_space = spaces.Dict({})
        self.obs_space()
        self.action_space = spaces.Discrete(len(ACTIONS))
        
        # Load state
        load_pyboy_state(self.game, self.load_last_state())
        self.env_id = Path(f'{str(uuid.uuid4())[:4]}')
        self.s_path = Path(f"videos/{self.env_id}")
        
        # Configs
        self.full_resets = False
        self.anneal = False
        self.manual_reset = False
        self.stream_wrapper = True
        self.max_episode_steps = 20480
        self.rew_reset = 10240

        # Misc
        self.is_dead = False
        self.used_cut = 0
        self.reset_mem = 5
        self.countdown = 10
        self.death_count = 0
        self.reset_count = 0
        self.full_reset_count = 0
        self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.poketower = [142, 143, 144, 145, 146, 147, 148]
        self.pokehideout = [199, 200, 201, 202, 203] # , 135
        self.silphco = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]

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
        return self.screen.screen_ndarray()[::2, ::2]
    
    def obs_space(self):
        if self.extra_obs:
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
                } #| {
                   # f"{event}": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8)for event in data.events_list
                    )
        else:
            self.observation_space = spaces.Dict(
                {
                    "screen": spaces.Box(low=0, high=255, shape=self.obs_size, dtype=np.uint8),
                    "fixed_window": spaces.Box(low=0, high=255, shape=(72,80,1), dtype=np.uint8),
                })

    def _get_obs(self):
        r, c, map_n = ram_map.position(self.game)
        mmap = self.screen_memory[map_n]
        if 0 <= r <= 254 and 0 <= c <= 254:
            mmap[r, c] = 255
        if self.extra_obs:
            return {
                "screen": self.render(),
                "fixed_window": self.get_fixed_window(mmap, r, c, self.observation_space['screen'].shape),
                "flute": np.array(ram_map.read_bit(self.game, 0xD76C, 0), dtype=np.uint8),
                "bike": np.array(ram_map.read_bit(self.game, 0xD75F, 0), dtype=np.uint8),
                "hideout": np.array(ram_map.read_bit(self.game, 0xD81B, 7), dtype=np.uint8),
                "tower": np.array(ram_map.read_bit(self.game, 0xD7E0, 7), dtype=np.uint8),
                "silphco": np.array(ram_map.read_bit(self.game, 0xD838, 7), dtype=np.uint8),
                "snorlax_12": np.array(ram_map.read_bit(self.game, 0xD7D8, 7), dtype=np.uint8),
                "snorlax_16": np.array(ram_map.read_bit(self.game, 0xD7E0, 1), dtype=np.uint8),
                "map_n": np.array(map_n, dtype=np.uint8),
            } #| {
            #     f"{event}": np.array(ram_map.read_bit(self.game, event[0], event[1]), dtype=np.uint8) for event in data.events_list
            # }
        else:
            return {
                "screen": self.render(),
                "fixed_window": self.get_fixed_window(mmap, r, c, self.observation_space['screen'].shape),
            }

    def reset(self, seed=None, options=None, reward_scale=4.0):
        self.reset_count += 1
        self.reset_state()

        if self.save_video:
            base_dir = self.s_path
            base_dir.mkdir(parents=True, exist_ok=True)
            full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()

        self.screen_memory = defaultdict(lambda: np.zeros((255, 255, 1), dtype=np.uint8))
        self.time = 0
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

        return self._get_obs(), {}

    def step(self, action, fast_video=True):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
        self.time += 1

        if self.manual_reset:
            if self.time == self.rew_reset:
                self.seen_coords = set()
                self.max_level_sum = 0
                self.seen_coords = set()
                self.total_healing = 0
                self.last_hp = 1.0
                self.last_party_size = 1
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

        if self.save_video:
            self.add_video_frame()
        
        # Exploration
        r, c, map_n = ram_map.position(self.game) # this is [y, x, z]
        # Exploration reward
        self.seen_coords.add((r, c, map_n))
        if int(ram_map.read_bit(self.game, 0xD77E, 1)) == 0: # pre hideout found
            if map_n in self.poketower:
                exploration_reward = 0
            elif map_n == 135:
                exploration_reward = (0.03 * len(self.seen_coords)) 
            else:
                exploration_reward = (0.02 * len(self.seen_coords))
        elif int(ram_map.read_bit(self.game, 0xD81B, 7)) == 0 and int(ram_map.read_bit(self.game, 0xD77E, 1)) == 1: # pre hideout done post found hideout
            if map_n in self.pokehideout:
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
        elif map_n == 7: # TODO: Cleanup maybe conditional on surf and teeth idk
            # if map_n == 7: # Fuchsia
            exploration_reward = (0.03 * len(self.seen_coords))
            # elif map_n in self.safari:
            #     exploration_reward = (0.03 * len(self.seen_coords))
            # else:
            #     exploration_reward = (0.02 * len(self.seen_coords))
        else:
            exploration_reward = (0.02 * len(self.seen_coords))

        # ## as more route trainer events are don reward by the length of how many are
        # ## do the same with other events such as gyms 
        # ## work on better map_n logic for late game expl
        # ## safari steps start at 502 make safari minigame
        # ##  lock agents to areas by loading states where they need to be on resets
        # ## 
        # ## 


        # # TODO: Cleanup
        # if pokeflute and not snorlax1
        #     if map_n == rt_16:
        #         extra_expl = 1
        #     else:
        #         extra_expl = 0
        # if pokeflute and not snorlax2
        #     if map_n == rt_12:
        #         extra_expl = 1
        #     else:
        #         extra_expl = 0




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


        if ram_map.used_cut(self.game) == 61 and self.cut_counter >= 5:
            ram_map.write_mem(self.game, 0xCD4D, 00) # address, byte to write resets tile check
            self.used_cut += 1
            self.cut_counter += 1

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
        moves_obtained_reward = self.reward_scale *sum(self.moves_obtained.values())
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
        
        if done:
            if self.save_video:
                self.full_frame_writer.close()
            poke = self.game.get_memory_value(0xD16B)
            level = self.game.get_memory_value(0xD18C)
            if poke == 57 and level == 0:
                self.glitch_state()
            info = self.infos_dict()
        
        return self._get_obs(), reward, done, done, info

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
                        if move_id == 15:
                            self.cut = 1
                            self.moves_obtained[move_id] = 9
                        else:
                            self.moves_obtained[move_id] = 1
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
    
    def infos_dict(self):
        return {
                "Data": {
                    "leader1": int(ram_map.read_bit(self.game, 0xD755, 7)),
                    "leader2": int(ram_map.read_bit(self.game, 0xD75E, 7)),
                    "leader3": int(ram_map.read_bit(self.game, 0xD773, 7)),
                    "leader4": int(ram_map.read_bit(self.game, 0xD792, 1)),
                    "leader5": int(ram_map.read_bit(self.game, 0xD792, 1)),
                    "leader6": int(ram_map.read_bit(self.game, 0xD7B3, 1)),
                    "leader7": int(ram_map.read_bit(self.game, 0xD79A, 1)),
                    "leader8": int(ram_map.read_bit(self.game, 0xD751, 1)),
                    "got_bike": int(ram_map.read_bit(self.game, 0xD75F, 0)),
                    "beat_hideout": int(ram_map.read_bit(self.game, 0xD81B, 7)),
                    "saved_fuji": int(ram_map.read_bit(self.game, 0xD7E0, 7)),
                    "got_flute": int(ram_map.read_bit(self.game, 0xD76C, 0)),
                    "beat_silphco": int(ram_map.read_bit(self.game, 0xD838, 7)),
                    "beat_snorlax_12": int(ram_map.read_bit(self.game, 0xD7D8, 7)),
                    "beat_snorlax_16": int(ram_map.read_bit(self.game, 0xD7E0, 1)),   
                },
                "Events": {
                    "silph": ram_map.silph_co(self.game),
                    "rock_tunnel": ram_map.rock_tunnel(self.game),
                    "ssanne": ram_map.ssanne(self.game),
                    "mtmoon": ram_map.mtmoon(self.game),
                    "routes": ram_map.routes(self.game),
                    "misc": ram_map.misc(self.game),
                    "snorlax": ram_map.snorlax(self.game),
                    "hmtm": ram_map.hmtm(self.game),
                    "bill": ram_map.bill(self.game),
                    "oak": ram_map.oak(self.game),
                    "towns": ram_map.towns(self.game),
                    "lab": ram_map.lab(self.game),
                    "mansion": ram_map.mansion(self.game),
                    "safari": ram_map.safari(self.game),
                    "dojo": ram_map.dojo(self.game),
                    "hideout": ram_map.hideout(self.game),
                    "tower": ram_map.poke_tower(self.game),
                    "gym1": ram_map.gym1(self.game),
                    "gym2": ram_map.gym2(self.game),
                    "gym3": ram_map.gym3(self.game),
                    "gym4": ram_map.gym4(self.game),
                    "gym5": ram_map.gym5(self.game),
                    "gym6": ram_map.gym6(self.game),
                    "gym7": ram_map.gym7(self.game),
                    "gym8": ram_map.gym8(self.game),
                    "rival": ram_map.rival(self.game),
                },
                "Misc": {
                    "cut": self.cut,
                    "moves_obtained": sum(self.moves_obtained),
                    "deaths": self.death_count,
                    'used_cut': self.used_cut,
                    "cut_tiles": len(self.cut_tiles),
                    "cut_coords": sum(self.cut_coords.values()),
                }
            }

    def close(self):
        self.game.stop(False)
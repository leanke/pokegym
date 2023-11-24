from pdb import set_trace as T
from gymnasium import Env, spaces
import numpy as np

from links_awaken.pyboy_binding import (ACTIONS, make_env, open_state_file,
    load_pyboy_state, run_action_on_emulator)
from links_awaken import ram_map as ram

def play():
    '''Creates an environment and plays it'''
    env = LinksAwaken(rom_path='loz.gb', state_path=None, headless=False,
        disable_input=False, sound=False, sound_emulated=False
    )
    env.reset()

    env.game.set_emulation_speed(6)
    while True:
        env.render()
        env.game.tick()

class LinksAwaken:
    def __init__(self, rom_path='loz.gb',
            state_path=None, headless=False, quiet=False,
            disable_input=True, sound=False, sound_emulated=False):
        '''Creates a LinksAwaken environment'''
        if state_path is None:
            state_path = __file__.rstrip('environment.py') + 'sword.state'

        self.game, self.screen = make_env(
            rom_path, headless, quiet,
            disable_input=disable_input,
            sound_emulated=sound_emulated,
            sound=sound,
        )
        self.initial_state = open_state_file(state_path)
        self.headless = headless

        R, C = self.screen.raw_screen_buffer_dims()
        self.observation_space = spaces.Box(
            low=0, high=255, dtype=np.uint8,
            shape=(R//2, C//2, 3),
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

    def reset(self, seed=None, options=None):
        '''Resets the game. Seeding is NOT supported'''
        load_pyboy_state(self.game, self.initial_state)
        return self.screen.screen_ndarray(), {}

    def render(self):
        return self.screen.screen_ndarray()

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        return self.render(), 0, False, False, {}

    def close(self):
        self.game.stop(False)

class LinksAwakenV1(LinksAwaken):
    def __init__(self, rom_path='loz.gb',
            state_path=None, headless=True, quiet=False):
        super().__init__(rom_path, state_path, headless, quiet)
        

    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
        '''Resets the game. Seeding is NOT supported'''
        load_pyboy_state(self.game, self.initial_state)

        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.died_count = 0
        self.money = 0
        self.map_status = 0
        self.seen_coords = set()
        self.seen_maps = set()
        self.keys = 0
        self.total_healing = 0
        self.last_health = 0
        self.last_reward = None
        self.slot_a = 0
        self.slot_b = 0
        self.held_item = 0

        return self.render()[::2, ::2], {}

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        self.time += 1

        #map status
        self.map_status = ram.map_explore(self.game)
        map_reward = self.map_status / 16

        # explore reward
        x, y  = ram.position(self.game)
        self.seen_coords.add((x, y))
        exploration_reward = 0.05 * len(self.seen_coords)

        # Healing rewards
        self.last_health = ram.hp_fraction(self.game)
        cur_health = ram.hp_fraction(self.game)
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                #self.total_healing_rew += heal_amount * 4

        #death rewards
        self.died_count = ram.death_count(self.game)
        died_count = ram.death_count(self.game)
        if died_count > self.died_count:
            self.died_count += 1


        death_reward = -0.05 * self.died_count

        # # money reward
        # money = ram.read_rupees(self.game)
        # self.money = money
        # money_reward = 0.2 * self.money

        # dungeon keys reward
        self.keys = ram.dung_keys(self.game)
        key_reward = self.keys

        # reward for held items
        slot_a, slot_b = ram.read_held_items(self.game)
        reward_a = 0  # Initialize reward_a to 0
        reward_b = 0  # Initialize reward_b to 0
        if ram.ITEMS_MAP.get(slot_a) in ['SWORD', 'SHIELD']:
            reward_a = 1
        if ram.ITEMS_MAP.get(slot_b) in ['SWORD', 'SHIELD']:
            reward_b = 1
        held_item = reward_a + reward_b 
        held_item_reward = held_item 

        #print rewards
        # print(f'Deaths:',self.died_count, 'Rupees:',self.money, 'HP:',self.last_health)

        # sum reward
        reward = self.reward_scale * (held_item_reward + death_reward + exploration_reward + key_reward + map_reward)

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
            print(f'----------reset----------')
            info = {
                'reward': {
                    'delta': reward,
                    'exploration': exploration_reward,
                    #'rupees': money_reward,
                    'dung_keys': key_reward,
                    'map_status': map_reward,
                    'held_items': held_item_reward,
                    'A_Slot': reward_a,
                    'B_Slot': reward_b
                },
                'maps_explored': self.seen_coords,
                'deaths': self.died_count,
                'money': self.money,
                'exploration': self.map_status,
                'dung_keys': self.keys
            }

        return self.render()[::2, ::2], reward, done, done, info





###########################################################################################


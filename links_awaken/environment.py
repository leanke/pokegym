from pdb import set_trace as T
from gymnasium import Env, spaces
import numpy as np

from links_awaken.pyboy_binding import (ACTIONS, make_env, open_state_file,
    load_pyboy_state, run_action_on_emulator)
from links_awaken import ram_map


class LinksAwaken:
    def __init__(self, rom_path='loz.gb',
            state_path=__file__.rstrip('environment.py') + 'loz.state',
            headless=True, quiet=False):
        self.game, self.screen = make_env(rom_path, headless, quiet)
        self.initial_state = open_state_file(state_path)
        self.headless = headless

        self.observation_space = spaces.Box(
            low=0, high=255, dtype=np.uint8,
            shape=(*self.screen.raw_screen_buffer_dims(), 3),
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
            state_path=__file__.rstrip('environment.py') + 'loz.state',
            headless=True, quiet=False):
        super().__init__(rom_path, state_path, headless, quiet)

    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
        '''Resets the game. Seeding is NOT supported'''
        load_pyboy_state(self.game, self.initial_state)

        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale

        self.money = 0
        self.map_status = 0
        self.seen_coords = set()
        self.seen_maps = set()
        self.keys = 0
        self.death_count = 0
        self.total_healing = 0
        self.last_hp_fraction = 1.0
        self.last_reward = None

        return self.render(), {}

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        self.time += 1

        #map status
        self.map_status = ram_map.map_explore(self.game)
        map_reward = self.map_status / 16

        # explore reward
        x, y  = ram_map.position(self.game)
        self.seen_coords.add((x, y))
        exploration_reward = 0.01 * len(self.seen_coords)

        # Healing rewards
        hp_fraction = ram_map.hp_fraction(self.game)
        fraction_increased = hp_fraction > self.last_hp_fraction
        if fraction_increased:
            if self.last_hp_fraction > 0:
                self.total_healing += hp_fraction - self.last_hp_fraction
        healing_reward = self.total_healing

        #death rewards
        self.death_count = ram_map.death_count(self.game)
        death_reward = -0.05 * self.death_count

        # money reward
        self.money = ram_map.read_rupees(self.game)
        money_reward = 0.2 * self.money

        # dungeon keys reward
        self.keys = ram_map.dung_keys(self.game)
        key_reward = self.keys

        # sum reward
        reward = self.reward_scale * (death_reward + exploration_reward + money_reward + key_reward + map_reward)

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
            info = {
                'reward': {
                    'delta': reward,
                    'death': death_reward,
                    'exploration': exploration_reward,
                    'rupees': money_reward,
                    'dung_keys': key_reward,
                    'map_status': map_reward
                },
                'maps_explored': self.seen_coords,
                'deaths': self.death_count,
                'money': self.money,
                'exploration': self.map_status,
                'dung_keys': self.keys
            }

        return self.render(), reward, done, done, info

from pdb import set_trace as T
from gymnasium import Env, spaces
import numpy as np

from links_awaken.pyboy_binding import (START, ACTIONS, make_env, open_state_file,
    load_pyboy_state, run_action_on_emulator)
from links_awaken import ram_map as ram

def play():
    '''Creates an environment and plays it'''
    env = LinksAwaken(rom_path='loz.gb', state_path=None, headless=False,
        disable_input=False, sound=False, sound_emulated=False
    )
    env.reset()

    env.game.set_emulation_speed(10)
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
        if state_path is None:
            state_path = __file__.rstrip('environment.py') + 'sword.state'
        

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
        self.seen_tile = set()
        self.keys = 0
        self.total_healing = 0
        self.last_health = 0
        self.last_reward = None
        self.slot_a = 0
        self.slot_b = 0
        self.held_item = 0
        self.shells = 0
        self.intro_reward = 0
        self.heal_amount = 0
        self.dest_reward = 0
        self.held_item = 0


        return self.render()[::2, ::2], {}

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        self.time += 1

        # explore reward
        x, y, tile  = ram.position(self.game)
        self.seen_coords.add((x, y, tile))
        exploration_reward = 0.05 * len(self.seen_coords)

        # Healing rewards
        cur_health = ram.hp_fraction(self.game)
        if cur_health > self.last_health:
            if self.last_health > 0:
                self.heal_amount = cur_health - self.last_health
                if self.heal_amount > 0.05:
                    print(f'healed: {self.heal_amount}')
                self.total_healing += self.heal_amount
        self.last_health = cur_health
        heal_reward = self.total_healing * 0.05

        #death rewards
        self.died_count = ram.death_count(self.game)
        died_count = ram.death_count(self.game)
        if died_count > self.died_count:
            self.died_count += 1
        death_reward = -0.05 * self.died_count

        #intro screen
        byte = ram.intro(self.game)
        if byte >= 1:
            self.intro_reward = -.5
        
        # money reward
        self.money = ram.read_rupees(self.game)
        money = ram.read_rupees(self.game)
        if money > self.money:
            self.money += 1
        money_reward = 0.02 * self.money

        # dungeon keys reward
        self.keys = ram.dung_keys(self.game)
        key_reward = self.keys

        #secret shell reward
        self.shells = ram.secret_shell(self.game)
        shells = ram.secret_shell(self.game)
        if shells > self.shells:
            self.shells += 1
        shell_reward = 0.2 * self.shells

        # reward for held items
        slot_a, slot_b = ram.read_held_items(self.game)
        reward_a = 0  # Initialize reward_a to 0
        reward_b = 0  # Initialize reward_b to 0
        if ram.ITEMS_MAP.get(slot_a) in ['SWORD']:
            reward_a = 1
        if ram.ITEMS_MAP.get(slot_b) in ['SHIELD']:
            reward_b = 1
        self.held_item = reward_a + reward_b 
        held_item_reward = self.held_item * .5

        # over world/dung reward
        dest, value = ram.dest_status(self.game)
        if value == 1: # 1 = dungeon status
            self.dest_reward += -0.01
        else:
            self.dest_reward += 0

        # sum reward
        reward = self.reward_scale * (self.dest_reward + held_item_reward + exploration_reward + self.intro_reward + heal_reward + shell_reward + money_reward + death_reward + key_reward) # + exploration_reward + held_item_reward
        reward1 = (held_item_reward + exploration_reward + heal_reward + shell_reward + money_reward + key_reward)
        neg_reward = 0.00000001 + (death_reward + self.intro_reward + self.dest_reward)

        #print rewards
        if self.headless == False:
            print(f'-------------Counter-------------')
            print(f'Steps:',self.time,)
            print(f'Sum Reward:',reward)
            print(f'Health:',self.last_health)
            print(f'Deaths:',self.died_count)
            print(f'Rupees:',self.money)
            print(f'Shells:',self.shells)
            print(f'DestID:',dest)
            print(f'-------------Rewards-------------')
            print(f'Total:',reward1)
            print(f'Explore:',exploration_reward,'--%',100 * (exploration_reward/reward1))
            print(f'Healing:',heal_reward,'--%',100 * (heal_reward/reward1))
            print(f'Shells:',shell_reward,'--%',100 * (shell_reward/reward1))
            print(f'Rupees:',money_reward,'--%',100 * (money_reward/reward1))
            print(f'Held Item:',held_item_reward,'--%',100 * (held_item_reward/reward1))
            print(f'-------------Negatives-------------')
            print(f'Total:',neg_reward)
            print(f'Intro:',self.intro_reward,'--%',100 * (self.intro_reward/neg_reward))
            print(f'Dest_status:',self.dest_reward,'--%',100 * (self.dest_reward/neg_reward))
            print(f'Deaths:',death_reward, '--%', 100 * (death_reward/neg_reward))
            # print(f'-------------Test-------------')
            # print(f'Last Health:',self.last_health)
            # print(f'Current Health:',cur_health)
            # print(f'Heal Amount',self.heal_amount)

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
        done = byte >= 1 or self.time >= self.max_episode_steps
        if done:
            print(f'----------reset----------')
            info = {
                'reward': {
                    'delta': reward,
                    'exploration': exploration_reward,
                    'rupees': money_reward,
                    'shells': shell_reward,
                    'dung_keys': key_reward,
                    'map_addr': self.dest_reward,
                    'held_items': held_item_reward,
                    'deaths': death_reward,
                    'healing': self.total_healing
                },
                'maps_explored': self.seen_tile,
                'deaths': self.died_count,
                'money': self.money,
                'exploration': self.seen_coords,
                'dung_keys': self.keys,
                'shells': self.shells,
                'intro': self.intro_reward
            }

        return self.render()[::2, ::2], reward, done, done, info





###########################################################################################


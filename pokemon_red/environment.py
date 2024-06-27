from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment

import pufferlib.emulation
import pufferlib.postprocess
from .stream_wrapper import StreamWrapper

def env_creator(name='pokemon_red'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None):
    '''Pokemon Red'''
    env = Environment(headless=headless, state_path=state_path)
    env = StreamWrapper(env, stream_metadata = { # stream_metadata is optional
            "user": f"username\n", # your username
            "color": "", # color for your text :)
            "extra": "", # any extra text you put here will be displayed
        }
    )
    env = RenderWrapper(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class RenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        self.env = env

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()

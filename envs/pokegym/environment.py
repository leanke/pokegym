from pdb import set_trace as T

import gymnasium
import functools

# import pokegym
from .pokegym import Pokegym
from .stream_wrapper import StreamWrapper

import pufferlib
import pufferlib.emulation


def env_creator(name='pokemon_red', env_config=None):
    return functools.partial(make, name, env_config)

def make(name, env_config, headless: bool = True, state_path=None, buf=None):
    '''Pokemon Red'''
    env = Pokegym(env_config, headless=headless, state_path=state_path)
    env = StreamWrapper(env, stream_metadata = {"user": f"leanke@dev_test\n",})
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class RenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        self.env = env

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()

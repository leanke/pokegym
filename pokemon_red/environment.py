from pdb import set_trace as T

import gymnasium
import functools
import uuid

from pokegym import Environment
import pufferlib.emulation
from .stream_wrapper import StreamWrapper



def env_creator(name='pokemon_red'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None):
    '''Pokemon Red'''
    env = Environment(headless=headless, state_path=state_path)
    env = StreamWrapper(env, stream_metadata = { # stream_metadata is optional
                "user": f"-TESTING-\n", # your username
                "color": "", # color for your text :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)

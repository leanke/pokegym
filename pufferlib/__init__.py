__version__ = '2.0.6'

import os
import sys

# Silence noisy dependencies
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Silence noisy packages
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
try:
    import gymnasium
    import pygame
except ImportError:
    pass
sys.stdout.close()
sys.stderr.close()
sys.stdout = original_stdout
sys.stderr = original_stderr

from pufferlib.pufferlib import *
from pufferlib import environments

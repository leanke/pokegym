#TODO:
# --no-build-isolation for 5090
# Make c and torch compile at the same time

from setuptools import find_packages, find_namespace_packages, setup, Extension
from Cython.Build import cythonize
import numpy
import os
import glob
import urllib.request
import zipfile
import tarfile
import platform
import shutil

from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

	
VERSION = "2.0.6"

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"

# Put full paths to Cython extension here
# Note we are trying to move away from Cython,
# because our C envs are lighter weigh and
# easier to debug (you can run gdb --args python ...)
cython_extension_paths = [
    # 'pong/cy_pong'
]

# Build raylib for your platform
RAYLIB_BASE = 'https://github.com/raysan5/raylib/releases/download/5.5/'
RAYLIB_NAME = 'raylib-5.5_macos' if platform.system() == "Darwin" else 'raylib-5.5_linux_amd64'
RLIGHTS_URL = 'https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h'

def download_raylib(platform, url):
    if not os.path.exists(platform):
        urllib.request.urlretrieve(url, platform + '.tar.gz')
        with tarfile.open(platform + '.tar.gz', 'r') as tar_ref:
            tar_ref.extractall()

        os.remove(platform + '.tar.gz')
        urllib.request.urlretrieve(RLIGHTS_URL, platform + '/include/rlights.h')


# RAYLIB_WASM = 'raylib-5.5_webassembly'
# RAYLIB_WASM_URL = RAYLIB_BASE + RAYLIB_WASM + '.zip'
# download_raylib(RAYLIB_WASM, RAYLIB_WASM_URL)

# Shared compile args for all platforms
extra_compile_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-DPLATFORM_DESKTOP',
]
extra_link_args = [
    '-fwrapv'
]
cxx_args = [
    '-fdiagnostics-color=always',
]
nvcc_args = []

if DEBUG:
    extra_compile_args += [
        '-O0',
        '-g',
        '-fsanitize=address,undefined,bounds,pointer-overflow,leak',
    ]
    extra_link_args += [
        '-g',
    ]
    cxx_args += [
        '-O0',
        '-g',
    ]
    nvcc_args += [
        '-O0',
        '-g',
    ]
else:
    extra_compile_args += [
        '-O2',
    ]
    extra_link_args += [
        '-O2',
    ]
    cxx_args += [
        '-O3',
    ]
    nvcc_args += [
        '-O3',
    ]

system = platform.system()
if system == 'Linux':
    extra_compile_args += [
        '-Wno-alloc-size-larger-than',
        '-fmax-errors=3',
    ]
    extra_link_args += [
        '-Bsymbolic-functions',
    ]
    RAYLIB_LINUX = 'raylib-5.5_linux_amd64'
    RAYLIB_LINUX_URL = RAYLIB_BASE + RAYLIB_LINUX + '.tar.gz'
    download_raylib(RAYLIB_LINUX, RAYLIB_LINUX_URL)
elif system == 'Darwin':
    extra_compile_args += [
    ]
    extra_link_args += [
        '-framework', 'Cocoa',
        '-framework', 'OpenGL',
        '-framework', 'IOKit',
    ]
    RAYLIB_MACOS = 'raylib-5.5_macos'
    RAYLIB_MACOS_URL = RAYLIB_BASE + RAYLIB_MACOS + '.tar.gz'
    download_raylib(RAYLIB_MACOS, RAYLIB_MACOS_URL)
else:
    raise ValueError(f'Unsupported system: {system}')

# Default Gym/Gymnasium/PettingZoo versions
# Gym:
# - 0.26 still has deprecation warnings and is the last version of the package
# - 0.25 adds a breaking API change to reset, step, and render_modes
# - 0.24 is broken
# - 0.22-0.23 triggers deprecation warnings by calling its own functions
# - 0.21 is the most stable version
# - <= 0.20 is missing dict methods for gym.spaces.Dict
# - 0.18-0.21 require setuptools<=65.5.0

GYMNASIUM_VERSION = '0.29.1'
GYM_VERSION = '0.23'
PETTINGZOO_VERSION = '1.24.1'


# Extensions 
class BuildExt(build_ext):
    def run(self):
        self.run_command('build_torch')
        self.run_command('build_c')

class CBuildExt(build_ext):
    def run(self):
        self.extensions = [e for e in self.extensions if e.name != "pufferlib._C"]
        super().run()

class TorchBuildExt(cpp_extension.BuildExtension):
    def run(self):
        self.extensions = [e for e in self.extensions if e.name == "pufferlib._C"]
        super().run()

RAYLIB_A = f'{RAYLIB_NAME}/lib/libraylib.a'
INCLUDE = [numpy.get_include(), 'raylib/include']
extension_kwargs = dict(
    include_dirs=INCLUDE,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=[RAYLIB_A],
)

# Put C env names here. PufferLib will look for
# pufferlib/ocean/<name>/binding.c
c_extensions_names = [
    'pong',
]

# TODO: Include other C files so rebuild is auto?
c_extensions = [
    Extension(
        path.rstrip('.c').replace('/', '.'),
        sources=[path],
        **extension_kwargs,
    )
    for path in glob.glob('**/binding.c', recursive=True)
]
cython_extensions = cythonize([
    Extension(
        path.replace('/', '.'),
        [path + '.pyx'],
        **extension_kwargs,
    )
    for path in cython_extension_paths
])

# Check if CUDA compiler is available. You need cuda dev, not just runtime.
if shutil.which("nvcc"):
    extension = CUDAExtension
else:
    extension = CppExtension

torch_extensions = [
   extension(
        "pufferlib._C",
        ["pufferlib/extensions/pufferlib.cpp", "pufferlib/extensions/cuda/pufferlib.cu"],
        extra_compile_args = {
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        }
    ),
]

# Prevent Conda from injecting garbage compile flags
from distutils.sysconfig import get_config_vars
cfg_vars = get_config_vars()
for key in ('CC', 'CXX', 'LDSHARED'):
    if cfg_vars[key]:
        cfg_vars[key] = cfg_vars[key].replace('-B /root/anaconda3/compiler_compat', '')
        cfg_vars[key] = cfg_vars[key].replace('-pthread', '')
        cfg_vars[key] = cfg_vars[key].replace('-fno-strict-overflow', '')

for key, value in cfg_vars.items():
    if value and '-fno-strict-overflow' in str(value):
        cfg_vars[key] = value.replace('-fno-strict-overflow', '')

setup(
    packages=find_namespace_packages() + find_packages(),
    package_data={"pufferlib": [RAYLIB_NAME + '/lib/libraylib.a']},
    include_package_data=True,
    ext_modules = cython_extensions + c_extensions + torch_extensions,
    cmdclass={
        "build_ext": BuildExt,
        "build_torch": TorchBuildExt,
        "build_c": CBuildExt,
    },
    include_dirs=[numpy.get_include(), RAYLIB_NAME + '/include'],
    keywords=["Puffer", "AI", "RL", "Reinforcement Learning"],
    entry_points={
        'console_scripts': [
            'mini = clean_pufferl:puffer',
        ],
    },
)
#stable_baselines3
#supersuit==3.3.5
#'git+https://github.com/oxwhirl/smac.git',

#curl -L -o smac.zip https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
#unzip -P iagreetotheeula smac.zip 
#curl -L -o maps.zip https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
#unzip maps.zip && mv SMAC_Maps/ StarCraftII/Maps/

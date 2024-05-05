from setuptools import find_packages, setup
from itertools import chain


GYMNASIUM_VERSION = '0.29.1'
GYM_VERSION = '0.23'

setup(
    name="pokegym",
    description="Pokemon Red Gymnasium environment for reinforcement learning",
    long_description_content_type="text/markdown",
    version=open('pokegym/version.py').read().split()[-1].strip("'"),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy==1.23.3',
        'opencv-python==3.4.17.63',
        'cython==3.0.0',
        'websockets',
        'PyYAML',
        'filelock',
        'psutil',
        'pettingzoo',
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'einops==0.6.1',
        'matplotlib',
        'scikit-image==0.21.0',
        'pyboy<2.0.0',
        'hnswlib==0.7.0',
        'mediapy',
        'pandas==2.0.2',
        'ray[all]==2.0.0',
        'setproctitle==1.1.10',
        'service-identity==21.1.0',
        'pydantic==1.9',
        'tensorboard==2.11.2',
        'torch',
        'wandb==0.13.7',
        'psutil==5.9.5',
        'tyro',
        'pufferlib==0.7.3',
    ],
    entry_points = {
        'console_scripts': [
            'pokegym.play = pokegym.environment:play'
        ]
    },
    python_requires=">=3.8",
    license="MIT",
    # @pdubs: Put your info here
    author="Joseph Suarez",
    author_email="jsuarez@mit.edu",
    url="https://github.com/PufferAI/pokegym",
    keywords=["Pokemon", "AI", "RL"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

import configparser
import argparse
from multiprocessing import Queue
import shutil
import glob
from typing import Callable, List, Dict, Type, Any
import uuid
import ast
import os
from pdb import set_trace as T

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install

from pokegym import Environment
from wrappers.render_wrapper import RenderWrapper
from wrappers.obs_wrapper import ObsWrapper
from wrappers.stream_wrapper import StreamWrapper
from wrappers.async_io import AsyncWrapper

import pufferlib.emulation
import pufferlib.postprocess



install(show_locals=False) # Rich tracebacks

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import clean_pufferl
   
def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args['policy'])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args['train']['device'])

def init_wandb(args, name, id=None, resume=True):
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args['wandb_project'],
        entity=args['wandb_entity'],
        group=args['wandb_group'],
        allow_val_change=True,
        save_code=True,
        resume=resume,
        config=args,
        name=name,
    )
    return wandb

def train(args, make_env, policy_cls, rnn_cls, async_config, wandb,):
    if args['vec'] == 'serial':
        vec = pufferlib.vector.Serial
    elif args['vec'] == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif args['vec'] == 'ray':
        vec = pufferlib.vector.Ray
    else:
        raise ValueError(f'Invalid --vector (serial/multiprocessing/ray).')
    # T()
    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=args['env'],
        num_envs=args['train']['num_envs'],
        num_workers=args['train']['num_workers'],
        batch_size=args['train']['env_batch_size'],
        zero_copy=args['train']['zero_copy'],
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)
    train_config = pufferlib.namespace(**args['train'], env=env_name,
        exp_id=args['exp_id'] or env_name + '-' + str(uuid.uuid4())[:8])
    data = clean_pufferl.create(train_config, vecenv, policy, async_config, wandb=wandb)
    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)

    uptime = data.profile.uptime
    stats = []
    for _ in range(10): # extra data for sweeps
        stats.append(clean_pufferl.evaluate(data)[0])

    clean_pufferl.close(data)
    return stats, uptime

def env_creator(train_config: List[Dict[str, Any]], wrappers: List[Dict[str, Any]], env_config: List[Dict[str, Any]], async_config) -> Callable[[], pufferlib.emulation.GymnasiumPufferEnv]:
    def make() -> pufferlib.emulation.GymnasiumPufferEnv:
        env = Environment(env_config)
        if wrappers['obs_wrapper']:
            env = ObsWrapper(env)
        if wrappers['swarming_wrapper']:
            env = AsyncWrapper(env, async_config['send_queues'], async_config['recv_queues'])
        if wrappers['stream_wrapper']:
            env = StreamWrapper(env, stream_metadata = {"user": f"{wrappers['stream_wrapper_name']}\n",})
        env = RenderWrapper(env)
        # env = pufferlib.postprocess.EpisodeStats(env)
        return pufferlib.emulation.GymnasiumPufferEnv(env=env)
    return make

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--default-config', default='config/default.ini')
    parser.add_argument('--config', default='config/pokemon_red.ini')
    parser.add_argument('--env', '--environment', type=str,
        default='pokemon_red', help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train',
        choices='train eval evaluate sweep sweep-carbs autotune profile'.split())
    parser.add_argument('--eval-model-path', type=str, default=None)
    parser.add_argument('--baseline', action='store_true',
        help='Pretrained baseline where available')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--vec', '--vector', '--vectorization', type=str,
        default='multiprocessing', choices=['serial', 'multiprocessing', 'ray'])
    parser.add_argument('--exp-id', '--exp-name', type=str,
        default=None, help="Resume from experiment")
    parser.add_argument('--wandb-entity', type=str, default='leanke')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    args = parser.parse_known_args()[0]

    if not os.path.exists(args.default_config):
        raise Exception(f'Default config {args.default_config} not found')

    for path in glob.glob('config/**/*.ini', recursive=True):
        p = configparser.ConfigParser()
        p.read(args.default_config)
        p.read(path)
        if args.env in p['base']['env_name'].split():
            break
    else:
        raise Exception('No config for env_name {}'.format(args.env))

    for section in p.sections():
        for key in p[section]:
            argparse_key = f'--{section}.{key}'.replace('_', '-')
            parser.add_argument(argparse_key, default=p[section][key])

    parsed = parser.parse_args().__dict__
    args = {'env': {}, 'policy': {}, 'rnn': {}}
    env_name = parsed.pop('env')
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    import importlib
    num_queues = (args['train']['num_envs']+1)
    env_send_queues = [Queue() for _ in range(args['train']['num_envs'] + 1)] #  + args['train']['num_workers']
    env_recv_queues = [Queue() for _ in range(args['train']['num_envs'] + 1)]
    async_config = {"send_queues": env_send_queues,"recv_queues": env_recv_queues}
    env_module = importlib.import_module(f'policies')
    make_env = env_creator(args['train'], args['wrappers'], args['env_config'], async_config)
    policy_cls = getattr(env_module, args['base']['policy_name'])
    
    rnn_name = args['base']['rnn_name']
    rnn_cls = None
    if rnn_name is not None:
        rnn_cls = getattr(env_module, args['base']['rnn_name'])

    if args['baseline']:
        assert args['mode'] in ('train', 'eval', 'evaluate')
        args['track'] = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args['exp_id'] = f'puf-{version}-{env_name}'
        args['wandb_group'] = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args["exp_id"]}', ignore_errors=True)
        run = init_wandb(args, env_name, args['exp_id'], resume=False)
        if args['mode'] in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{env_name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args['eval_model_path'] = os.path.join(data_dir, model_file)
    if args['mode'] == 'train':
        wandb = None
        if args['track']:
            wandb = init_wandb(args, env_name, id=args['exp_id'])
        train(args, make_env, policy_cls, rnn_cls, async_config, wandb=wandb,)
    elif args['mode'] in ('eval', 'evaluate'):
        clean_pufferl.rollout(
            make_env,
            args['env'],
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            model_path=args['eval_model_path'],
            render_mode=args['render_mode'],
            device=args['train']['device'],
        )
    elif args['mode'] == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
    elif args['mode'] == 'profile':
        import cProfile
        cProfile.run('train(args, env_module, make_env)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)
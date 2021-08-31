import os
from datetime import datetime
from shutil import copy

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import A2C, PPO, DQN

from environment import make_env
from utils import load_hparams

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    print("Starting baseline training")
    params = load_hparams()
    num_workers = params['workers']

    # setup logging
    logdir = None
    if params['log_results']:
        logdir = f"{BASE_DIR}/log/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        copy(f"{BASE_DIR}/hparams.json", f"{logdir}/hparams.json")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # envs = SubprocVecEnv([make_env(params['game'], params['state'], i, None) if i > 0 else make_env(params['game'], params['state'], i, logdir) for i in range(num_workers)])
    envs = SubprocVecEnv([make_env(params['game'], params['state'], i, None) if i > 0 else make_env(params['game'], params['state'], i, logdir) for i in range(1)])

    # load, train and save an already implemented agent
    model = A2C("MlpPolicy",
                envs,
                learning_rate=params['learning_rate'],
                n_steps=params['horizon'],
                gamma=params['gae']['gamma'],
                gae_lambda=params['gae']['gae_lambda'],
                verbose=1,
                tensorboard_log=logdir)
    # model = PPO("MlpPolicy",
    #             envs,
    #             learning_rate=params['learning_rate'],
    #             n_steps=params['horizon'],
    #             gamma=params['gae']['gamma'],
    #             verbose=1,
    #             tensorboard_log=logdir)
    # model = DQN("CnnPolicy",
    #             envs,
    #             buffer_size=1000,
    #             learning_rate=params['learning_rate'],
    #             gamma=params['gae']['gamma'],
    #             verbose=1,
    #             tensorboard_log=logdir)
    model.learn(total_timesteps=params['epochs']*params['horizon']*params['iterations']*num_workers)
    model.save(f"{logdir}/a2c_SF")

import numpy as np
from PIL import Image
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from environment import make_env
from utils import load_hparams

if __name__ == '__main__':
    params = load_hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    envs = SubprocVecEnv([make_env(params['game'], params['state'], i, None) for i in range(4)])

    obs = envs.reset()
    obs = [Image.fromarray(o, 'RGB') for o in obs]

    for i in range(10):
        action = np.array([torch.randint(0, 31, (1,)).item() for _ in range(4)])
        obs, rew, done, _info = envs.step(action)

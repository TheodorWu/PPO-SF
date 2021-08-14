import retro
import torch.cuda

from model import FeatureExtractor
from utils import load_hparams
import numpy as np
from discretizer import SFDiscretizer
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm

def main():
    """This will be the main loop, creating the environment and starting the PPO algorithm"""
    print("Loading hyperparameters...")
    params = load_hparams()

    print("Accessing GPU...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    print("Starting environment...")
    env = retro.make(game=params['game'], state=params['state'])
    env = SFDiscretizer(env)

    observation_length = params['observation_length']

    print("Loading feature extractor...")
    feature_extractor = FeatureExtractor(device)

    env.reset()
    print("Starting main loop")
    done = False
    while not done:
        # choose action according to policy
        action = env.action_space.sample()

        # collect a few frames and calculate mean
        obs, rew, done, _info = env.step(action)
        obs = obs.astype('float')
        for i in range(observation_length-1):
            nobs, nrew, ndone, _info = env.step(action)
            nobs = nobs.astype('float')
            obs += nobs
            rew += nrew
            if ndone:
                done = True
                break
        obs /= observation_length
        obs = Image.fromarray(obs.astype('uint8'), 'RGB')

        features = feature_extractor.extract_features(obs)

        env.render()
        if done:
            env.reset()

    # obs.show()
    env.close()
    print("Done!")

if __name__ == "__main__":
    main()
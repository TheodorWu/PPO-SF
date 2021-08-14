import retro
from utils import load_hparams
import numpy as np
from discretizer import SFDiscretizer
import matplotlib.pyplot as plt

def main():
    """This will be the main loop, creating the environment and starting the PPO algorithm"""
    params = load_hparams()
    # env = retro.make(game='Airstriker-Genesis')
    env = retro.make(game=params['game'], state=params['state'])
    # env = SFDiscretizer(env)

    observation_length = params['observation_length']

    env.reset()
    obs, rew, done, _info = env.step(env.action_space.sample())
    while not done:
        # choose action according to policy
        action = env.action_space.sample()

        # collect a few frames and calculate mean
        obs, rew, done, _info = env.step(action)
        for i in range(observation_length-1):
            nobs, nrew, ndone, _info = env.step(action)
            obs = np.mean(np.array([obs, nobs], dtype='uint8'), axis=0, dtype='uint8')
            rew += nrew
            if ndone:
                done = True
                break

        plt.imshow(obs)
        plt.show
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
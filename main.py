from datetime import datetime
import numpy as np
import retro
import os
import torch.cuda
from torch.utils.tensorboard import SummaryWriter

from model import FeatureExtractor, Actor, Critic
from utils import load_hparams
from discretizer import SFDiscretizer
from PIL import Image

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    """This will be the main loop, creating the environment and starting the PPO algorithm"""

    print("Loading hyperparameters...")
    params = load_hparams()

    if params['log_results']:
        logdir = f"{BASE_DIR}/log/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log = SummaryWriter(logdir)

    print("Accessing GPU...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Starting environment...")
    env = retro.make(game=params['game'], state=params['state'])
    env = SFDiscretizer(env)

    observation_length = params['observation_length']
    explore_steps = params['explore_steps']

    print("Loading feature extractor...")
    feature_extractor = FeatureExtractor(device)

    print("Loading actor...")
    actor = Actor(env.action_space.n)
    actor.train()
    actor.to(device)

    print("Loading critic...")
    critic = Critic()
    critic.train()
    critic.to(device)

    print("Starting main loop")
    done = False
    global_step = 0
    obs = env.reset()
    obs = Image.fromarray(obs, 'RGB')

    actions = torch.zeros(explore_steps).to(device)
    action_dists = torch.zeros(explore_steps, env.action_space.n).to(device)
    actions_one_hot = torch.zeros(explore_steps, env.action_space.n).to(device)
    q_values = torch.zeros(explore_steps).to(device)
    masks = torch.tensor([False] * explore_steps, dtype=torch.bool).to(device)
    states = torch.zeros(explore_steps, 512).to(device)
    rewards = torch.tensor(explore_steps).to(device)

    ### Gather experience
    for step in range(explore_steps):
        # choose best action according to policy
        features = feature_extractor.extract_features(obs)
        action_dist = actor(features)
        action = torch.multinomial(action_dist, 1)  # sample action based on probabilities from softmax output of actor

        # collect a few frames and calculate mean based on action
        obs, rew, done, _info = env.step(action)
        obs = obs.astype('float')
        num_obs = 0
        for i in range(observation_length - 1):
            nobs, nrew, ndone, _info = env.step(action)
            nobs = nobs.astype('float')
            obs += nobs
            rew += nrew
            num_obs += 1
            if ndone:
                done = True
                break
        obs /= num_obs
        obs = Image.fromarray(obs.astype('uint8'), 'RGB')

        # store information gained from current step
        q_values[step] = critic(features)
        states[step] = features
        actions[step] = action
        action_dists[step] = action_dist
        actions_one_hot[step, action] = 1
        rewards[step] = rew
        masks[step] = not done

        if params['log_results']:
            log.add_scalar("Reward", rew, global_step)

        env.render()
        if done:
            env.reset()

        global_step += 1

    # obs.show()
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

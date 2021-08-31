import json
import os
import seaborn as sns

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from environment import make_env
from model import FeatureExtractor, Actor
from utils import load_hparams

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
logdir = f"{BASE_DIR}/evaluation"

def eval(actor_dir, tag):
    """Based on function collect_experience in ppo.py. Basically goes through the inner loop of ppo to interact with the environment"""
    print("Loading hyperparameters...")
    params = load_hparams()

    print("Accessing GPU...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading feature extractor...")
    feature_extractor = FeatureExtractor(device)

    env = make_env(params['game'], params['state'], 0, logdir=f"{logdir}/{tag}")()

    print("Loading actor...")
    actor = Actor(env.action_space.n)
    actor.load_state_dict(torch.load(f"{BASE_DIR}/{actor_dir}"))
    actor.to(device)
    actor.eval()

    if not os.path.exists(f"{logdir}/{tag}"):
        os.makedirs(f"{logdir}/{tag}")

    obs = env.reset()
    obs = Image.fromarray(obs, 'RGB')

    rounds = 0
    rewards = []
    with open(f"{BASE_DIR}/evaluation/cumulative_rewards.json") as f:
        cumulative_reward = json.load(f)

    cumulative_reward[tag] = [0] * 7
    crew = 0
    while rounds < 7:
        features = feature_extractor.extract_features(obs)
        action_dist = actor(features)
        action = torch.multinomial(action_dist, 1).item()  # sample action based on probabilities from softmax output of actor

        obs, rew, done, _info = env.step(action)
        obs = obs.astype('float')
        num_obs = 0
        for i in range(params['observation_length'] - 1):
            nobs, nrew, ndone, _info = env.step(action)
            nobs = nobs.astype('float')
            obs += nobs
            rew += nrew
            num_obs += 1
            done = np.logical_or(done, ndone)
            if done:
                break

        obs /= num_obs
        obs = Image.fromarray(obs, 'RGB')

        rewards.append(rew)
        crew += rew

        if done:
            cumulative_reward[tag][rounds] = crew
            crew = 0
            obs = env.reset()
            obs = Image.fromarray(obs, 'RGB')
            rounds += 1

    plt.plot(rewards)
    plt.title("Rewards of my PPO (trained on health policy) during evaluation")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.savefig(f"{logdir}/{tag}.png")
    plt.clf()

    with open('evaluation/cumulative_rewards.json', 'w') as fp:
        json.dump(cumulative_reward, fp)


def plot_cumulative_rewards():
    """Plot the cumulative rewards that are stored in the corresponding json"""
    sns.set_theme()
    with open(f"{BASE_DIR}/evaluation/cumulative_rewards.json") as f:
        c_reward = json.load(f)

    for k, v in c_reward.items():
        if k != "a2c_health":
            plt.plot(v, label=k)

    plt.title("Cumulative rewards")
    plt.rcParams["figure.figsize"] = (12, 12)
    plt.legend(loc='upper left')
    plt.savefig(f"{logdir}/cumulative_rewards.png")


if __name__ == '__main__':
    # eval("log/20210826-180535/model/actor.pt", "my_ppo_health")
    # eval("log/20210820-233301/model/actor.pt", "my_ppo_score")
    # eval("log/20210829-234248/model/actor.pt", "my_ppo_health_2")
    # eval("log/20210830-084923/model/actor.pt", "my_ppo_score_2")
    eval("log/20210830-131326/model/actor.pt", "my_ppo_score_3")

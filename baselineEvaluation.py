import json

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from environment import make_env
from utils import load_hparams
from stable_baselines3 import A2C, PPO, DQN

# TODO: Refactor this file

if __name__ == '__main__':
    sns.set_theme()
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    logdir = f"{BASE_DIR}/evaluation"
    params = load_hparams()
    cumulative_reward = {}

    print("Evaluating baseline A2C")
    print("Policy: based on health only")
    if not os.path.exists(f"{logdir}/a2c_health"):
        os.makedirs(f"{logdir}/a2c_health")
    env = make_env(params['game'], params['state'], 0, logdir=f"{logdir}/a2c_health")()
    model = A2C.load(f"{BASE_DIR}/log/20210826-121022/a2c_SF")

    obs = env.reset()
    rounds = 0
    rewards = []
    cumulative_reward['a2c_health'] = [0] * 7
    crew = 0
    while rounds < 7:
        action, _states = model.predict(obs)
        obs, reward, done, _info = env.step(action)

        rewards.append(reward)
        crew += reward

        if done:
            cumulative_reward['a2c_health'][rounds] = crew
            crew = 0
            obs = env.reset()
            rounds += 1

    env.close()

    plt.plot(rewards)
    plt.title("Rewards of baseline A2C (trained on health policy) during evaluation")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.savefig(f"{logdir}/a2c_health.png")
    plt.clf()

    print("Policy: score")
    if not os.path.exists(f"{logdir}/a2c_score"):
        os.makedirs(f"{logdir}/a2c_score")
    env = make_env(params['game'], params['state'], 0, logdir=f"{logdir}/a2c_score")()
    model = A2C.load(f"{BASE_DIR}/log/20210828-180514/a2c_SF")

    obs = env.reset()
    rounds = 0
    rewards = []
    cumulative_reward['a2c_score'] = [0] * 7
    crew = 0
    while rounds < 7:
        action, _states = model.predict(obs)
        obs, reward, done, _info = env.step(action)
        rewards.append(reward)
        crew += reward

        if done:
            cumulative_reward['a2c_score'][rounds] = crew
            crew = 0
            obs = env.reset()
            rounds += 1

    env.close()

    plt.plot(rewards)
    plt.title("Rewards of baseline A2C (trained on score policy) during evaluation")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.savefig(f"{logdir}/a2c_score.png")
    plt.clf()

    print("-----------------------")
    print("Evaluating baseline PPO")
    print("Policy: based on health only")
    if not os.path.exists(f"{logdir}/ppo_health"):
        os.makedirs(f"{logdir}/ppo_health")
    env = make_env(params['game'], params['state'], 0, logdir=f"{logdir}/ppo_health")()
    model = PPO.load(f"{BASE_DIR}/log/20210826-135139/ppo_SF")

    obs = env.reset()
    rounds = 0
    rewards = []
    cumulative_reward['ppo_health'] = [0] * 7
    crew = 0
    while rounds < 7:
        action, _states = model.predict(obs)
        obs, reward, done, _info = env.step(action)

        rewards.append(reward)
        crew += reward

        if done:
            cumulative_reward['ppo_health'][rounds] = crew
            crew = 0
            obs = env.reset()
            rounds += 1

    env.close()

    plt.plot(rewards)
    plt.title("Rewards of baseline PPO (trained on health policy) during evaluation")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.savefig(f"{logdir}/ppo_health.png")
    plt.clf()

    print("Policy: score")
    if not os.path.exists(f"{logdir}/ppo_score"):
        os.makedirs(f"{logdir}/ppo_score")
    env = make_env(params['game'], params['state'], 0, logdir=f"{logdir}/ppo_score")()
    model = PPO.load(f"{BASE_DIR}/log/20210827-113506/ppo_SF")

    obs = env.reset()
    rounds = 0
    rewards = []
    cumulative_reward['ppo_score'] = [0] * 7
    crew = 0
    while rounds < 7:
        action, _states = model.predict(obs)
        obs, reward, done, _info = env.step(action)
        rewards.append(reward)
        crew += reward

        if done:
            cumulative_reward['ppo_score'][rounds] = crew
            crew = 0
            obs = env.reset()
            rounds += 1

    env.close()

    plt.plot(rewards)
    plt.title("Rewards of baseline PPO (trained on score policy) during evaluation")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.savefig(f"{logdir}/ppo_score.png")
    plt.clf()

    print("-----------------------")
    print("Evaluating baseline DQN")
    print("Policy: based on health only")
    if not os.path.exists(f"{logdir}/dqn_health"):
        os.makedirs(f"{logdir}/dqn_health")
    env = make_env(params['game'], params['state'], 0, logdir=f"{logdir}/dqn_health")()
    model = DQN.load(f"{BASE_DIR}/log/20210827-212603/dqn_SF")

    obs = env.reset()
    rounds = 0
    rewards = []
    cumulative_reward['dqn_health'] = [0] * 7
    crew = 0
    while rounds < 7:
        action, _states = model.predict(obs)
        obs, reward, done, _info = env.step(action)
        rewards.append(reward)
        crew += reward

        if done:
            cumulative_reward['dqn_health'][rounds] = crew
            crew = 0
            obs = env.reset()
            rounds += 1
    env.close()

    plt.plot(rewards)
    plt.title("Rewards of baseline DQN (trained on health policy) during evaluation")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.savefig(f"{logdir}/dqn_health.png")
    plt.clf()

    print("Policy: score")
    if not os.path.exists(f"{logdir}/dqn_score"):
        os.makedirs(f"{logdir}/dqn_score")
    env = make_env(params['game'], params['state'], 0, logdir=f"{logdir}/dqn_score")()
    model = DQN.load(f"{BASE_DIR}/log/20210828-102509/dqn_SF")

    obs = env.reset()
    rounds = 0
    rewards = []
    cumulative_reward['dqn_score'] = [0] * 7
    crew = 0
    while rounds < 7:
        action, _states = model.predict(obs)
        obs, reward, done, _info = env.step(action)

        rewards.append(reward)
        crew += reward

        if done:
            cumulative_reward['dqn_score'][rounds] = crew
            crew = 0
            obs = env.reset()
            rounds += 1
    env.close()

    plt.plot(rewards)
    plt.title("Rewards of baseline DQN (trained on score policy) during evaluation")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.savefig(f"{logdir}/dqn_score.png")
    plt.clf()

    for v in cumulative_reward.values():
        plt.plot(v)

    plt.title("Cumulative rewards")
    plt.legend(loc='upper left', labels=cumulative_reward.keys())
    plt.savefig(f"{logdir}/cumulative_rewards.png")

    with open('evaluation/cumulative_rewards.json', 'w') as fp:
        json.dump(cumulative_reward, fp)

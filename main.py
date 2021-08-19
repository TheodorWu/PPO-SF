from datetime import datetime
from shutil import copy

import numpy as np
import retro
import os
import torch.cuda
import torch.multiprocessing as mp
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

from model import FeatureExtractor, Actor, Critic
from ppo import PPO, GAE
from utils import load_hparams
from discretizer import SFDiscretizer
from PIL import Image
from torch.optim import Adam


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
torch.autograd.set_detect_anomaly(True)

def main():
    """This will be the main loop, creating the environment and starting the PPO algorithm"""
    # TODO: refactoring
    print("Loading hyperparameters...")
    params = load_hparams()

    logdir = "."
    if params['log_results']:
        logdir = f"{BASE_DIR}/log/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log = SummaryWriter(logdir)
        copy(f"{BASE_DIR}/hparams.json", f"{logdir}/hparams.json")

    print("Accessing GPU...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Starting environment...")
    env = retro.make(game=params['game'], state=params['state'], record=logdir)
    env = SFDiscretizer(env)

    observation_length = params['observation_length']
    horizon = params['horizon']
    num_epochs = params['epochs']

    alpha = 1.0 # will be reduced during training, linearly annealed
    ppo = PPO(params['ppo']['epsilon'], alpha, device)
    gae = GAE(horizon, params['gae']['gamma'], params['gae']['gae_lambda'], device)

    print("Loading feature extractor...")
    feature_extractor = FeatureExtractor(device)

    print("Loading actor...")
    actor = Actor(env.action_space.n)
    actor.train()
    actor.to(device)
    actor_optim = Adam(actor.parameters(), lr=params['learning_rate'])

    print("Loading critic...")
    critic = Critic()
    critic.train()
    critic.to(device)
    critic_optim = Adam(critic.parameters(), lr=params['learning_rate'])

    print("Starting main loop")
    global_step = 0
    global_update_step = 0
    runs = 0
    cumulative_reward = 0.0
    obs = env.reset()
    obs = Image.fromarray(obs, 'RGB')

    for epoch in range(num_epochs):
        print(f"-------- Epoch {epoch} -------------")
        # init ground truth policy values before update
        batch_log_probs = torch.zeros(horizon, device=device)
        batch_advantage_estimates = torch.zeros(horizon, device=device)
        for i_update in range(params['updates_per_epoch']+1): # adding one since first run collects samples

            # init policy values for current observations
            actions = torch.zeros(horizon, device=device)
            action_dists = torch.zeros(horizon, env.action_space.n, device=device)
            log_probs = torch.zeros(horizon, device=device)
            q_values = torch.zeros(horizon, device=device)
            masks = torch.zeros(horizon, device=device)
            states = torch.zeros(horizon, 512, device=device)
            rewards = torch.zeros(horizon, device=device)

            if i_update > 0:
                print(f"Update {i_update}")
            else:
                print("Gathering comparison batch")

            ### Gather experience
            # TODO: Threading
            for step in range(horizon):
                # choose best action according to policy
                features = feature_extractor.extract_features(obs)
                action_dist = actor(features)
                action = torch.multinomial(action_dist, 1)  # sample action based on probabilities from softmax output of actor

                # collect a few frames and calculate mean based on action
                obs, rew, done, _info = env.step(action)
                rew /= 176.0
                obs = obs.astype('float')
                num_obs = 0
                for i in range(observation_length - 1):
                    nobs, nrew, ndone, _info = env.step(action)
                    nrew /= 176.
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
                rewards[step] = rew
                cumulative_reward += rew
                masks[step] = 0 if done else 1
                log_probs[step] = torch.log(action_dist[action])

                if done:
                    if params['log_results']:
                        log.add_scalar("Cumulative reward", cumulative_reward, runs)
                    runs += 1
                    env.reset()

                global_step += 1

            if i_update == 0:
                # collected first batch -> reference policy
                batch_log_probs = log_probs.detach().clone()
                batch_advantage_estimates, _ = gae.generalized_advantage_estimation(q_values, rewards, masks)
            else:
                # update step
                _, advantage_estimates = gae.generalized_advantage_estimation(q_values, rewards, masks)

                print("Updating actor")
                actor_loss = ppo.clipped_surrogate_loss(log_probs, batch_log_probs, advantage_estimates)

                actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_optim.step()

                print("Updating critic")
                critic_loss = MSELoss()(q_values, batch_advantage_estimates)

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                if params['log_results']:
                    log.add_scalar("ActorLoss", actor_loss.item(), global_update_step)
                    log.add_scalar("CriticLoss", critic_loss.item(), global_update_step)
                    log.add_scalar("MeanAdvantageEstimates", torch.mean(advantage_estimates).item(), global_update_step)
                    log.add_scalar("MeanRewards", torch.mean(rewards).item(), global_update_step)

                global_update_step += 1

        ppo.alpha -= 1.0/num_epochs

        if params['log_results']:
            print("Saving checkpoints")
            torch.save({
                'epoch': epoch,
                'model_state_dict': actor.state_dict(),
                'optimizer_state_dict': actor_optim.state_dict(),
                'loss': critic_loss,
                }, f"{logdir}/checkpoint_{epoch}/actor.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': critic.state_dict(),
                'optimizer_state_dict': critic_optim.state_dict(),
                'loss': critic_loss,
                }, f"{logdir}/checkpoint_{epoch}/critic.pt")

    print("Saving trained models")
    torch.save(actor.state_dict(), f"{logdir}/model/actor.pt")
    torch.save(critic.state_dict(), f"{logdir}/model/critic.pt")
    # obs.show()
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

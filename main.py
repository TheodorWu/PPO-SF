from datetime import datetime
from shutil import copy

import os
import torch.cuda
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

from environment import make_env
from model import FeatureExtractor, Actor, Critic
from ppo import PPO, GAE, collect_experience
from utils import load_hparams
from torch.optim import Adam
from stable_baselines3.common.vec_env import SubprocVecEnv

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
torch.autograd.set_detect_anomaly(True)



def main():
    """This will be the main loop, creating the environment and starting the PPO algorithm"""
    # TODO: refactoring
    print("Loading hyperparameters...")
    params = load_hparams()

    num_workers = params['workers']
    logdir = "."
    if params['log_results']:
        logdir = f"{BASE_DIR}/log/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log = SummaryWriter(logdir)
        copy(f"{BASE_DIR}/hparams.json", f"{logdir}/hparams.json")

    print("Accessing GPU...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Starting environments for {num_workers} workers...")
    envs = SubprocVecEnv([make_env(params['game'], params['state'], i, None) if i > 0 else make_env(params['game'], params['state'], i, logdir) for i in range(num_workers)])

    observation_length = params['observation_length']
    horizon = params['horizon']
    num_epochs = params['epochs']

    alpha = 1.0 # will be reduced during training, linearly annealed
    ppo = PPO(params['ppo']['epsilon'], alpha, device)
    gae = GAE(horizon, params['gae']['gamma'], params['gae']['gae_lambda'], device)

    print("Loading feature extractor...")
    feature_extractor = FeatureExtractor(device)

    print("Loading actor...")
    actor = Actor(envs.action_space.n)
    actor.train()
    actor.to(device)
    actor_optim = Adam(actor.parameters(), lr=params['learning_rate'])

    print("Loading critic...")
    critic = Critic()
    critic.train()
    critic.to(device)
    critic_optim = Adam(critic.parameters(), lr=params['learning_rate'])

    print("Starting main loop")
    global_update_step = 0

    for epoch in range(num_epochs):
        print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] -------- Epoch {epoch} -------------")
        # init ground truth policy values before update
        batch_log_probs = torch.zeros(horizon, num_workers, device=device)
        batch_advantage_estimates = torch.zeros(horizon, num_workers, device=device)
        for i_update in range(params['updates_per_epoch']+1): # adding one since first run collects samples
            if i_update > 0:
                print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] Update {i_update}")
            else:
                print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}]Gathering comparison batch")

            ### Gather experience
            log_probs, q_values, rewards, masks, crew = collect_experience(envs, feature_extractor, actor, critic, horizon, observation_length, num_workers, device)
            cumulative_reward = torch.mean(crew)

            if i_update == 0:
                # collected first batch -> reference policy
                batch_log_probs = torch.mean(log_probs, dim=0).detach().clone()
                batch_advantage_estimates, _ = gae.generalized_advantage_estimation(torch.mean(q_values, dim=0), torch.mean(rewards, dim=0), torch.mean(masks, dim=0))
            else:
                actor_loss = 0
                critic_loss = 0
                for i in range(num_workers):
                    # update step
                    _, advantage_estimates = gae.generalized_advantage_estimation(q_values[i], rewards[i], masks[i])

                    actor_loss += ppo.clipped_surrogate_loss(log_probs[i], batch_log_probs, advantage_estimates)

                    critic_loss += MSELoss()(q_values[i], batch_advantage_estimates)

                print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] Updating actor")
                actor_loss = actor_loss / num_workers
                print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] Actor loss {actor_loss.item()}")
                actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_optim.step()

                print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] Updating critic")
                critic_loss = critic_loss / num_workers
                print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] Critic loss {critic_loss.item()}")

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                if params['log_results']:
                    log.add_scalar("ActorLoss", actor_loss.item(), global_update_step)
                    log.add_scalar("CriticLoss", critic_loss.item(), global_update_step)
                    log.add_scalar("MeanAdvantageEstimates", torch.mean(advantage_estimates).item(), global_update_step)
                    log.add_scalar("MeanRewards", torch.mean(rewards[i]).item(), global_update_step)
                    log.add_scalar("MeanCumulativeReward", cumulative_reward.item(), global_update_step)

                    global_update_step += 1

        ppo.alpha -= 1.0/num_epochs

        if params['log_results']:
            print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] Saving checkpoints")
            if not os.path.exists(f"{logdir}/checkpoint_{epoch}"):
                os.makedirs(f"{logdir}/checkpoint_{epoch}")
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

    print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] Saving trained models")
    if not os.path.exists(f"{logdir}/model"):
        os.makedirs(f"{logdir}/model")
    torch.save(actor.state_dict(), f"{logdir}/model/actor.pt")
    torch.save(critic.state_dict(), f"{logdir}/model/critic.pt")
    # obs.show()
    envs.close()
    print(f"[{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}] Done!")


if __name__ == "__main__":
    main()

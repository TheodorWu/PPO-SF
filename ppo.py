import torch
import numpy as np
from PIL import Image


class GAE:
    """Generalized Advantage Estimation"""
    def __init__(self, horizon, gamma, gae_lambda, device):
        """Stores the relevant parameters on initialization"""
        self.horizon = horizon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def generalized_advantage_estimation(self, q_values, rewards, masks):
        """Computes the estimate of the generalized advantage. The output is a tuple containing the not normalized and normalized advantage estimates."""
        advantage_estimates = np.zeros(self.horizon, dtype='float32')
        advantage_estimates[self.horizon-1] = rewards[self.horizon-1] - q_values[self.horizon-1]
        for i in range(self.horizon-2, -1, -1):
            # masks excludes q_values after done condition was reached because next state would be after reset
            delta = rewards[i] + self.gamma * q_values[i + 1] * masks[i] - q_values[i]
            advantage_estimates[i] = delta + self.gamma * self.gae_lambda * advantage_estimates[i+1] * masks[i]

        normalized_estimates = (advantage_estimates - advantage_estimates.mean()) / (advantage_estimates.std() + 1e-10)
        return torch.tensor(advantage_estimates, dtype=torch.float32, device=self.device), torch.tensor(normalized_estimates, dtype=torch.float32, device=self.device)

class PPO:
    """Proximal Policy Optimization. Contained only the loss function in the end"""
    def __init__(self, epsilon, alpha, device):
        """Stores the relevant parameters on initialization"""
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device

    def clipped_surrogate_loss(self, probs, old_probs, advantage_estimates):
        """Calculates the clipped surrogate loss"""
        # expecting log probabilities as input for the next line to hold
        probability_ratio = torch.exp(probs - old_probs)
        clipped = torch.clamp(probability_ratio, 1-self.epsilon*self.alpha, 1+self.epsilon*self.alpha) * advantage_estimates
        unclipped = probability_ratio * advantage_estimates

        return torch.mean(-torch.min(unclipped, clipped))

def collect_experience(envs, feature_extractor, actor, critic, horizon, observation_length, num_workers, device):
    """Inner loop of the PPO algorithm"""
    # get first observation array (unprocessed)
    obs = envs.reset()
    obs = [Image.fromarray(o, 'RGB') for o in obs]

    # init policy values for current observations
    actions = torch.zeros(horizon, num_workers, device=device, dtype=torch.int)
    action_dists = torch.zeros(horizon, num_workers, envs.action_space.n, device=device)
    log_probs = torch.zeros(horizon, num_workers, device=device)
    q_values = torch.zeros(horizon, num_workers, device=device)
    masks = torch.zeros(horizon, num_workers, device=device)
    states = torch.zeros(horizon, num_workers, 512, device=device)
    rewards = torch.zeros(horizon, num_workers, device=device)
    cumulative_reward = torch.zeros(num_workers, device=device)

    for step in range(horizon):
        # preprocess observation
        features = [feature_extractor.extract_features(o) for o in obs]
        # choose best action according to policy
        action_dist = [actor(f) for f in features]
        action = np.array([torch.multinomial(a, 1).item() for a in action_dist])  # sample action based on probabilities from softmax output of actor

        # collect a few frames and calculate mean based on action
        obs, rew, done, _info = envs.step(action)
        obs = obs.astype('float')
        num_obs = 0

        # to include temporal knowledge in an observation, the mean is taken over multiple frames
        for i in range(observation_length - 1):  # one obs has been collected already, hence the -1
            nobs, nrew, ndone, _info = envs.step(action)
            nobs = nobs.astype('float')
            obs += nobs
            rew += nrew
            num_obs += 1
            done = np.logical_or(done, ndone)
            # if any(ndone): # not needed since they reset automatically
            #     break
        obs /= num_obs
        obs = [Image.fromarray(o, 'RGB') for o in obs]

        # store information gained from current step
        q_values[step] = torch.stack([critic(f) for f in features]).squeeze()
        states[step] = torch.stack(features)
        actions[step] = torch.from_numpy(action)
        action_dists[step] = torch.stack(action_dist)
        rewards[step] = torch.tensor(rew)
        cumulative_reward += torch.from_numpy(rew).to(device)

        # the mask ensures that the next observation after the done condition has been reached does not influence the
        # GAE since the reward then would already be from a new run.
        masks[step] = torch.tensor([0 if d else 1 for d in done])
        log_probs[step] = torch.log(torch.stack([action_dist[i][a.item()] for i, a in enumerate(action)]))

    return torch.transpose(log_probs, 0, 1), torch.transpose(q_values, 0, 1), torch.transpose(rewards, 0, 1), torch.transpose(masks, 0, 1), cumulative_reward, torch.transpose(states, 0, 1), torch.transpose(actions, 0, 1)



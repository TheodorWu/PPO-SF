import torch
import numpy as np

class GAE:
    def __init__(self, horizon, gamma, gae_lambda, device):
        self.horizon = horizon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def generalized_advantage_estimation(self, q_values, rewards, masks):
        advantage_estimates = np.zeros(self.horizon, dtype='float32')
        advantage_estimates[self.horizon-1] = rewards[self.horizon-1] - q_values[self.horizon-1]
        for i in range(self.horizon-2, -1, -1):
            # masks excludes q_values after done condition has reached because next state would be after reset
            delta = rewards[i] + self.gamma * q_values[i + 1] * masks[i] - q_values[i]
            advantage_estimates[i] = delta + self.gamma * self.gae_lambda * advantage_estimates[i+1] * masks[i]

        normalized_estimates = (advantage_estimates - advantage_estimates.mean()) / (advantage_estimates.std() + 1e-10)
        return torch.tensor(advantage_estimates, dtype=torch.float32, device=self.device), torch.tensor(normalized_estimates, dtype=torch.float32, device=self.device)

class PPO:
    def __init__(self, epsilon, alpha, device):
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device

    def clipped_surrogate_loss(self, probs, old_probs, advantage_estimates):
        # expecting log probabilities for the next line to hold
        probability_ratio = torch.exp(probs - old_probs)
        clipped = torch.clamp(probability_ratio, 1-self.epsilon*self.alpha, 1+self.epsilon*self.alpha) * advantage_estimates
        unclipped = probability_ratio * advantage_estimates

        return torch.mean(-torch.min(unclipped, clipped))




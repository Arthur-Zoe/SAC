import copy

import numpy as np
import torch
from torch.nn import functional as F

from .actor import GaussianActor
from .critic import Critic


class SACAgent:
    def __init__(
        self, obs_dim, action_dim, action_low, action_high, device="cpu",
        gamma=0.99, tau=0.005, learning_rate=3e-4, init_alpha=0.2,
        auto_alpha=True, hidden_dim=256, double_q: bool = True,
    ):
        if init_alpha <= 0:
            raise ValueError("init_alpha must be positive")
        if type(double_q) is not bool:
            raise ValueError("double_q must have type bool")
        self.double_q = double_q
        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau
        self.auto_alpha = auto_alpha
        self.actor = GaussianActor(
            obs_dim, action_dim, action_low, action_high, hidden_dim
        ).to(self.device)
        self.q1 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = Critic(obs_dim, action_dim, hidden_dim).to(self.device) if double_q else None
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False) if double_q else None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + (list(self.q2.parameters()) if double_q else []),
            lr=learning_rate
        )
        self.target_entropy = -float(action_dim)
        if auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(init_alpha), dtype=torch.float32,
                device=self.device, requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.fixed_alpha = torch.tensor(init_alpha, device=self.device)

    def get_alpha(self):
        return self.log_alpha.exp() if self.auto_alpha else self.fixed_alpha

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = (
            self.actor.deterministic_action(obs)
            if deterministic else self.actor.sample(obs)[0]
        )
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, next_obs, done):
        q1_value = self.q1(obs, action)
        q2_value = self.q2(obs, action) if self.double_q else q1_value
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            if self.double_q:
                target_min_q = torch.min(
                    self.target_q1(next_obs, next_action),
                    self.target_q2(next_obs, next_action),
                )
            else:
                target_min_q = self.target_q1(next_obs, next_action)
            # y = r + gamma (1 - terminated) [min target_Q(s', a') - alpha log pi(a'|s')]
            target_y = reward + self.gamma * (1 - done) * (
                target_min_q - self.get_alpha() * next_log_prob
            )
        if self.double_q:
            critic_loss = F.mse_loss(q1_value, target_y) + F.mse_loss(q2_value, target_y)
        else:
            critic_loss = F.mse_loss(q1_value, target_y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {
            "loss/critic": critic_loss.detach(),
            "critic/q1": q1_value.detach().mean(),
            # Preserve the metric schema; single-Q reports Q1 here as well.
            "critic/q2": q2_value.detach().mean(),
            "critic/target_q": target_y.detach().mean(),
        }

    def update_actor(self, obs):
        self.q1.requires_grad_(False)
        if self.double_q:
            self.q2.requires_grad_(False)
        try:
            new_action, log_prob = self.actor.sample(obs)
            if self.double_q:
                min_q = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
            else:
                min_q = self.q1(obs, new_action)
            # Freeze Q parameters but keep dQ/da: L_pi = E[alpha log pi - min Q].
            actor_loss = (self.get_alpha().detach() * log_prob - min_q).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.q1.requires_grad_(True)
            if self.double_q:
                self.q2.requires_grad_(True)
        return actor_loss.detach(), log_prob.detach()

    def update_alpha(self, log_prob):
        if not self.auto_alpha:
            return torch.zeros((), device=self.device)
        # Same sampled log pi as the actor update; alpha cannot backprop into the actor.
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return alpha_loss.detach()

    @torch.no_grad()
    def soft_update(self):
        pairs = ((self.q1, self.target_q1),)
        if self.double_q:
            pairs += ((self.q2, self.target_q2),)
        for critic, target in pairs:
            for param, target_param in zip(critic.parameters(), target.parameters()):
                # target <- (1 - tau) target + tau critic
                target_param.mul_(1 - self.tau)
                target_param.add_(self.tau * param)

    def update(self, replay_buffer, batch_size):
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size, self.device)
        metrics = self.update_critic(obs, action, reward, next_obs, done)
        actor_loss, log_prob = self.update_actor(obs)
        alpha_loss = self.update_alpha(log_prob)
        self.soft_update()
        metrics.update({
            "loss/actor": actor_loss,
            "loss/alpha": alpha_loss,
            "policy/alpha": self.get_alpha().detach(),
            "policy/entropy": -log_prob.mean(),
            "policy/log_prob": log_prob.mean(),
        })
        # Convert to Python numbers only at the training logger's frequency.
        return metrics

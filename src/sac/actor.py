import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if not torch.isfinite(low).all() or not torch.isfinite(high).all() or not (high > low).all():
            raise ValueError("Actor requires finite action bounds with high > low")
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h).clamp(-20, 2)
        return Normal(mu, log_std.exp())

    def sample(self, obs):
        dist = self(obs)
        u = dist.rsample()  # Reparameterization retains the actor gradient through Q(s, a).
        tanh_u = torch.tanh(u)
        action = tanh_u * self.action_scale + self.action_bias
        # Change of variables: Gaussian -> tanh -> environment action units.
        # Retain the original epsilon placement and log_std bounds.
        log_prob = dist.log_prob(u) - torch.log(
            self.action_scale * (1 - tanh_u.pow(2)) + 1e-6
        )
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic_action(self, obs):
        return torch.tanh(self(obs).mean) * self.action_scale + self.action_bias

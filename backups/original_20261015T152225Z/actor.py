import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_limit, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.action_limit = action_limit

    def forward(self, obs):
        h = self.net(obs)

        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)

        # 防止 std 太大或太小
        log_std = torch.clamp(log_std, -20, 2)

        std = torch.exp(log_std)

        dist = Normal(mu, std)

        return dist

    def sample(self, obs):
        dist = self.forward(obs)

        # 1. 重参数化采样，先得到未经过 tanh 的 u
        u = dist.rsample()

        # 2. tanh squash
        tanh_u = torch.tanh(u)

        # 3. 映射到环境动作范围
        action = tanh_u * self.action_limit

        # 4. 先计算高斯分布下 u 的 log_prob
        log_prob = dist.log_prob(u)

        # 5. 对 tanh 和缩放做修正
        log_prob = log_prob - torch.log(
            self.action_limit * (1 - tanh_u.pow(2)) + 1e-6
        )

        # 6. 多维动作时，把各维 log_prob 相加
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

if __name__ == "__main__":

    obs_dim = 3
    action_dim = 1
    batch_size = 256

    actor = GaussianActor(
        obs_dim=3,
        action_dim=1,
        action_limit=2.0
    )
    obs = torch.randn(batch_size, obs_dim)

    action, log_prob = actor.sample(obs)

    print("obs shape:", obs.shape)
    print("action shape:", action.shape)
    print("log_prob shape:", log_prob.shape)

    print("action min:", action.min().item())
    print("action max:", action.max().item())

    print("\nfirst 5 actions:")
    print(action[:5])

    print("\nfirst 5 log_probs:")
    print(log_prob[:5])
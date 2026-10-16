import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

if __name__ == "__main__":

    batch_size = 256
    obs_dim = 3
    action_dim = 1

    critic = Critic(
        obs_dim=obs_dim,
        action_dim=action_dim
    )

    obs = torch.randn(batch_size, obs_dim)

    action = torch.randn(batch_size, action_dim)

    q = critic(obs, action)

    print("obs shape:", obs.shape)
    print("action shape:", action.shape)
    print("q shape:", q.shape)

    print("\nfirst 5 Q values:")
    print(q[:5])


def update_critic(
    obs,
    action,
    reward,
    next_obs,
    done,
    actor,
    q1,
    q2,
    target_q1,
    target_q2,
    critic_optimizer,
    gamma=0.99,
    alpha=0.2,
):

    # 当前 Q
    q1_value = q1(obs, action)
    q2_value = q2(obs, action)

    # target
    with torch.no_grad():

        next_action, next_log_prob = actor.sample(next_obs)

        target_q1_value = target_q1(
            next_obs,
            next_action
        )

        target_q2_value = target_q2(
            next_obs,
            next_action
        )

        target_min_q = torch.min(
            target_q1_value,
            target_q2_value
        )

        soft_next_q = (
            target_min_q
            - alpha * next_log_prob
        )

        target_y = (
            reward
            + gamma
            * (1 - done)
            * soft_next_q
        )

    # loss
    q1_loss = torch.nn.functional.mse_loss(
        q1_value,
        target_y
    )

    q2_loss = torch.nn.functional.mse_loss(
        q2_value,
        target_y
    )

    critic_loss = q1_loss + q2_loss

    # update
    critic_optimizer.zero_grad()

    critic_loss.backward()

    critic_optimizer.step()

    return critic_loss.item()
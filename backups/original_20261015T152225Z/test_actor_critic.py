import copy
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# 1. Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.storage = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.storage.append(
            (
                np.array(obs, dtype=np.float32, copy=True),
                np.array(action, dtype=np.float32, copy=True),
                float(reward),
                np.array(next_obs, dtype=np.float32, copy=True),
                float(done),
            )
        )

    def sample(self, batch_size, device):
        indices = np.random.randint(
            0,
            len(self.storage),
            size=batch_size
        )

        batch = [self.storage[i] for i in indices]

        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.tensor(
            np.stack(obs),
            dtype=torch.float32,
            device=device
        )

        actions = torch.tensor(
            np.stack(actions),
            dtype=torch.float32,
            device=device
        )

        rewards = torch.tensor(
            np.array(rewards),
            dtype=torch.float32,
            device=device
        ).unsqueeze(-1)

        next_obs = torch.tensor(
            np.stack(next_obs),
            dtype=torch.float32,
            device=device
        )

        dones = torch.tensor(
            np.array(dones),
            dtype=torch.float32,
            device=device
        ).unsqueeze(-1)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.storage)


# ============================================================
# 2. Gaussian Actor
# ============================================================

class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        hidden_dim=256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(
            hidden_dim,
            action_dim
        )

        self.log_std_layer = nn.Linear(
            hidden_dim,
            action_dim
        )

        # 环境动作范围
        action_low = torch.tensor(
            action_low,
            dtype=torch.float32
        )

        action_high = torch.tensor(
            action_high,
            dtype=torch.float32
        )

        # [-1, 1] -> [low, high]
        action_scale = (
            action_high - action_low
        ) / 2.0

        action_bias = (
            action_high + action_low
        ) / 2.0

        self.register_buffer(
            "action_scale",
            action_scale
        )

        self.register_buffer(
            "action_bias",
            action_bias
        )

    def forward(self, obs):

        h = self.net(obs)

        mu = self.mu_layer(h)

        log_std = self.log_std_layer(h)

        # 防止 std 太大或太小
        log_std = torch.clamp(
            log_std,
            -20,
            2
        )

        std = torch.exp(log_std)

        dist = Normal(mu, std)

        return dist

    def sample(self, obs):

        # ----------------------------------------------------
        # s -> Actor -> Normal(mu, std)
        # ----------------------------------------------------

        dist = self.forward(obs)

        # ----------------------------------------------------
        # reparameterization:
        #
        # u = mu + std * epsilon
        #
        # 这里必须 rsample 才能让 actor 得到梯度
        # ----------------------------------------------------

        u = dist.rsample()

        # ----------------------------------------------------
        # tanh squash
        # (-inf, inf) -> (-1, 1)
        # ----------------------------------------------------

        tanh_u = torch.tanh(u)

        # ----------------------------------------------------
        # 缩放到环境真正的动作范围
        #
        # Pendulum:
        # [-1,1] -> [-2,2]
        # ----------------------------------------------------

        action = (
            tanh_u * self.action_scale
            + self.action_bias
        )

        # ----------------------------------------------------
        # log pi(a|s)
        #
        # 先算 Gaussian 中 u 的 log probability
        # ----------------------------------------------------

        log_prob = dist.log_prob(u)

        # ----------------------------------------------------
        # tanh + action scaling 的 change-of-variable 修正
        # ----------------------------------------------------

        log_prob -= torch.log(
            self.action_scale
            * (1 - tanh_u.pow(2))
            + 1e-6
        )

        # ----------------------------------------------------
        # 多维 action:
        #
        # log p(a1,...,ad)
        # =
        # log p(a1) + ... + log p(ad)
        # ----------------------------------------------------

        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True
        )

        return action, log_prob

    def deterministic_action(self, obs):

        dist = self.forward(obs)

        # evaluation 时不随机采样
        u = dist.mean

        action = (
            torch.tanh(u)
            * self.action_scale
            + self.action_bias
        )

        return action


# ============================================================
# 3. Critic
# ============================================================

class Critic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=256
    ):
        super().__init__()

        self.q_net = nn.Sequential(

            # 输入 = state + action
            nn.Linear(
                obs_dim + action_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            # Q(s,a) 是 scalar
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):

        # [B, obs_dim] + [B, action_dim]  ->   [B, obs_dim + action_dim]

        x = torch.cat(
            [obs, action],
            dim=-1
        )

        q = self.q_net(x)

        return q


# ============================================================
# 4. SAC Agent
# ============================================================

class SACAgent:

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        device,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        init_alpha=0.2,
        auto_alpha=True,
    ):

        self.device = device

        self.gamma = gamma
        self.tau = tau

        self.auto_alpha = auto_alpha

        # ====================================================
        # Actor
        # ====================================================

        self.actor = GaussianActor(
            obs_dim,
            action_dim,
            action_low,
            action_high
        ).to(device)

        # ====================================================
        # Double Critic
        # ====================================================

        self.q1 = Critic(
            obs_dim,
            action_dim
        ).to(device)

        self.q2 = Critic(
            obs_dim,
            action_dim
        ).to(device)

        # ====================================================
        # Target Critics
        # ====================================================

        self.target_q1 = copy.deepcopy(
            self.q1
        )

        self.target_q2 = copy.deepcopy(
            self.q2
        )

        # Target networks 不直接优化
        for p in self.target_q1.parameters():
            p.requires_grad = False

        for p in self.target_q2.parameters():
            p.requires_grad = False

        # ====================================================
        # Optimizers
        # ====================================================

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters())
            + list(self.q2.parameters()),
            lr=lr
        )

        # ====================================================
        # Temperature alpha
        # ====================================================

        if auto_alpha:

            # 学 log(alpha)，而不是直接学 alpha
            self.log_alpha = torch.tensor(
                np.log(init_alpha),
                dtype=torch.float32,
                device=device,
                requires_grad=True
            )

            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=lr
            )

            # SAC 常用 heuristic
            self.target_entropy = -float(
                action_dim
            )

        else:

            self.fixed_alpha = init_alpha

    def get_alpha(self):

        if self.auto_alpha:
            return self.log_alpha.exp()

        return torch.tensor(
            self.fixed_alpha,
            device=self.device
        )

    # ========================================================
    # 与环境交互时选择 action
    # ========================================================

    @torch.no_grad()
    def act(self, obs, deterministic=False):

        obs = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        if deterministic:

            action = self.actor.deterministic_action(
                obs
            )

        else:

            action, _ = self.actor.sample(
                obs
            )

        return (
            action
            .cpu()
            .numpy()[0]
        )

    # ========================================================
    # SAC Update
    # ========================================================

    def update(self, replay_buffer, batch_size):

        (
            obs,
            action,
            reward,
            next_obs,
            done
        ) = replay_buffer.sample(
            batch_size,
            self.device
        )

        # ====================================================
        # A. Critic Update
        # ====================================================

        # Replay Buffer 中真正执行过的 action
        #
        # Q(s, a_buffer)

        q1_value = self.q1(
            obs,
            action
        )

        q2_value = self.q2(
            obs,
            action
        )

        # ----------------------------------------------------
        # target y
        # ----------------------------------------------------

        with torch.no_grad():

            # s'
            # ↓
            # Actor
            # ↓
            # a' ~ pi(.|s')

            next_action, next_log_prob = (
                self.actor.sample(next_obs)
            )

            # Target Critics 评价同一个 a'

            target_q1_value = self.target_q1(
                next_obs,
                next_action
            )

            target_q2_value = self.target_q2(
                next_obs,
                next_action
            )

            # Clipped Double Q

            target_min_q = torch.min(
                target_q1_value,
                target_q2_value
            )

            alpha = self.get_alpha()

            # Soft value:
            #
            # Q - alpha log pi

            soft_next_q = (
                target_min_q
                - alpha * next_log_prob
            )

            # Bellman target:
            #
            # y =
            # r + gamma(1-d)
            #     [Q - alpha log pi]

            target_y = (
                reward
                + self.gamma
                * (1 - done)
                * soft_next_q
            )

        # ----------------------------------------------------
        # Critic losses
        # ----------------------------------------------------

        q1_loss = F.mse_loss(
            q1_value,
            target_y
        )

        q2_loss = F.mse_loss(
            q2_value,
            target_y
        )

        critic_loss = (
            q1_loss + q2_loss
        )

        # ----------------------------------------------------
        # 更新 Q1/Q2
        # ----------------------------------------------------

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # ====================================================
        # B. Actor Update
        # ====================================================

        # 更新 Actor 时，不需要更新 Critic 参数
        #
        # 但是仍然需要：
        #
        # dQ / da
        #
        # 所以不能 torch.no_grad()

        for p in self.q1.parameters():
            p.requires_grad = False

        for p in self.q2.parameters():
            p.requires_grad = False

        # 当前 state s
        # ↓
        # 当前 Actor
        # ↓
        # 新动作 a_new

        new_action, log_prob = (
            self.actor.sample(obs)
        )

        q1_new = self.q1(
            obs,
            new_action
        )

        q2_new = self.q2(
            obs,
            new_action
        )

        min_q_new = torch.min(
            q1_new,
            q2_new
        )

        # alpha 不应该通过 actor loss 更新
        alpha = self.get_alpha().detach()

        # SAC Actor loss:
        #
        # L_pi =
        # alpha log pi(a|s)
        # -
        # min(Q1,Q2)

        actor_loss = (
            alpha * log_prob
            - min_q_new
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # 恢复 Critic
        for p in self.q1.parameters():
            p.requires_grad = True

        for p in self.q2.parameters():
            p.requires_grad = True

        # ====================================================
        # C. Automatic Alpha Update
        # ====================================================

        if self.auto_alpha:

            alpha_loss = -(
                self.log_alpha
                * (
                    log_prob
                    + self.target_entropy
                ).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()

            alpha_loss.backward()

            self.alpha_optimizer.step()

            alpha_loss_value = (
                alpha_loss.item()
            )

        else:

            alpha_loss_value = 0.0

        # ====================================================
        # D. Soft Target Update
        # ====================================================

        self.soft_update(
            self.q1,
            self.target_q1
        )

        self.soft_update(
            self.q2,
            self.target_q2
        )

        # ====================================================
        # Logging
        # ====================================================

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),

            "q1": q1_value.mean().item(),
            "q2": q2_value.mean().item(),

            "target_q": target_y.mean().item(),

            "log_prob": log_prob.mean().item(),

            "alpha": self.get_alpha().item(),

            "alpha_loss": alpha_loss_value,
        }

    # ========================================================
    # Target soft update
    # ========================================================

    def soft_update(
        self,
        critic,
        target_critic
    ):

        with torch.no_grad():

            for param, target_param in zip(
                critic.parameters(),
                target_critic.parameters()
            ):

                target_param.data.mul_(
                    1 - self.tau
                )

                target_param.data.add_(
                    self.tau * param.data
                )


# ============================================================
# 5. Evaluation
# ============================================================

def evaluate(
    agent,
    env_name,
    episodes=5
):

    env = gym.make(env_name)

    returns = []

    for episode in range(episodes):

        obs, _ = env.reset()

        episode_return = 0.0

        terminated = False
        truncated = False

        while not (
            terminated or truncated
        ):

            # evaluation 用 deterministic action
            action = agent.act(
                obs,
                deterministic=True
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                _
            ) = env.step(action)

            episode_return += reward

        returns.append(
            episode_return
        )

    env.close()

    return np.mean(returns)


# ============================================================
# 6. Main Training Loop
# ============================================================

def main():

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    env_name = "Pendulum-v1"

    seed = 42

    total_steps = 100_000

    buffer_capacity = 100_000

    batch_size = 256

    # 前 5000 步完全随机探索
    start_steps = 5_000

    # buffer 有一定数据之后开始更新
    update_after = 1_000

    updates_per_step = 1

    eval_interval = 5_000

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = gym.make(env_name)

    obs, _ = env.reset(
        seed=seed
    )

    env.action_space.seed(seed)

    obs_dim = (
        env.observation_space.shape[0]
    )

    action_dim = (
        env.action_space.shape[0]
    )

    action_low = (
        env.action_space.low
    )

    action_high = (
        env.action_space.high
    )

    print("obs_dim:", obs_dim)

    print(
        "action_dim:",
        action_dim
    )

    print(
        "action range:",
        action_low,
        action_high
    )

    # --------------------------------------------------------
    # Agent + Buffer
    # --------------------------------------------------------

    replay_buffer = ReplayBuffer(
        buffer_capacity
    )

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,

        action_low=action_low,
        action_high=action_high,

        device=device,

        gamma=0.99,
        tau=0.005,
        lr=3e-4,

        init_alpha=0.2,

        # 如果想先固定 alpha:
        #
        # auto_alpha=False
        #
        auto_alpha=True,
    )

    # --------------------------------------------------------
    # Episode logging
    # --------------------------------------------------------

    episode_return = 0.0
    episode_length = 0
    episode_number = 0

    last_info = None

    # ========================================================
    # Training
    # ========================================================

    for step in range(
        1,
        total_steps + 1
    ):

        # ====================================================
        # 1. 选择动作
        # ====================================================

        if step <= start_steps:

            # 前期用环境随机动作
            action = (
                env.action_space.sample()
            )

        else:

            # 后面使用当前 Actor
            action = agent.act(
                obs,
                deterministic=False
            )

        # ====================================================
        # 2. Environment Step
        # ====================================================

        (
            next_obs,
            reward,
            terminated,
            truncated,
            _
        ) = env.step(action)

        episode_return += reward

        episode_length += 1

        # ----------------------------------------------------
        # 非常重要：
        #
        # truncated = TimeLimit
        #
        # 通常不应该把 TimeLimit 当成真正 terminal，
        # 所以 Bellman target 中只使用 terminated。
        # ----------------------------------------------------

        done_for_buffer = float(
            terminated
        )

        # ====================================================
        # 3. 存 Replay Buffer
        # ====================================================

        replay_buffer.add(
            obs,
            action,
            reward,
            next_obs,
            done_for_buffer
        )

        # s <- s'
        obs = next_obs

        # ====================================================
        # 4. Episode 结束
        # ====================================================

        if terminated or truncated:

            episode_number += 1

            print(
                f"step={step:6d} | "
                f"episode={episode_number:4d} | "
                f"return={episode_return:8.2f} | "
                f"len={episode_length:3d}"
            )

            obs, _ = env.reset()

            episode_return = 0.0
            episode_length = 0

        # ====================================================
        # 5. SAC Update
        # ====================================================

        if (
            step >= update_after
            and len(replay_buffer)
            >= batch_size
        ):

            for _ in range(
                updates_per_step
            ):

                last_info = agent.update(
                    replay_buffer,
                    batch_size
                )

        # ====================================================
        # 6. Evaluation
        # ====================================================

        if step % eval_interval == 0:

            eval_return = evaluate(
                agent,
                env_name,
                episodes=5
            )

            print("\n==============================")

            print(
                f"Evaluation @ step {step}"
            )

            print(
                f"Eval return: "
                f"{eval_return:.2f}"
            )

            if last_info is not None:

                print(
                    "critic_loss:",
                    round(
                        last_info[
                            "critic_loss"
                        ],
                        4
                    )
                )

                print(
                    "actor_loss:",
                    round(
                        last_info[
                            "actor_loss"
                        ],
                        4
                    )
                )

                print(
                    "Q1:",
                    round(
                        last_info["q1"],
                        4
                    )
                )

                print(
                    "target_Q:",
                    round(
                        last_info[
                            "target_q"
                        ],
                        4
                    )
                )

                print(
                    "alpha:",
                    round(
                        last_info[
                            "alpha"
                        ],
                        4
                    )
                )

            print("==============================\n")

    env.close()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        agent.actor.state_dict(),
        "sac_pendulum_actor.pt"
    )

    print(
        "training finished"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
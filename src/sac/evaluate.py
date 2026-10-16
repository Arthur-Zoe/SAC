import gymnasium as gym
import numpy as np


def evaluate(agent, env_name, episodes=20, seed_start=1000):
    """Deterministic actions and the same initial-state seeds at every evaluation."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = gym.make(env_name)
    returns = []
    try:
        for episode_idx in range(episodes):
            obs, _ = env.reset(seed=seed_start + episode_idx)
            episode_return = 0.0
            while True:
                obs, reward, terminated, truncated, _ = env.step(
                    agent.act(obs, deterministic=True)
                )
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
    return float(np.mean(returns)), float(np.std(returns))

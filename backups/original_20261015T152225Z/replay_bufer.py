import gymnasium as gym
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []

    def add(self, obs, action, reward, next_obs, done):
        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
        self.storage.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.storage), size = batch_size, replace = False)
        obs, actions, rewards, next_obs, dones = zip(*[self.storage[i] for i in indices])
        return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs), np.array(dones)

    def __len__(self):
        return len(self.storage)
env = gym.make("Pendulum-v1")
buffer = ReplayBuffer(capacity=10000)
obs, _ = env.reset()
for steps in range(1000):
    
    action = env.action_space.sample()
    next_obs, reward, done, _, _ = env.step(action)
    buffer.add(obs, action, reward, next_obs, done)
    obs = next_obs
    if done:
        obs, info = env.reset()

print(f"Buffer size: {len(buffer)}")
obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = buffer.sample(256)
print(obs_batch.shape)
print(action_batch.shape)
print(reward_batch.shape)
print(next_obs_batch.shape)
print(done_batch.shape)
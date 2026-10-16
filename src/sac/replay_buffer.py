import numpy as np
import torch


class ReplayBuffer:
    """Bounded ring storage; sample with replacement as in the original trainer."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.storage = []
        self.position = 0

    def add(self, obs, action, reward, next_obs, done):
        # done means true termination; TimeLimit truncation must be stored as 0.
        transition = (
            np.array(obs, dtype=np.float32, copy=True),
            np.array(action, dtype=np.float32, copy=True),
            float(reward),
            np.array(next_obs, dtype=np.float32, copy=True),
            float(done),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device="cpu"):
        if not self.storage or batch_size <= 0:
            raise ValueError("sample needs a nonempty buffer and positive batch_size")
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        obs, actions, rewards, next_obs, dones = zip(*(self.storage[i] for i in indices))
        return (
            torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device).unsqueeze(-1),
            torch.as_tensor(np.stack(next_obs), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.storage)

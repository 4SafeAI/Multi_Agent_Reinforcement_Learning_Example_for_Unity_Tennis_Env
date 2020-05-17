import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer.

        Args:
            buffer_size (int): maximum size of buffer.

            batch_size (int): size of each training batch.

            seed (int): random seed.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, experience):
        """Add the experience tuple (s, a, r, s', d) as a new experience to buffer."""
        self.buffer.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from buffer."""
        experiences = random.sample(self.buffer, k=self.batch_size)
        states = np.array([e[0] for e in experiences if e is not None])
        actions = np.array([e[1] for e in experiences if e is not None])
        rewards = np.array([e[2] for e in experiences if e is not None])
        next_states = np.array([e[3] for e in experiences if e is not None])
        dones = np.array([e[4] for e in experiences if e is not None]).astype(np.uint8)
        experiences = (states, actions, rewards, next_states, dones)
        return experiences

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.buffer)

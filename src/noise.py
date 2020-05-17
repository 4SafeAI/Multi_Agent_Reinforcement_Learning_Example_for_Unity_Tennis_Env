import copy

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process.

        Args:
            size: size of the random variable (noise).

            seed: random seed.

            mu: mean reversion level (equilibrium position).

            theta: mean reversion rate (rigidity of Ornstein-Uhlenbeck process).

            sigma: diffusion (impact of randomness on outcome of the process).
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(len(x))
        self.state = x + dx
        return self.state

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
TAU = 3e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 2e-3        # learning rate of the critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SelfPlayAgent(object):
    """Learns from interaction with the environment."""
    def __init__(self, actor_local, actor_target, critic_local, critic_target, noise_processes, replay_buffer):
        """Initialize an Agent.
        
        Args:
            actor_local (torch.nn.Module): local actor model.

            actor_target (torch.nn.Module): target actor model.

            critic_local (torch.nn.Module): local critic model.

            critic_target (torch.nn.Module): target critic model.

            noise (OUNoise): a noise process (used for exploration).

            replay_buffer (ReplayBuffer): a ReplayBuffer for experience storage.
        """
        # Local and Target Actor Networks
        self.actor_local = actor_local.to(device)
        self.actor_target = actor_target.to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Local and Target Critic Networks
        self.critic_local = critic_local.to(device)
        self.critic_target = critic_target.to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise processes
        self.noise_processes = noise_processes

        # Replay memory
        self.replay_buffer = replay_buffer

    def reset_noise_processes(self):
        """Resets the initial state of the noise processes used for exploration."""
        for noise_process in self.noise_processes:
            noise_process.reset()

    def act(self, state, add_noise=True):
        """Returns an action for a given state as per current policy.

        Args:
            state (array): present environment state (minibatch).

            add_noise (boolean): flag, whether noise shall be considered (defaults to True).

        Returns:
            *array*: action associated with state (minibatch).
        """
        state = self.to_device_tensor(state)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.to_cpu_numpy(self.actor_local(state))
        self.actor_local.train()
        if add_noise:
            action += [noise.sample() for noise in self.noise_processes]
        return np.clip(action, -1, 1)

    def add_to_replay_buffer(self, experience):
        """Save experience in replay buffer.

        Args:
            experience (tuple): an experience tuple of (state, action, reward, next_state, done).
        """
        self.replay_buffer.add(experience)

    def learn(self):
        """Make the model learn from one sample of replay buffer.

        Args:
            tau (float): param for soft-update param interpolation.
        """
        try:
            experiences = self.replay_buffer.sample()
            self._update_networks(experiences, TAU)
        except ValueError:
            pass

    def _update_networks(self, experiences, tau):
        """Update actor (policy) & critic (value) model parameters using given batch of experiences.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples

            tau (float): param for soft-update param interpolation.
        """
        self._update_local_critic_network(experiences)
        self._update_local_actor_network(experiences)
        self._update_target_networks(tau)

    def _update_local_critic_network(self, experiences):
        """Updates the critic (value) network from a batch of experiences.

        Args:
            experiences (tuple): tuple of experiences.
        """
        loss_critic = self._calculate_local_critic_loss(experiences)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    def _update_local_actor_network(self, experiences):
        """Updates the actor (policy) network from a batch of experiences.

        Args:
            experiences (tuple): tuple of experiences.
        """
        loss_actor = self._calculate_local_actor_loss(experiences)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

    def _update_target_networks(self, tau):
        """Update the target critic & actor neworks using the soft_update routine.

        Args:
            tau (float): interpolation parameter.
        """
        self._soft_update(self.critic_local, self.critic_target, tau)
        self._soft_update(self.actor_local, self.actor_target, tau)

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        Note:
            θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model: PyTorch model (weights will be copied from)

            target_model: PyTorch model (weights will be copied to)

            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def _calculate_local_critic_loss(self, experiences):
        """Calculates the loss of the critic (value) network from a batch of experiences.

        Note:
            * Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

            where:

                actor_target(state) -> action

                critic_target(state, action) -> Q-value

            * Q_predicted = critic_local(states, actions)

            * Loss_critic = MSE(Q_targets, Q_predicted)

        Args:
            experiences (tuple): tuple of experiences.

        Returns:
            torch.nn.functional.loss: the critic's loss.
        """
        experiences = tuple(map(self.to_device_tensor, experiences))
        states, actions, rewards, next_states, dones = experiences

        # Calculate target Q values
        next_states1 = next_states[:, 0, :]
        next_states2 = next_states[:, 1, :]
        with torch.no_grad():
            next_actions = [self.actor_target(next_states1), self.actor_target(next_states2)]
        next_actions = torch.cat(next_actions, dim=1)
        with torch.no_grad():
            next_states = torch.cat([next_states1, next_states2], dim=1)
            Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + GAMMA * Q_targets_next * (1 - dones)

        # Calculate predicted Q values
        states = states.view(states.shape[0], -1)
        actions = actions.view(actions.shape[0], -1)
        Q_predicted = self.critic_local(states, actions)

        loss_critic = F.mse_loss(Q_predicted, Q_targets)
        return loss_critic

    def _calculate_local_actor_loss(self, experiences):
        """Calculates the actor (policy) network's loss function from a batch of experiences.

        Note:
            Loss_actor = -mean(critic_local(states, actor(states)))

        Args:
            experiences (tuple): a tuple of experiences.

        Returns:
            torch.nn.functional.loss: the actor's loss function.
        """
        experiences = tuple(map(self.to_device_tensor, experiences))
        states, actions, rewards, next_states, dones = experiences

        states1 = states[:, 0, :]
        states2 = states[:, 1, :]
        actions_pred = [self.actor_local(states1), self.actor_local(states2)]
        actions_pred = torch.cat(actions_pred, dim=1)

        states = states.view(states.shape[0], -1)
        loss_actor = -self.critic_local(states, actions_pred).mean()
        return loss_actor

    def to_device_tensor(self, array):
        """Converts a numpy array to a torch tensor and moves it to the device.

        Args:
            array (numpy.ndarray): an array to be converted.

        Returns:
            torch.tensor: array as torch tensor.
        """
        return torch.from_numpy(array).float().to(device)

    def to_cpu_numpy(self, tensor):
        """Converts a torch tensor to numpy array moves it from device to cpu.

        Args:
            tensor (torch.tensor): a tensor to be converted.

        Returns:
            numpy.ndarray: a tensor as a numpy ndarray.
        """
        return tensor.cpu().data.numpy()

from unityagents import UnityEnvironment
from unityagents import UnityEnvironmentException


class Env(object):
    """Environment wrapper for unity environments."""
    def __init__(self, env_file_name):
        """Initializes an Environment wrapper.

        Args:
             env_file_name (str): filename of the Unity environment.
        """
        self._env = UnityEnvironment(file_name=env_file_name)
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]
        self._env_info = self._env.reset(train_mode=True)[self._brain_name]
        self._num_agents = len(self._env_info.agents)
        self._action_size = self._brain.vector_action_space_size
        self._state_size = self.states.shape[1]

    @property
    def action_size(self):
        """tuple: the expected action size."""
        return self._action_size

    @property
    def state_size(self):
        """tuple: the expected state size."""
        return self._state_size

    @property
    def num_agents(self):
        """int: the number of agents in the environment."""
        return self._num_agents

    @property
    def states(self):
        """array: state of the environment."""
        return self._env_info.vector_observations

    @property
    def rewards(self):
        """array: rewards returned by the environment."""
        return self._env_info.rewards

    @property
    def dones(self):
        """array: dones (indicating end of episode) returned by the environment."""
        return self._env_info.local_done

    def reset(self, train_mode=True):
        """Resets the environment (states etc.).

        Args:
            train_mode: True, if in training mode (default), False, if in testing mode.
        """
        self._env_info = self._env.reset(train_mode=train_mode)[self._brain_name]

    def step(self, actions):
        """Transition to next state via actions sent to the environment.

        Args:
            actions (array): an array of actions making the env transition to another state.
        """
        self._env_info = self._env.step(actions)[self._brain_name]

    def __del__(self):
        """Deletes the environment and tries to close it."""
        try:
            self._env.close()
        except UnityEnvironmentException:
            print("Environment already closed!")

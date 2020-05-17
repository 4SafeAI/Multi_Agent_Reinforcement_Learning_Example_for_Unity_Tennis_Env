import numpy as np
from collections import deque


class ModifiedMADDPG(object):
    """ModifiedMADDPG algorithm in control of the agents' learning."""
    def __init__(self, env, target_score=0.0):
        """Initializes a ModifiedMADDPG algorithm that control the agents' learning.

        Args:
            env (Env): a Unity environment wrapper.

            target_score (float): target score (defaults to 0.) (serves as break condition).
        """
        self._env = env
        self._target_score = target_score

    def set_target_score(self, target_score):
        """Sets the target score to a new value.

        Args:
            target_score (float): new target score.
        """
        self._target_score = target_score

    def test(self, agent, max_episodes):
        """Tests an agent.

        Args:
            agent (Agent): some Agent to be tested.

            max_episodes (int): maximum number of test episodes
        """
        mean_scores = []
        for episode in range(1, max_episodes + 1):
            max_score = self._play_episode(agent, train=False)
            print("Episode: {}, Score: {}".format(episode, max_score))
        mean_score = np.mean(mean_scores)
        print("Achieved a mean score of {} in {} episodes".format(mean_score, max_episodes))

    def train(self, agent, max_episodes):
        """Trains an agent.

        Args:
            agent (Agent): some Agent to be trained.

            max_episodes (int): maximum number of training episodes.

        Returns:
            list: max scores per episode.

            list: mean scores over the last 100 episodes.
        """
        max_scores_per_episode = []
        max_scores_last_100_episodes = deque(maxlen=100)
        mean_scores_last_100_episodes = []

        for episode in range(1, max_episodes + 1):
            max_score = self._play_episode(agent, train=True)

            max_scores_last_100_episodes.append(max_score)
            max_scores_per_episode.append(max_score)

            mean_score = np.mean(max_scores_last_100_episodes)
            mean_scores_last_100_episodes.append(mean_score)

            if episode % 1 == 0:
                print('\nEpisode {}\tMean Score: {:.2f}'.
                      format(episode, np.mean(max_scores_last_100_episodes)))
            if mean_score >= self._target_score:
                print('\nEnvironment solved in {:d} episodes!\tMean Score: {:.2f}'.
                      format(episode - 100, mean_score))
                break

        return max_scores_per_episode, mean_scores_last_100_episodes

    def _play_episode(self, agent, train=True):
        """Plays an entire episode and returns the achieved max score.

        Args:
            agent (Agent): some agent.

            train (bool): flag, whether agent shall be trained during playing.

        Returns:
            (float): max score achieved in this episode.
        """
        self._env.reset(train_mode=train)
        scores = np.zeros(self._env.num_agents)

        while True:

            states = self._env.states
            actions = agent.act(states)
            self._env.step(actions)
            next_states = self._env.states
            rewards = self._env.rewards
            dones = self._env.dones

            if train:
                experience = (states, actions, rewards, next_states, dones)
                agent.add_to_replay_buffer(experience)
                agent.learn()

            scores += rewards

            if any(dones):
                break

        max_score = np.max(scores)
        return max_score

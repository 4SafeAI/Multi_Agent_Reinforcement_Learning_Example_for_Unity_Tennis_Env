# import necessary packages
import matplotlib.pyplot as plt
import numpy as np

from src.env_wrapper import Env
from src.self_play_agent import SelfPlayAgent
from src.algorithm import ModifiedMADDPG
from src.model import Actor
from src.model import Critic
from src.model import load_actor
from src.model import load_critic
from src.noise import OUNoise
from src.replay_buffer import ReplayBuffer

####################################################################################################


# load the Unity environment into an Env wrapper
env_file_name = 'resources/environments/Tennis_Windows_x86_64/Tennis.exe'
env = Env(env_file_name)

# number of agents
num_agents = env.num_agents
print('Number of agents:', num_agents)

# examine the state and action space
states = env.states
state_size = env.state_size
action_size = env.action_size
print('There are {} agents.'.format(states.shape[0]))
print('Each agent observes a state with length: {}'.format(state_size))
print('Each agent performs an action of size: {}'.format(action_size))
print('The state for the first agent looks like:', states[0])
print('The state shape looks like:', states.shape)


####################################################################################################


BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
random_seed = 0

# Local and Target Actor Networks
actor_local = Actor(state_size, action_size, random_seed)
actor_target = Actor(state_size, action_size, random_seed)

# Local and Traget Critic Networks
state_action_size = state_size + action_size
critic_local = Critic(num_agents * state_action_size, num_agents, random_seed)
critic_target = Critic(num_agents * state_action_size, num_agents, random_seed)

# Noise processes
noise_process1 = OUNoise(action_size, random_seed, mu=0., theta=0.15, sigma=0.1)
noise_process2 = OUNoise(action_size, random_seed, mu=0., theta=0.15, sigma=0.1)
noise_processes = [noise_process1, noise_process2]

# Replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)


####################################################################################################

# train the Agent
max_episodes = 2000
target_score = 0.5

model_directory = 'resources/models/'
actor_checkpoint_file = model_directory + "checkpoint_actor.pth"
critic_checkpoint_file = model_directory + "checkpoint_critic.pth"
actor_model_file = model_directory + "model_actor.pt"
critic_model_file = model_directory + "model_critic.pt"

agent = SelfPlayAgent(actor_local, actor_target,
                      critic_local, critic_target,
                      noise_processes, replay_buffer)

maddpg = ModifiedMADDPG(env)
maddpg.set_target_score(target_score)
max_scores, mean_scores = maddpg.train(agent, max_episodes)

if max(mean_scores) >= target_score:
    agent.actor_local.save_checkpoint(actor_checkpoint_file)
    agent.actor_local.save_model(actor_model_file)
    agent.critic_local.save_checkpoint(critic_checkpoint_file)
    agent.critic_local.save_model(critic_model_file)


####################################################################################################


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(max_scores)), max_scores)
plt.plot(np.arange(len(mean_scores)), mean_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('MADDPG result: Max / Mean Scores vs. Episodes #')
plt.legend(['max_scores', 'mean_scores'], loc='best')
plt.show()


####################################################################################################

# test the Agent
agent = SelfPlayAgent(actor_local, actor_target,
                      critic_local, critic_target,
                      noise_processes, replay_buffer)
agent.actor_local = load_actor(actor_model_file)
agent.critic_local = load_critic(critic_model_file)
maddpg.test(agent, max_episodes=5)

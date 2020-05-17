[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

**Note: This README.md-file was borrowed from the udacity/deep-reinforcement-learning repository  
(https://github.com/udacity/deep-reinforcement-learning.git) and slightly modified**.

# Project 3: Collaboration and Competition

### Introduction

![Trained Agent][image1]

In this project, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Setup the dependencies as described [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md).

2. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. In case you have downloaded your own environment, place it in the *resources/environments/*-folder. 
Alternatively, you can use a Windows (64-bit) version of the Tennis environment, that is already placed there for you.
 
4. Update the *env_file_name* variable in the **main.py**-script accordingly. 

### Folder structure

   - *resources*: The trained MADDPG-agent-associated files are located here.
   
      - *environments*: Contains the Unity Tennis environment.
          
      - *models*: Contains the actor's and critic's *checkpoint_x.pth* and *model_x.pt* files.
      
      - *results*: Contains GIFS
      
         - gifs: GIFS of playing agents 
    
   - *src*: The python files are located here.

### Instructions

To train the agent run the **main.py**-script. 
    
   • Further details and descriptions of the implementation is provided in the **Report.md**-file. 

   • For technical details see the **.py-files* in the **src** folder. 

   • Actor and critic model **weights** are stored in **checkpoint_actor.pth** and **checkpoint_critic.pth**, respectively.

   • Actor and critic **models themselves** are stored in **model_actor.pt** and **model_critic.pt**, respectively.
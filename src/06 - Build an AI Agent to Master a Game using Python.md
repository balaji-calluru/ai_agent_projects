# Build an AI Agent to Master a Game using Python

Do you know how AI opponents in video games seem to think, adapt, and sometimes, even outsmart you? The secret behind this is a powerful branch of AI called Reinforcement Learning. In fact, AI agents trained with these very principles have achieved superhuman performance in complex games like StarCraft II. In this article, I’ll explain how to build an AI agent from scratch using Python that learns to master a classic control game, known as the CartPole game.

What Exactly Is an AI Agent?
An AI agent is a program that can observe its environment, make decisions, and take actions to achieve a specific goal. It learns through trial and error, much like a person learning a new skill. It’s all done using:

The Environment: The world the Agent lives in (in our case, the CartPole game).
The Action: A move the Agent can make (e.g., push the cart left or right).
The Reward: Feedback from the environment. A positive reward (like +1) for a good action, and a negative one (or no reward) for a bad one.
The Agent’s only goal is to maximize its total reward over time. It does this by learning a policy, a strategy for choosing the best action in any given situation. This learning process is called Reinforcement Learning.

Build an AI Agent to Master a Game using Python
Before our Agent can learn, it needs a world to interact with. We’ll use Gymnasium, a popular Python library that provides a vast collection of game environments for testing RL algorithms. We also need PyTorch to build the Agent’s brain.

First, let’s install these libraries:

pip install gymnasium torch
Step 1: Setting Up Our Playground
The game we’ll be tackling is CartPole-v1. The goal is simple:

![CartPole Game](./images/CartPole-v1.webp)

Build an AI Agent to Master a Game
A pole is attached to a cart, and our Agent must move the cart left or right to keep the pole balanced upright.
For every moment it holds the pole balanced, it gets a reward of +1.
If the pole falls over, the game ends.
Let’s see what this environment looks like. The following code initializes the CartPole game and has it take random actions for a few seconds:

1
import gymnasium as gym
2
import time
3
​
4
# load the CartPole environment
5
env = gym.make("CartPole-v1", render_mode="human")
6
​
7
# every game starts with a reset
8
state, info = env.reset()
9
​
10
# run the game for a short period
11
for _ in range(50):
12
    # render the current frame
13
    env.render()
14
​
15
    # choose a random action (0 for push left, 1 for push right)
16
    action = env.action_space.sample()
17
​
18
    next_state, reward, terminated, truncated, info = env.step(action)
19
​
20
    print(f"State: {state.shape}, Action: {action}, Reward: {reward}")
21
​
22
    # update the state for the next loop
23
    state = next_state
24
​
25
    # if the game is over, reset it to start a new game
26
    if terminated or truncated:
27
        state, info = env.reset()
28
​
29
    time.sleep(0.02) # slow down for visualization
30
​
31
# close the environment window
32
env.close()
State: (4,), Action: 1, Reward: 1.0
State: (4,), Action: 1, Reward: 1.0
State: (4,), Action: 1, Reward: 1.0
State: (4,), Action: 0, Reward: 1.0
...
State: (4,), Action: 1, Reward: 1.0
State: (4,), Action: 1, Reward: 1.0
State: (4,), Action: 1, Reward: 1.0
If you run this, you’ll see the cart jittering around randomly and the pole falling over very quickly. This is our baseline, an unintelligent agent. Now, let’s build its brain.

Step 2: Building the Agent’s Brain with a Neural Network
Our Agent needs a way to decide which action is best in any given state. To do this, we’ll use a neural network called a Q-Network.

The network will take the game’s state as input (which for CartPole is a set of 4 numbers: cart position, cart velocity, pole angle, and pole angular velocity) and output a Q-value for each possible action (push left or push right). The higher the Q-value, the more suitable the network perceives that action to be for the current state.

Here’s how we define this network using PyTorch:

1
import torch
2
import torch.nn as nn
3
import torch.optim as optim
4
​
5
class QNetwork(nn.Module):
6
    """
7
    Neural Network to approximate the Q-value function.
8
    """
9
    def __init__(self, state_size, action_size):
10
        """
11
        Initializes the network layers.
12
        :param state_size: The number of features in the game state (e.g., 4 for CartPole).
13
        :param action_size: The number of possible actions (e.g., 2 for CartPole).
14
        """
15
        super(QNetwork, self).__init__()
16
        self.network = nn.Sequential(
17
            nn.Linear(state_size, 128),
18
            nn.ReLU(), # activation function
19
            nn.Linear(128, 128),
20
            nn.ReLU(),
21
            nn.Linear(128, action_size)
22
        )
23
​
24
    def forward(self, state):
25
        """
26
        Defines the forward pass of the network.
27
        It takes a state and returns the Q-values for each action.
28
        """
29
        return self.network(state)
This simple network is the core of our Agent’s intelligence. It will learn to map game states to smart actions.

Step 3: Creating the Agent’s Logic
Now we wrap our Q-Network inside a DQNAgent class. This class will handle all the logic for interacting with the environment, learning from experiences, and making decisions. Here are its three key jobs:

Remembering Experiences: The Agent doesn’t just learn from its last move. It stores thousands of past experiences (state, action, reward, next_state, done) in a replay buffer. It then learns from random samples of these memories, which is a much more stable way to train.
Deciding a Move: To choose an action, the Agent uses an epsilon-greedy strategy. Most of the time, it exploits its knowledge by picking the action its network predicts is best. But sometimes (controlled by a value called epsilon), it explores by taking a random action to discover potentially better strategies. Early on, the Agent explores a lot, but as it gets smarter, epsilon decreases, and it relies more on what it has learned.
Getting Smarter: This is the core of the learning process. The Agent compares the Q-value its network expects for an action with a more accurate target Q-value calculated from the actual reward and outcome. The difference between the expected and target values is the loss. The Agent then updates its network to minimize this loss, slowly making its predictions closer to reality.
Here’s how to implement the DQN Algorithm for creating the Agent’s logic:

1
import torch
2
import torch.nn as nn
3
import torch.optim as optim
4
import torch.nn.functional as F
5
import numpy as np
6
import random
7
from collections import deque, namedtuple
8
​
9
# Hyperparameters
10
BUFFER_SIZE = 10000     # replay buffer size
11
BATCH_SIZE = 64         # minibatch size for training
12
GAMMA = 0.99            # discount factor for future rewards
13
LR = 5e-4               # learning rate
14
UPDATE_EVERY = 4        # how often to update the network
15
​
16
# use GPU if available
17
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
18
​
19
class DQNAgent():
20
    """Interacts with and learns from the environment."""
21
​
22
    def __init__(self, state_size, action_size):
23
        self.state_size = state_size
24
        self.action_size = action_size
25
​
26
        # Q-Network
27
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
28
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
29
​
30
        # replay memory
31
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
32
        # initialize time step
33
        self.t_step = 0
34
​
35
    def step(self, state, action, reward, next_state, done):
36
        # save experience in replay memory
37
        self.memory.add(state, action, reward, next_state, done)
38
​
39
        # learn every UPDATE_EVERY time steps.
40
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
41
        if self.t_step == 0:
42
            # if enough samples are available in memory, get random subset and learn
43
            if len(self.memory) > BATCH_SIZE:
44
                experiences = self.memory.sample()
45
                self.learn(experiences, GAMMA)
46
​
47
    def act(self, state, eps=0.):
48
        """Returns actions for given state as per current policy."""
49
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
50
        self.qnetwork_local.eval() # set network to evaluation mode
51
        with torch.no_grad():
52
            action_values = self.qnetwork_local(state)
53
        self.qnetwork_local.train() # set network back to training mode
54
​
55
        # epsilon-greedy action selection
56
        if random.random() > eps:
57
            return np.argmax(action_values.cpu().data.numpy())
58
        else:
59
            return random.choice(np.arange(self.action_size))
60
​
61
    def learn(self, experiences, gamma):
62
        """Update value parameters using given batch of experience tuples."""
63
        states, actions, rewards, next_states, dones = experiences
64
​
65
        # get max predicted Q-values for next states from the network
66
        Q_targets_next = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
67
​
68
        # compute Q targets for current states
69
        # target = reward + gamma * Q_next (if not done)
70
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
71
​
72
        # get expected Q values from local model
73
        Q_expected = self.qnetwork_local(states).gather(1, actions)
74
​
75
        # compute loss
76
        loss = F.mse_loss(Q_expected, Q_targets)
77
​
78
        # minimize the loss
79
        self.optimizer.zero_grad()
80
        loss.backward()
81
        self.optimizer.step()
82
​
83
# Replay Buffer
84
class ReplayBuffer:
85
    """Fixed-size buffer to store experience tuples."""
86
    def __init__(self, action_size, buffer_size, batch_size):
87
        self.memory = deque(maxlen=buffer_size)
88
        self.batch_size = batch_size
89
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
90
​
91
    def add(self, state, action, reward, next_state, done):
92
        """Add a new experience to memory."""
93
        e = self.experience(state, action, reward, next_state, done)
94
        self.memory.append(e)
95
​
96
    def sample(self):
97
        """Randomly sample a batch of experiences from memory."""
98
        experiences = random.sample(self.memory, k=self.batch_size)
99
​
100
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
101
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
102
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
103
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
104
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
105
​
106
        return (states, actions, rewards, next_states, dones)
107
​
108
    def __len__(self):
109
        """Return the current size of internal memory."""
110
        return len(self.memory)
Over thousands of these minor updates, the Q-Network’s predictions become incredibly accurate, and the Agent’s policy becomes very effective.

Step 4: The Training Loop
Now we bring everything together in a training loop. We’ll set some hyperparameters, like the number of episodes to train for and the epsilon decay rate, and then set the Agent loose in the environment:

1
import gymnasium as gym
2
​
3
# Initialize Environment and Agent
4
env = gym.make("CartPole-v1")
5
state_size = env.observation_space.shape[0]
6
action_size = env.action_space.n
7
agent = DQNAgent(state_size=state_size, action_size=action_size)
8
​
9
# Training Hyperparameters
10
n_episodes = 2000       # max number of training episodes
11
max_t = 1000            # max number of timesteps per episode
12
eps_start = 1.0         # starting value of epsilon
13
eps_end = 0.01          # minimum value of epsilon
14
eps_decay = 0.995       # multiplicative factor for decreasing epsilon
15
​
16
def train():
17
    scores = []                         # list containing scores from each episode
18
    scores_window = deque(maxlen=100)   # last 100 scores
19
    eps = eps_start                     # initialize epsilon
20
​
21
    for i_episode in range(1, n_episodes + 1):
22
        state, info = env.reset()
23
        score = 0
24
        for t in range(max_t):
25
            action = agent.act(state, eps)
26
            next_state, reward, terminated, truncated, info = env.step(action)
27
            done = terminated or truncated
28
​
29
            agent.step(state, action, reward, next_state, done)
30
​
31
            state = next_state
32
            score += reward
33
            if done:
34
                break
35
​
36
        scores_window.append(score)
37
        scores.append(score)
38
​
39
        # decrease epsilon
40
        eps = max(eps_end, eps_decay * eps)
41
​
42
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
43
        if i_episode % 100 == 0:
44
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
45
​
46
        # check if the environment is solved
47
        if np.mean(scores_window) >= 195.0:
48
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
49
            # save the trained model's weights
50
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
51
            break
52
​
53
    return scores
54
​
55
scores = train()
56
env.close()
Episode 100	Average Score: 19.83
Episode 200	Average Score: 66.43
Episode 261	Average Score: 196.76
Environment solved in 161 episodes!	Average Score: 196.76
Watching the output, you’ll see the average score start low and steadily climb. In the provided test run, the Agent achieved an average score of 196.76 after just 261 episodes, solving the environment in only 161 learning episodes!

Watching Our AI Agent Play the Game
Training is done. We’ve saved our Agent’s brain. Now, let’s load those trained weights and watch our Agent play flawlessly:

1
# Testing the AI Agent
2
​
3
# Step 1: Initialize the Environment and Agent
4
env = gym.make("CartPole-v1", render_mode="human")
5
state_size = env.observation_space.shape[0]
6
action_size = env.action_space.n
7
​
8
# create an agent
9
agent = DQNAgent(state_size=state_size, action_size=action_size)
10
​
11
# Step 2: Load the Trained Weights
12
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
13
​
14
# Step 3: Watch the Smart Agent Play
15
num_episodes_to_watch = 10
16
​
17
for i in range(num_episodes_to_watch):
18
    # reset the environment to get the initial state
19
    state, info = env.reset()
20
    done = False
21
    episode_reward = 0
22
​
23
    print(f"--- Watching Episode {i+1} ---")
24
​
25
    while not done:
26
        # render the environment
27
        env.render()
28
​
29
        # choose the best action using the trained network (epsilon=0)
30
        action = agent.act(state)
31
​
32
        # perform the action in the environment
33
        next_state, reward, terminated, truncated, info = env.step(action)
34
        done = terminated or truncated
35
​
36
        # update the state
37
        state = next_state
38
        episode_reward += reward
39
​
40
        # add a small delay to make it watchable
41
        time.sleep(0.02)
42
​
43
    print(f"Score for Episode {i+1}: {episode_reward}")
44
​
45
# close the environment window
46
env.close()
--- Watching Episode 1 ---
Score for Episode 1: 500.0
--- Watching Episode 2 ---
Score for Episode 2: 500.0
--- Watching Episode 3 ---
Score for Episode 3: 460.0
--- Watching Episode 4 ---
Score for Episode 4: 420.0
--- Watching Episode 5 ---
Score for Episode 5: 498.0
--- Watching Episode 6 ---
Score for Episode 6: 438.0
--- Watching Episode 7 ---
Score for Episode 7: 410.0
--- Watching Episode 8 ---
Score for Episode 8: 451.0
--- Watching Episode 9 ---
Score for Episode 9: 388.0
--- Watching Episode 10 ---
Score for Episode 10: 419.0
When you run this code, you’ll see the AI Agent expertly balancing the pole, often for hundreds of steps. The final scores speak for themselves, with the Agent consistently achieving high scores, like 500, 460, and 498!

Final Words
So, in this article, you learned to build a fully trained AI agent to master a game. We didn’t just use a pre-built model; we built the Agent’s memory, its decision-making logic, and its learning mechanism from the ground up.

This is the foundation. The same principles of states, actions, rewards, and learning from experience are used to build AI Agents that can master far more complex challenges, from a self-driving car navigating a city to an AI champion in a professional video game.

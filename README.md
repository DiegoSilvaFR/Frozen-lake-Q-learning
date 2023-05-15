# Frozen Lake V1 with Q-Learning
This project demonstrates the use of the gym package to implement the Frozen Lake V1 environment and solve it using the Q-learning algorithm.

# Prerequisites
 > Python 3.x
 > gym package
 > numpy package
# Installation
## Clone the repository:
git clone https://github.com/your-username/frozen-lake-q-learning.git
cd frozen-lake-q-learning
Install the required packages using pip:
> pip install gym numpy
# Usage
To run the Q-learning algorithm on the Frozen Lake V1 environment, execute the following command:

python example.py
This will start the training process, where the agent will learn the optimal policy for navigating the frozen lake.

# Environment
The Frozen Lake V1 environment is a grid-world game where the agent must navigate from a starting point to the goal while avoiding holes. The surface of the lake is described using a grid, where:

S represents the starting point.
F represents frozen tiles.
H represents holes.
G represents the goal.
The agent can take four actions: move up, down, left, or right. However, due to the slippery nature of the lake, the agent may not always move in the intended direction. It has a probability of 0.33 of sliding into an adjacent tile instead.

Q-Learning Algorithm
Q-learning is a reinforcement learning algorithm that aims to find the optimal action-value function, Q(s, a), for an agent in an environment. The action-value function represents the expected cumulative rewards the agent can obtain by taking action a in state s and following the optimal policy thereafter.

The Q-learning algorithm uses the following update rule to iteratively update the Q-values based on the agent's experience:


Q(s, a) = Q(s, a) + α * (R + γ * max[Q(s', a')] - Q(s, a))
where:

Q(s, a) is the current Q-value for state s and action a.
α is the learning rate (0 <= α <= 1) that controls the weight given to the new information.
R is the immediate reward received after taking action a in state s.
γ is the discount factor (0 <= γ <= 1) that determines the importance of future rewards.
s' is the next state after taking action a in state s.
a' is the action that maximizes the Q-value in the next state s'.
max[Q(s', a')] is the maximum Q-value among all possible actions in the next state s'.
The algorithm continues updating the Q-values until convergence or a predefined number of episodes.

# Results
The training process outputs the following information:

Episode: The current episode number.
Steps: The number of steps taken in the episode.
Total Reward: The cumulative reward obtained in the episode.
Success: Whether the agent reached the goal in the episode.
Once the training is complete, the algorithm outputs the optimal policy and the success rate of the learned policy on 100 evaluation episodes.

# Customization
If you want to modify the Q-learning algorithm or experiment with different hyperparameters, you can open the main.py file and make the necessary changes. Feel free to explore and tweak the code to suit your requirements.

# Credits
This project is based on the gym package and the Frozen Lake V1 environment provided by OpenAI. The implementation of the Q-learning algorithm is inspired by various sources in the field of reinforcement learning.

# License
The project is licensed under the MIT License. Feel





import gymnasium as gym

# Create the environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')

# Get the number of states
num_states = env.observation_space.n
num_actions = env.action_space.n

# Generate a list of states
states = range(num_states)
actions= range(num_actions)

#Make a Q-table
q_table={}
for s in states:
    for a in actions:
        state_action=(s,a)
        q_table[state_action]=0

import numpy as np

n_episodes=10
alpha=0.001
gamma=0.9
epsilon=0.1

def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Choose a random action
    else:
        return max(actions, key=lambda a: q_table[(state, a)])  # Choose the action with the highest Q-value

for i in range(n_episodes):
    state, info = env.reset()  # Reset the environment
    done = False
    while not done:
        action = epsilon_greedy_policy(state, epsilon)  # Choose action based on epsilon-greedy policy
        next_state, reward, done, truncated, info = env.step(action)  # Take the action

        # Update Q-table using the Q-learning formula
        best_next_action = max(actions, key=lambda a: q_table[(next_state, a)])
        td_target = reward + gamma * q_table[(next_state, best_next_action)]
        td_error = td_target - q_table[(state, action)]
        q_table[(state, action)] += alpha * td_error

        # Move to the next state
        state = next_state
print(q_table)
import numpy as np

# Define the environment
num_states = 6
num_actions = 2
reward_matrix = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

# Q-learning parameters
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 1000

# Initialize Q-table
q_table = np.zeros((num_states, num_actions))

# Q-learning algorithm
for episode in range(num_episodes):
    state = np.random.randint(0, num_states)  # Start from a random state

    while state != 5:  # Goal state is 5
        action = np.argmax(q_table[state, :]) if np.random.rand() < 0.8 else np.random.randint(0, num_actions)
        next_state = np.argmax(q_table[state, :])

        # Q-value update
        q_table[state, action] += learning_rate * (reward_matrix[state, action] + 
                                                   discount_factor * np.max(q_table[next_state, :]) - 
                                                   q_table[state, action])

        state = next_state

# Print the learned Q-table
print("Learned Q-table:")
print(q_table)

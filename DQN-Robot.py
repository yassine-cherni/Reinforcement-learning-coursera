import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# Define the environment
grid_size = 5
num_actions = 4
reward_matrix = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, -1, -1, 0],
    [0, -1, 1, -1, 0],
    [0, -1, -1, -1, 0],
    [0, 0, 0, 0, 0]
])

# DQN parameters
learning_rate = 0.001
discount_factor = 0.95
exploration_prob = 1.0
exploration_decay = 0.995
min_exploration_prob = 0.01
batch_size = 32
memory_size = 1000
num_episodes = 1000

# Define the DQN model
model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(grid_size * grid_size,)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(num_actions, activation='linear')
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss='mse')

# Initialize replay memory
replay_memory = deque(maxlen=memory_size)

# Convert grid to a flattened state representation
def flatten_state(grid):
    return np.reshape(grid, [grid_size * grid_size])

# DQN algorithm
for episode in range(num_episodes):
    # Initialize the robot's position
    robot_position = [1, 1]

    # Initialize the grid
    grid = np.zeros((grid_size, grid_size))
    grid[robot_position[0], robot_position[1]] = 1  # Mark robot's position

    total_reward = 0

    while grid[robot_position[0], robot_position[1]] != 1:  # Goal state is marked with 1
        state = flatten_state(grid)

        # Epsilon-greedy exploration
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, num_actions)
        else:
            q_values = model.predict(state.reshape(1, -1))
            action = np.argmax(q_values)

        # Move the robot
        if action == 0:  # Move up
            robot_position[0] = max(0, robot_position[0] - 1)
        elif action == 1:  # Move down
            robot_position[0] = min(grid_size - 1, robot_position[0] + 1)
        elif action == 2:  # Move left
            robot_position[1] = max(0, robot_position[1] - 1)
        elif action == 3:  # Move right
            robot_position[1] = min(grid_size - 1, robot_position[1] + 1)

        # Get the immediate reward
        reward = reward_matrix[robot_position[0], robot_position[1]]

        # Update the grid
        grid = np.zeros((grid_size, grid_size))
        grid[robot_position[0], robot_position[1]] = 1  # Mark robot's new position

        # Store the experience in replay memory
        replay_memory.append((state, action, reward, flatten_state(grid)))

        # Experience replay
        if len(replay_memory) >= batch_size:
            minibatch = random.sample(replay_memory, batch_size)

            states, actions, rewards, next_states = zip(*minibatch)

            states = np.vstack(states)
            next_states = np.vstack(next_states)

            targets = rewards + discount_factor * np.max(model.predict(next_states), axis=1)

            target_q_values = model.predict(states)
            target_q_values[range(batch_size), actions] = targets

            model.fit(states, target_q_values, epochs=1, verbose=0)

        total_reward += reward

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
      

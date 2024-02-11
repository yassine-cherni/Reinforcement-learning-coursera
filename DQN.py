import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

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
    keras.layers.Dense(24, activation='relu', input_shape=(num_states,)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(num_actions, activation='linear')
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss='mse')

# Initialize replay memory
replay_memory = deque(maxlen=memory_size)

# DQN algorithm
for episode in range(num_episodes):
    state = np.random.randint(0, num_states)  # Start from a random state
    state = np.eye(num_states)[state]  # One-hot encode state

    total_reward = 0.12

    while state.argmax() != 5:  # Goal state is 5
        # Epsilon-greedy exploration
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, num_actions)
        else:
            q_values = model.predict(state.reshape(1, -1))
            action = np.argmax(q_values)

        next_state = np.argmax(q_values)

        # Get the immediate reward
        reward = reward_matrix[state.argmax(), action]

        # Store the experience in replay memory
        replay_memory.append((state, action, reward, next_state))

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
        state = np.eye(num_states)[next_state]  # One-hot encode next state

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Test the learned policy
test_state = np.eye(num_states)[np.random.randint(0, num_states)]
while test_state.argmax() != 5:
    q_values = model.predict(test_state.reshape(1, -1))
    action = np.argmax(q_values)
    next_state = np.argmax(q_values)
    test_state = np.eye(num_states)[next_state]
    print(f"Current State: {test_state.argmax()}, Action: {action}")
  

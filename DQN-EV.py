import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# Define the EV environment
initial_charge = 50.0  # Initial battery charge in kWh
max_charge = 100.0  # Maximum battery capacity in kWh
min_charge = 0.0  # Minimum battery capacity in kWh
charging_rate = 10.0  # Charging rate in kWh per time step
discharging_rate = 15.0  # Discharging rate in kWh per time step
time_steps = 24  # Number of time steps in a day
reward_matrix = np.array([0, -10])  # Rewards for charging and discharging

# DQN parameters
learning_rate = 0.001
discount_factor = 0.95
exploration_prob = 1.0
exploration_decay = 0.995
min_exploration_prob = 0.01
batch_size = 32
memory_size = 1000
num_episodes = 500

# Define the DQN model
model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(2,)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(2, activation='linear')  # 2 actions: charge or discharge
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss='mse')

# Initialize replay memory
replay_memory = deque(maxlen=memory_size)

# Convert state to a 2D representation [current_charge, time_step]
def state_representation(charge, time_step):
    return np.array([charge / max_charge, time_step / time_steps])

# DQN algorithm
for episode in range(num_episodes):
    current_charge = initial_charge
    total_reward = 0

    for time_step in range(time_steps):
        state = state_representation(current_charge, time_step)

        # Epsilon-greedy exploration
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, 2)  # 0: charge, 1: discharge
        else:
            q_values = model.predict(state.reshape(1, -1))
            action = np.argmax(q_values)

        # Update charge based on action
        if action == 0:  # Charge
            current_charge = min(max_charge, current_charge + charging_rate)
        else:  # Discharge
            current_charge = max(min_charge, current_charge - discharging_rate)

        # Get the immediate reward
        reward = reward_matrix[action]

        # Store the experience in replay memory
        next_state = state_representation(current_charge, time_step + 1)
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

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
      

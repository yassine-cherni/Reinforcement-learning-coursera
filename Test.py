import numpy as 
import tensorflow as tf
from tensorflow import keras
import gym

# Define the CartPole environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the deep Q-network model
model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(state_size,)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(action_size, activation='linear')
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse')

# Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
memory = []

# Experience replay parameters
batch_size = 32
memory_capacity = 10000

# Main training loop
for episode in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    total_reward = 0

    for time_step in range(500):  # Limit the number of time steps per episode
        # Uncomment the next line to visualize the environment
        # env.render()

        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Store the experience in the replay memory
        memory.append((state, action, reward, next_state, done))

        # Limit the size of the replay memory
        if len(memory) > memory_capacity:
            memory.pop(0)

        state = next_state
        total_reward += reward

        # Sample a random batch from the replay memory for training
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)
            states = np.vstack(states)
            next_states = np.vstack(next_states)

            targets = rewards + gamma * np.max(model.predict(next_states), axis=1)
            targets[dones] = rewards[dones]  # If the episode is done, no future rewards

            target_q_values = model.predict(states)
            target_q_values[range(batch_size), actions] = targets

            # Train the model on the batch
            model.fit(states, target_q_values, epochs=1, verbose=0)

        if done:
            break

    # Decay epsilon for exploration-exploitation trade-off
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Close the environment
env.close()

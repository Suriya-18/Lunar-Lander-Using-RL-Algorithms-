import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import logging
import imageio

# Configure logging to capture error messages
logging.basicConfig(level=logging.ERROR, filename="training_errors.log", filemode="w")

# Create environment with render_mode="human" for visualization after training
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env.reset(seed=0)

# Set random seeds
np.random.seed(0)
tf.random.set_seed(0)

# Get state and action shapes
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

# Print state and action shapes
print("State Shape:", state_shape)
print("Action Shape:", action_shape)


def create_actor_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(state_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_shape, activation='softmax')
    ])
    return model

def create_critic_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(state_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model

actor_model = create_actor_model()
critic_model = create_critic_model()

# Define optimizers
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)


def get_action(state):
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    state = tf.expand_dims(state, axis=0)
    probs = actor_model(state)
    action = np.random.choice(action_shape, p=np.squeeze(probs.numpy()))
    return action

def compute_advantage(rewards, values, gamma=0.99):
    advantages = []
    discounted_sum = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        discounted_sum = r + gamma * discounted_sum
        advantages.insert(0, discounted_sum - v)
    return advantages

def train_step(states, actions, rewards, values, gamma=0.99):
    advantages = compute_advantage(rewards, values, gamma)

    # Convert to tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)

    # Check for NaN/Inf values using tf.reduce_any
    if (tf.reduce_any(tf.math.is_nan(states)).numpy() or tf.reduce_any(tf.math.is_inf(states)).numpy() or
            tf.reduce_any(tf.math.is_nan(rewards)).numpy() or tf.reduce_any(tf.math.is_inf(rewards)).numpy() or
            tf.reduce_any(tf.math.is_nan(values)).numpy() or tf.reduce_any(tf.math.is_inf(values)).numpy() or
            tf.reduce_any(tf.math.is_nan(advantages)).numpy() or tf.reduce_any(tf.math.is_inf(advantages)).numpy()):
        print("Encountered NaN/Inf values. Skipping this training step.")
        return

    # Update Critic
    with tf.GradientTape() as tape:
        value_pred = critic_model(states, training=True)
        rewards = tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), (-1, 1))
        value_loss = tf.reduce_mean(tf.square(value_pred - rewards))
    grads = tape.gradient(value_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(zip(grads, critic_model.trainable_variables))

    # Update Actor
    with tf.GradientTape() as tape:
        probs = actor_model(states, training=True)
        action_probs = tf.gather_nd(probs, tf.stack([tf.range(probs.shape[0], dtype=tf.int32), actions], axis=1))
        log_probs = tf.math.log(action_probs)
        actor_loss = -tf.reduce_mean(log_probs * advantages)
    grads = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))


# Set training parameters
episodes = 1000
gamma = 0.99
episode_rewards = []
max_episode_length = 1000  # Set a maximum episode length

# Start training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    states, actions, rewards, values = [], [], [], []
    step_count = 0

    while not done and step_count < max_episode_length:
        try:
            action = get_action(state.astype(np.float32))
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            values.append(critic_model(state_tensor).numpy()[0][0])

            if (np.isnan(state).any() or np.isinf(state).any() or
                np.isnan(reward) or np.isinf(reward)):
                print("Encountered NaN/Inf values. Stopping episode.")
                break

            state = next_state
            total_reward += reward
            step_count += 1

            if step_count % 100 == 0:
                print(f"Episode: {episode + 1}, Step: {step_count}, Total Reward: {total_reward:.2f}")

        except Exception as e:
            logging.exception(f"Error in episode {episode}, step {step_count}: {e}")
            print(f"Error in episode {episode}, step {step_count}: {e}")
            break

    if done or step_count >= max_episode_length:
        episode_rewards.append(total_reward)
        train_step(states, actions, rewards, values, gamma)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}")


import matplotlib.pyplot as plt # Import the necessary module

print("Training complete. Now visualizing the trained agent...")

# Visualization after training
state, _ = env.reset()
done = False
while not done:
    # Get action from the trained actor model
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
    probs = actor_model(state_tensor)
    action = np.random.choice(action_shape, p=np.squeeze(probs.numpy()))

    # Take a step and render the environment
    next_state, reward, done, _, _ = env.step(action)
    rendered_frame = env.render()

    # Update state for the next step
    state = next_state
    plt.imshow(rendered_frame) # Now plt is defined and can be used
    plt.pause(0.01)  # Pause briefly to show the frame
    plt.clf()

# Close environment after training and visualization
env.close()

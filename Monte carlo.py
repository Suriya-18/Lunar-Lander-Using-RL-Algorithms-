import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import imageio
from pyvirtualdisplay import Display
import matplotlib.pyplot as plt

# Setup virtual display for Colab
Display(visible=0, size=(1400, 900)).start()

# Environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Hyperparameters (reduced for 20 episodes)
EPISODES = 20
GAMMA = 0.99
LR = 0.005  # Slightly higher learning rate for fast convergence

# Simplified Policy Network
model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(state_dim,)),
    layers.Dense(n_actions, activation='softmax')
])
optimizer = tf.keras.optimizers.Adam(LR)

# Tracking
episode_rewards = []

def get_action(state):
    probs = model(np.array([state], dtype=np.float32))
    return np.random.choice(n_actions, p=probs.numpy()[0])

def train_episode(states, actions, rewards):
    # Monte Carlo returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    with tf.GradientTape() as tape:
        probs = model(tf.convert_to_tensor(states, dtype=tf.float32))
        action_probs = tf.gather_nd(probs, tf.stack([tf.range(len(actions)), actions], axis=1))
        loss = -tf.reduce_mean(tf.math.log(action_probs) * returns)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Training with video for last episode
for ep in range(EPISODES):
    state, _ = env.reset()
    states, actions, rewards = [], [], []
    frames = [] if ep == EPISODES - 1 else None  # Record only last episode

    while True:
        if frames is not None:
            frames.append(env.render())

        action = get_action(state)
        next_state, reward, done, _, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

        if done:
            break

    train_episode(states, actions, rewards)
    episode_rewards.append(sum(rewards))
    print(f"Episode {ep + 1}: Reward = {sum(rewards):.1f}")

    # Save last episode video
    if frames:
        imageio.mimsave('final_episode.mp4', frames, fps=30)

# Plot results
plt.plot(episode_rewards)
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

env.close()

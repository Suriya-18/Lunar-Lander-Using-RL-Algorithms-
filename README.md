# 🛰️ Lunar Lander - Reinforcement Learning Agents

This repository contains multiple implementations of Reinforcement Learning (RL) agents trained on the **LunarLander-v3** environment using various algorithms:

- ✅ Advantage Actor-Critic (A2C)
- ✅ Deep Q-Network (DQN)
- ✅ Monte Carlo Policy Gradient
- ✅ Proximal Policy Optimization (PPO)

Each agent is implemented from scratch using either **TensorFlow**, **PyTorch**, or **Stable-Baselines3**.

---

## 🧠 Algorithms Used

### 1. Advantage Actor-Critic (A2C)
- Built using TensorFlow 2.x
- Separate actor and critic networks
- Computes advantage and updates both networks accordingly
- Includes error handling for NaN/Inf values during training

### 2. Deep Q-Network (DQN)
- Implemented in PyTorch
- Uses replay buffer and target network
- Epsilon-greedy exploration
- Solves environment by averaging over 100 episodes

### 3. Monte Carlo Policy Gradient
- TensorFlow-based policy gradient with Monte Carlo return estimates
- Lightweight network for quick convergence
- Trains on full episodes and updates policy after each

### 4. Proximal Policy Optimization (PPO)
- Uses Stable-Baselines3 implementation
- Trained with parallel environments (`make_vec_env`)
- Tuned hyperparameters for stable and efficient training

---

## 🎮 Environment

- **Name**: `LunarLander-v3` (or `v2` for PPO)
- **Framework**: [Gymnasium](https://gymnasium.farama.org/)
- **Observation Space**: 8-dimensional continuous
- **Action Space**: 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine)

---



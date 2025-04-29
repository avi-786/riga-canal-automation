# Reinforcement Learning for Reservoir Operation
# Complete Colab Notebook

# Install required libraries
!pip install stable-baselines3 gym==0.21.0 matplotlib numpy pandas
!pip install 'shimmy>=2.0'

# Install Pywr - this may take a few minutes
!pip install pywr
!pip install stable_baselines3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
import json
from stable_baselines3 import PPO

# Configure the environment
class ReservoirEnv(gym.Env):
    def __init__(self):
        super(ReservoirEnv, self).__init__()

        # Define action and observation spaces
        self.observation_space = spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # Initialize time and simulation parameters
        self.current_day = 0
        self.simulation_days = 30

        # Initialize reservoir parameters
        self.max_volume = 1000.0
        self.min_volume = 0.0
        self.initial_volume = 500.0
        self.current_volume = self.initial_volume

        # Define constant inflow (can be modified to variable inflow)
        self.inflow = 100.0

        # Reset to initialize state
        self.reset()

    def step(self, action):
        # Extract action (water release amount)
        release_amt = float(action[0])
        release_amt = max(0, min(release_amt, self.current_volume))  # Cannot release more than available

        # Update reservoir volume: previous volume + inflow - release
        new_volume = self.current_volume + self.inflow - release_amt
        new_volume = max(self.min_volume, min(new_volume, self.max_volume))

        # Compute reward components
        demand_met = min(release_amt, 10.0)  # Demand satisfaction (capped at 10)

        # Penalties
        penalty = 0.0
        if new_volume < 50:
            penalty += 5.0  # Too empty
        if new_volume > 950:
            penalty += 5.0  # Too full

        # Compute reward
        reward = demand_met - penalty

        # Update state
        self.current_volume = new_volume
        self.current_day += 1

        # Check if done
        done = (self.current_day >= self.simulation_days)

        # Return state, reward, done, info
        obs = np.array([self.current_volume, self.inflow], dtype=np.float32)
        return obs, reward, done, {}

    def reset(self):
        # Reset reservoir volume
        self.current_volume = self.initial_volume
        self.current_day = 0

        # Return initial observation
        return np.array([self.current_volume, self.inflow], dtype=np.float32)

# Create and train the RL model
def train_model():
    print("Creating environment...")
    env = ReservoirEnv()

    print("Creating PPO model...")
    model = PPO("MlpPolicy", env, verbose=1)

    print("Training model...")
    model.learn(total_timesteps=10000)

    print("Saving model...")
    model.save("ppo_reservoir")

    return model, env

# Evaluate the trained model
def evaluate_model(model, env):
    print("Evaluating model...")
    obs = env.reset()

    # Initialize tracking variables
    storage_levels = []
    inflows = []
    actions = []
    rewards = []

    # Run simulation
    for i in range(60):  # Simulate up to 60 days
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        storage, inflow = obs
        storage_levels.append(storage)
        inflows.append(inflow)
        actions.append(action[0])
        rewards.append(reward)

        if done:
            print(f"Simulation ended after {i+1} days")
            break

    # Plot results
    plot_results(storage_levels, inflows, actions, rewards)

    return storage_levels, inflows, actions, rewards

# Plot the results
def plot_results(storage_levels, inflows, actions, rewards):
    days = np.arange(len(storage_levels))

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(days, storage_levels, label='Reservoir Storage')
    plt.axhline(950, color='red', linestyle='--', label='Flood Threshold')
    plt.axhline(50, color='orange', linestyle='--', label='Empty Threshold')
    plt.ylabel("Storage")
    plt.title("Reservoir Operation Results")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(days, actions, label='Release Action', color='blue')
    plt.ylabel("Action")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(days, inflows, label='Inflow', color='green')
    plt.ylabel("Flow")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(days, rewards, label='Reward', color='red')
    plt.xlabel("Day")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reservoir_results.png")
    plt.show()

# Main execution
print("Starting Reservoir RL simulation...")

# First, train the model
model, env = train_model()

# Then evaluate the trained model
storage_levels, inflows, actions, rewards = evaluate_model(model, env)

print("Simulation completed!")

# Display summary statistics
print("\nSummary Statistics:")
print(f"Average storage level: {np.mean(storage_levels):.2f}")
print(f"Average inflow: {np.mean(inflows):.2f}")
print(f"Average release: {np.mean(actions):.2f}")
print(f"Average reward: {np.mean(rewards):.2f}")
print(f"Minimum storage: {np.min(storage_levels):.2f}")
print(f"Maximum storage: {np.max(storage_levels):.2f}")

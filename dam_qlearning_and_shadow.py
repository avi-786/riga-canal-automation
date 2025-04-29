import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

# Constants
NUM_DAYS = 10  # can be increased
ACTIONS = np.linspace(500, 5000, 10)  # Discharge in m³/s
STATE_BINS = np.linspace(0, 100, 10)  # Reservoir level percent
DAM_CAPACITY = 5896 * 1e6  # Hirakud full capacity in m³

# Initialize Q-tables
Q = np.zeros((len(STATE_BINS), len(ACTIONS)))        # Main agent
Q_shadow = np.zeros((len(STATE_BINS), len(ACTIONS))) # Shadow learning

# For reward tracking
rewards_main = []
rewards_shadow = []
inflow_history = []

# Discretize reservoir level
def discretize_state(reservoir_percent):
    return np.digitize(reservoir_percent, STATE_BINS) - 1

# Environment
class DamEnvironment:
    def __init__(self):
        self.level_percent = 60
        self.current_date = datetime(2025, 4, 1)

    def step(self, action_index):
        discharge_cms = ACTIONS[action_index]
        discharge = discharge_cms * 86400  # m³/day
        inflow = np.random.uniform(500, 2000) * 86400  # Random inflow

        # Volume change
        current_volume = (self.level_percent / 100) * DAM_CAPACITY
        net_change = inflow - discharge
        new_volume = np.clip(current_volume + net_change * 0.5, 0, DAM_CAPACITY)
        self.level_percent = (new_volume / DAM_CAPACITY) * 100
        self.current_date += timedelta(days=1)

        return inflow, discharge, new_volume, self.level_percent

# Phase 1: Active training
def train_agent():
    env = DamEnvironment()
    alpha, gamma = 0.1, 0.9

    for day in range(NUM_DAYS):
        state_idx = discretize_state(env.level_percent)

        # Epsilon-greedy
        epsilon = max(0.1, 1.0 - (day / NUM_DAYS))
        action_idx = (
            np.argmax(Q[state_idx]) if random.random() > epsilon
            else random.randint(0, 9)
        )

        inflow, discharge, volume, level = env.step(action_idx)
        inflow_history.append(inflow)


        # Show status
        print(f"\nDay {day+1} — {env.current_date.strftime('%d %b %Y')}")
        print(f"Reservoir Level: {level:.2f}%")
        print(f"Inflow: {inflow/1e6:.2f} million m³/day")
        print(f"Discharge: {discharge/1e6:.2f} million m³/day")
        print(f"Reservoir Volume: {volume/1e6:.2f} million m³")

        # Scores
        impact = int(input("Flood-Demand Impact (1-best to 100-worst): "))
        electricity = int(100 * np.exp(-((env.level_percent - 80) ** 2) / (2 * 15 ** 2)))
        print(f"Electricity Rating: {electricity}/100 based on {env.level_percent:.2f}% reservoir level")


        # Reward
        reward = np.exp(-0.05 * impact) * 100 + (electricity - 50)
        rewards_main.append(reward)

        # Update Q-table
        next_state_idx = discretize_state(level)
        old_q = Q[state_idx, action_idx]
        Q[state_idx, action_idx] += alpha * (
            reward + gamma * np.max(Q[next_state_idx]) - old_q
        )
        print(f"Q[{state_idx}, {action_idx}] updated from {old_q:.2f} ➜ {Q[state_idx, action_idx]:.2f}")

# Phase 2: Shadow Learning
def shadow_learning():
    env = DamEnvironment()
    alpha, gamma = 0.1, 0.9

    for day in range(NUM_DAYS):
        state_idx = discretize_state(env.level_percent)

        # USER decides discharge
        print(f"\n Shadow Mode — Day {day+1} — {env.current_date.strftime('%d %b %Y')}")
        print(f"Reservoir Level: {env.level_percent:.2f}%")
        user_discharge = float(input("Your discharge decision (in million m³/s): ")*1000000)
        action_idx = np.argmin(np.abs(ACTIONS - user_discharge))

        inflow, discharge, volume, level = env.step(action_idx)

        print(f"Inflow: {inflow/1e6:.2f} million m³/day")
        print(f"Reservoir Volume: {volume/1e6:.2f} million m³")

        # Scores
        impact = int(input("Flood-Demand Impact (1-best to 10-worst): "))
        electricity = int(100 * np.exp(-((env.level_percent - 80) ** 2) / (2 * 15 ** 2)))
        print(f"Electricity Rating: {electricity}/100 based on {env.level_percent:.2f}% reservoir level")
       \ reward = (10 - impact) + (electricity - 5)
        rewards_shadow.append(reward)

        # Update shadow Q-table
        next_state_idx = discretize_state(level)
        old_q = Q_shadow[state_idx, action_idx]
        Q_shadow[state_idx, action_idx] += alpha * (
            reward + gamma * np.max(Q_shadow[next_state_idx]) - old_q
        )
        print(f"👤 Shadow Q[{state_idx}, {action_idx}] updated from {old_q:.2f} ➜ {Q_shadow[state_idx, action_idx]:.2f}")

# Plot rewards
def plot_rewards():
    plt.figure(figsize=(10,5))
    plt.plot(rewards_main, label="Training Mode")
    plt.plot(rewards_shadow, label="Shadow Mode")
    plt.xlabel("Day")
    plt.ylabel("Reward")
    plt.title("Reward Trend Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run both parts
train_agent()
shadow_learning()
plot_rewards()

import matplotlib.pyplot as plt

# Example: Assuming you collected this during training
# inflow_history = [daily inflow in m³/day for each iteration]
# Here's a dummy sample if you haven't collected it yet:
# inflow_history = [random.uniform(500, 2000) * 86400 for _ in range(NUM_DAYS)]

def plot_inflow(inflow_history):
    days = list(range(1, len(inflow_history) + 1))
    inflow_mcm = [val / 1e6 for val in inflow_history]  # Convert to million cubic meters

    plt.figure(figsize=(12, 6))
    plt.plot(days, inflow_mcm, color='dodgerblue', linewidth=2)
    plt.title("Daily Inflow into Reservoir", fontsize=14)
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Inflow (million m³/day)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Example usage
plot_inflow(inflow_history)

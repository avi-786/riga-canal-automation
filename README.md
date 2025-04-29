# River Canal and Dam Operation Automation

## Overview
This repository contains two implementations of reinforcement learning and Q-learning algorithms for automating reservoir and dam operations:

1. **Reservoir Operation with PPO**: Learns an optimal release policy for a reservoir with constant inflow using Proximal Policy Optimization (PPO).
2. **Hirakud Dam Control with Q-Learning & Shadow Learning**: Uses a tabular Q-learning agent supplemented by a shadow learning phase where user feedback refines the policy.

---

## Repository Structure
```bash
.
├── reservoir_operation_rl.ipynb        # Jupyter notebook implementing PPO for reservoir control
├── dam_qlearning_and_shadow.py         # Python script for Q-learning + shadow learning on Hirakud Dam
├── requirements.txt                    # Project dependencies
├── README.md                           # This file
└── reservoir_results.png               # Generated performance plots from the PPO model
```

---

## Requirements
- Python 3.7+
- gym==0.21.0
- stable-baselines3
- shimmy>=2.0
- pywr
- matplotlib
- numpy
- pandas

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Reservoir Operation with PPO

- **Notebook**: Open and run `reservoir_operation_rl.ipynb` in Colab or JupyterLab to:
  - Train the PPO model (`train_model()`)
  - Evaluate and visualize results (`evaluate_model()`)

- **Script** (optional): If converted to Python:
  ```bash
  python reservoir_operation_rl.py
  ```

**Outputs:**
- Trained model file (`ppo_reservoir.zip`)
- Plot image (`reservoir_results.png`)
- Console logs with summary statistics

---

### 2. Hirakud Dam Control with Q-Learning & Shadow Learning

Run the interactive CLI script:
```bash
python dam_qlearning_and_shadow.py
```

**Phases:**
1. **Training Phase** (`train_agent()`): Agent explores discharge actions over multiple days, updating the main Q-table based on simulated inflows and user‑scored impact metrics.
2. **Shadow Learning Phase** (`shadow_learning()`): User provides discharge decisions and flood‑demand impact scores to update a secondary (shadow) Q-table.
3. **Visualization**: After both phases, `plot_rewards()` and `plot_inflow()` display reward trends and daily inflow patterns.

---

## Configuration & Customization
All key parameters are defined at the top of each file:
- **`NUM_DAYS`**: Total days per training phase
- **`ACTIONS`**: Discrete discharge rates
- **`STATE_BINS`**: Reservoir-level discretization
- **`DAM_CAPACITY`**: Maximum reservoir volume

Adjust these constants to fit different scenarios or reservoir specifications.

---

## Results
- Inspect `reservoir_results.png` for PPO performance (storage, release, reward over time).
- In the Q-learning script, reward and inflow plots appear interactively at the end.

---

## License
This project is released under the MIT License. Feel free to use and modify as needed.



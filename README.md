# 🚀 Autonomous Lunar Lander: Deep Q-Learning & Evolutionary Algorithms

This repository implements and evaluates **Deep Q-Network (DQN)** and **Genetic Algorithm (GA)** agents to solve the `LunarLander-v3` environment from Gymnasium. The primary objective is to achieve **ultra-precise, fuel-efficient, and perfectly centered landings** ($\vert{}X\vert{} \le 0.08$) while eliminating hovering, leg bounce, or out-of-bounds crashes.

---

## 📸 Visual Showcase & Media Setup

> **Media Setup Guide:**
> 1. Save your captured landing GIF to `assets/landing_demo.gif`.
> 2. Save your generated comparison charts inside the `assets/` directory.

| Perfect Landing Demonstration (DQN Full) |
| :---: |
| ![Lunar Lander Demo](assets/landing_demo.gif) |
| *Agent executing controlled descent, zeroing thrusters upon touchdown, and centering strictly between the flags.* |

---

## 🛠️ Custom Wrapper Architecture (`wrappers.py`)

To enforce strict, realistic flight dynamics and eliminate exploit behaviors (such as infinite hovering or bouncing), the raw environment is wrapped with `LimitedFuelAndImpactWrapper`:

* **Capped Fuel Capacity (200.0 Units):** Deducts fuel on every main ($0.8$) and side ($0.2$) engine firing.
* **Narrow Target Pad ($\vert{}X\vert{} \le 0.08$):** Rewards landing strictly on the center pad while penalizing lateral drift.
* **Ground Thruster Lockout:** Applies severe penalties for firing engines while leg contact is active, forcing immediate engine cutoff.
* **Strict Landing Conditions:** Requires simultaneous **dual-leg ground contact**, **zero active thrusters**, and a **10-step stability hold**.
* **Out-of-Bounds Enforcement (-300.0 Penalty):** Immediately terminates episodes if the vehicle drifts off-screen.

---

## 📊 Performance Analysis & Comparisons

### 1. Training Progression: 300 vs 600 vs 1000 Episodes

![Training Progression](assets/training_comparison_300_600_1000.png)

Evaluating model checkpoints across training epochs illustrates the clear convergence curve of the DQN policy:

* **300 Episodes (Underfitting):** The network has not converged. It exhibits a **0.0% success rate**, a severe average negative reward (**-517.0**), and depletes almost all fuel (**16.0 units left** out of 200.0) in chaotic attempts.
* **600 Episodes (Intermediate Convergence):** Success rate jumps to **38.0%** with a positive average reward of **230.9**, leaving **81.5 fuel units**.
* **1000 Episodes (Fully Trained):** Achieves a **64.0% Perfect Landing success rate**, **416.7 average reward**, cuts flight duration down to **185.8 steps**, and preserves **122.1 fuel units** (>60% of tank capacity).

---

### 2. DQN Full vs DQN Incomplete (Reward Shaping Impact)

![DQN Full vs Incomplete](assets/dqn_full_vs_incomplete.png)

Comparing the fully tuned agent (`DQN Full`) against a baseline trained without anti-hovering penalties and tight centering constraints (`DQN Incomplete`):

| Metric / Behavior | DQN Full (Strict Wrapper) | DQN Incomplete (Unrestricted Wrapper) |
| :--- | :---: | :---: |
| **Landing Alignment ($\vert{}X\vert{} \le 0.08$)** | 🚀 **Strictly Centered** | ⚠️ Off-Target / Drift |
| **Hovering & Leg Bouncing** | ❌ **Eliminated** | ⚠️ High Occurrence |
| **Engine Cutoff at Touchdown** | 🛑 **Immediate** | ⚠️ Continuous Fuel Waste |
| **Average Reward** | **+416.7** | ~ +180.0 |
| **Remaining Fuel Efficiency** | **122.1 / 200.0** | < 45.0 / 200.0 |

---

### 3. DQN vs Genetic Algorithm (RL vs Evolutionary Approach)

![DQN vs Genetic Algorithm](assets/dqn_vs_genetic.png)

A comparative evaluation between Deep Reinforcement Learning (DQN) and Evolutionary Optimization (Genetic Algorithm):

* **Deep Q-Network (DQN):**
  * **Strengths:** Superior generalization capability when handling random initial state displacements ($\vert{}X_{init}\vert{} > 0.20$) and high initial lateral velocities.
  * **Performance:** Consistently stabilizes the lander over required hold-step windows.
* **Genetic Algorithm (GA):**
  * **Strengths:** Rapid convergence on fixed trajectories with zero gradient-computation overhead.
  * **Limitations:** Higher sensitivity to unseen random seeds, resulting in lower success rates when environment initial conditions vary significantly.

---

## 💻 Repository Structure

```text
.
├── assets/                                  # Visual assets for documentation
│   ├── landing_demo.gif                     # Recorded GIF of optimal landing
│   ├── landing_demo2.gif                     # Recorded GIF of optimal landing
│   ├── landing_demo3.gif                     # Recorded GIF of optimal landing
│   ├── training_comparison_300_600_1000.png # 3-checkpoint comparative dashboard
│   ├── dqn_full_vs_incomplete.png           # Reward shaping comparison chart
│   └── dqn_vs_genetic.png                   # RL vs GA comparison chart
├── models/                                  # Saved PyTorch model checkpoints
│   ├── dqn_300.pt
│   ├── dqn_600.pt
│   ├── dqn_1000.pt
│   └── dqn_full.pt
├── agent.py                                 # DQNAgent class & PyTorch network architecture
├── wrappers.py                              # LimitedFuelAndImpactWrapper implementation
├── train.py                                 # Main training loop script
├── Evaluate.py                              # Evaluation script with terminal diagnostics
├── generate_gif.py                          # Smart GIF generator filtered by fuel economy
├── compare_models.py                        # Automated benchmark and plot generator
└── README.md                                # Single-file project documentation
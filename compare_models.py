import os
import argparse
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from agent import DQNAgent
from wrappers import LimitedFuelAndImpactWrapper


def load_agent(model_path, state_dim, action_dim):
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    if hasattr(agent, "load") and callable(getattr(agent, "load")):
        agent.load(model_path)
    elif hasattr(agent, "q_network"):
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.q_network.eval()
    elif hasattr(agent, "model"):
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()
    elif hasattr(agent, "policy_net"):
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.policy_net.eval()
        
    # Standardize greedy evaluation
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
        
    return agent


def evaluate_model(model_path, max_fuel=200.0, num_episodes=50):
    raw_env = gym.make("LunarLander-v3")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=max_fuel)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Checkpoint file '{model_path}' not found! Returning zeroed metrics.")
        return {
            "rewards": [0], "fuel": [0], "steps": [0],
            "success_rate": 0.0, "avg_reward": 0.0, "avg_fuel": 0.0, "avg_steps": 0.0
        }

    agent = load_agent(model_path, state_dim, action_dim)

    rewards = []
    fuel_leftover = []
    flight_steps = []
    perfect_landings = 0

    print(f"📊 Evaluating '{model_path}' (max_fuel={max_fuel}) over {num_episodes} episodes...")

    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            with torch.no_grad():
                try:
                    action = agent.select_action(state, epsilon=0.0)
                except TypeError:
                    try:
                        action = agent.select_action(state, 0.0)
                    except TypeError:
                        action = agent.select_action(state)

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        status = info.get("impact_status", "")
        fuel = info.get("fuel_left", 0.0)

        rewards.append(total_reward)
        fuel_leftover.append(fuel)
        flight_steps.append(steps)
        
        if "PERFECT LANDING" in status:
            perfect_landings += 1

    env.close()

    success_rate = (perfect_landings / num_episodes) * 100
    return {
        "rewards": rewards,
        "fuel": fuel_leftover,
        "steps": flight_steps,
        "success_rate": success_rate,
        "avg_reward": np.mean(rewards),
        "avg_fuel": np.mean(fuel_leftover),
        "avg_steps": np.mean(flight_steps),
    }


def plot_triple_comparison(results_300, results_600, results_1000, max_fuel=200.0, output_path="training_comparison_300_600_1000.png"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Progression: 300 vs 600 vs 1000 Episodes (Max Fuel: {max_fuel})", fontsize=15, fontweight="bold")

    labels = ["300 Episodes", "600 Episodes", "1000 Episodes"]
    colors = ["#FF5722", "#FFC107", "#4CAF50"]

    # 1. Recompensa Média
    avg_rewards = [results_300["avg_reward"], results_600["avg_reward"], results_1000["avg_reward"]]
    axs[0, 0].bar(labels, avg_rewards, color=colors, alpha=0.85)
    axs[0, 0].set_title("Average Episode Reward")
    axs[0, 0].set_ylabel("Reward Points")
    for i, v in enumerate(avg_rewards):
        axs[0, 0].text(i, v / 2 if v > 0 else v + 20, f"{v:.1f}", ha="center", fontweight="bold")

    # 2. Taxa de Sucesso (%)
    success_rates = [results_300["success_rate"], results_600["success_rate"], results_1000["success_rate"]]
    axs[0, 1].bar(labels, success_rates, color=colors, alpha=0.85)
    axs[0, 1].set_title("Perfect Landing Success Rate (%)")
    axs[0, 1].set_ylabel("Percentage (%)")
    axs[0, 1].set_ylim(0, 105)
    for i, v in enumerate(success_rates):
        axs[0, 1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    # 3. Combustível Restante Médio
    avg_fuels = [results_300["avg_fuel"], results_600["avg_fuel"], results_1000["avg_fuel"]]
    axs[1, 0].bar(labels, avg_fuels, color=colors, alpha=0.85)
    axs[1, 0].set_title("Average Fuel Remaining at Touchdown")
    axs[1, 0].set_ylabel(f"Fuel Units Left (out of {max_fuel})")
    axs[1, 0].set_ylim(0, max_fuel)
    for i, v in enumerate(avg_fuels):
        axs[1, 0].text(i, v / 2 if v > 0 else 10, f"{v:.1f}", ha="center", fontweight="bold")

    # 4. Duração Média do Voo (Passos)
    avg_steps = [results_300["avg_steps"], results_600["avg_steps"], results_1000["avg_steps"]]
    axs[1, 1].bar(labels, avg_steps, color=colors, alpha=0.85)
    axs[1, 1].set_title("Average Flight Duration (Steps)")
    axs[1, 1].set_ylabel("Steps Count")
    for i, v in enumerate(avg_steps):
        axs[1, 1].text(i, v / 2 if v > 0 else 10, f"{v:.1f} steps", ha="center", fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"\n📈 Comparison dashboard saved to '{output_path}'!")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare performance across 300, 600, and 1000 episode checkpoints.")
    parser.add_argument("--model300", type=str, default="models/dqn_300.pt", help="Path to 300 episode checkpoint")
    parser.add_argument("--model600", type=str, default="models/dqn_600.pt", help="Path to 600 episode checkpoint")
    parser.add_argument("--model1000", type=str, default="models/dqn_1000.pt", help="Path to 1000 episode checkpoint")
    parser.add_argument("--max_fuel", type=float, default=200.0, help="Maximum starting fuel for environment")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes per checkpoint")

    args = parser.parse_args()

    res_300 = evaluate_model(args.model300, max_fuel=args.max_fuel, num_episodes=args.episodes)
    res_600 = evaluate_model(args.model600, max_fuel=args.max_fuel, num_episodes=args.episodes)
    res_1000 = evaluate_model(args.model1000, max_fuel=args.max_fuel, num_episodes=args.episodes)

    plot_triple_comparison(res_300, res_600, res_1000, max_fuel=args.max_fuel)
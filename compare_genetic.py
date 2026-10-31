import os
import argparse
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from agent import DQNAgent
from wrappers import LimitedFuelAndImpactWrapper


def load_dqn_agent(model_path, state_dim, action_dim):
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
        
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    return agent


def run_evaluation(agent_type, model_path, max_fuel=200.0, num_runs=50):
    raw_env = gym.make("LunarLander-v3")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=max_fuel)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    rewards = []
    perfect_landings = 0

    if agent_type == "dqn":
        agent = load_dqn_agent(model_path, state_dim, action_dim)
    elif agent_type == "genetic":
        # Load weights or policy for Genetic Algorithm model checkpoint
        genetic_weights = torch.load(model_path)

    print(f"📊 Running {num_runs} test runs for [{agent_type.upper()}] using '{model_path}'...")

    for run in range(num_runs):
        state, info = env.reset(seed=42 + run)  # Standardized seeds across both models
        done = False
        total_reward = 0.0

        while not done:
            if agent_type == "dqn":
                with torch.no_grad():
                    try:
                        action = agent.select_action(state, epsilon=0.0)
                    except TypeError:
                        try:
                            action = agent.select_action(state, 0.0)
                        except TypeError:
                            action = agent.select_action(state)
            elif agent_type == "genetic":
                # Linear/MLP evaluation for GA
                if isinstance(genetic_weights, torch.nn.Module):
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0)
                        action = torch.argmax(genetic_weights(state_t)).item()
                else:
                    # Matrix multiplication for raw weight arrays
                    action = np.argmax(np.dot(state, genetic_weights))

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if "PERFECT LANDING" in info.get("impact_status", ""):
            perfect_landings += 1

    env.close()
    return rewards, (perfect_landings / num_runs) * 100


def plot_runs_comparison(dqn_rewards, ga_rewards, dqn_acc, ga_acc, output_path="assets/dqn_vs_genetic.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    runs = np.arange(1, len(dqn_rewards) + 1)

    # 1. Rewards per Run (Line Plot)
    axs[0].plot(runs, dqn_rewards, label=f"DQN 1000 (Success: {dqn_acc:.1f}%)", color="#4CAF50", linewidth=2.0)
    axs[0].plot(runs, ga_rewards, label=f"Genetic Alg (Success: {ga_acc:.1f}%)", color="#FF9800", linewidth=1.8, linestyle="--")
    axs[0].set_title("Reward per Evaluation Run", fontsize=13, fontweight="bold")
    axs[0].set_xlabel("Run Index")
    axs[0].set_ylabel("Reward Points")
    axs[0].grid(True, linestyle=":", alpha=0.6)
    axs[0].legend(loc="upper right")

    # 2. Reward Distribution (Box Plot)
    data = [dqn_rewards, ga_rewards]
    axs[1].boxplot(data, tick_labels=["DQN 1000 Episodes", "Genetic Algorithm"], patch_artist=True,
                   boxprops=dict(facecolor="#4CAF50", alpha=0.7),
                   medianprops=dict(color="black", linewidth=2))
    axs[1].set_title("Reward Distribution & Stability", fontsize=13, fontweight="bold")
    axs[1].set_ylabel("Reward Points")
    axs[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n📈 Run comparison chart saved to '{output_path}'!")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DQN 1000 vs Genetic Algorithm across test runs.")
    parser.add_argument("--dqn_model", type=str, default="models/dqn_1000.pt", help="Path to DQN 1000 model")
    parser.add_argument("--ga_model", type=str, default="models/genetic_best.pt", help="Path to Genetic Algorithm model")
    parser.add_argument("--max_fuel", type=float, default=200.0, help="Maximum starting fuel")
    parser.add_argument("--runs", type=int, default=50, help="Number of evaluation runs")

    args = parser.parse_args()

    dqn_rewards, dqn_acc = run_evaluation("dqn", args.dqn_model, max_fuel=args.max_fuel, num_runs=args.runs)
    ga_rewards, ga_acc = run_evaluation("genetic", args.ga_model, max_fuel=args.max_fuel, num_runs=args.runs)

    plot_runs_comparison(dqn_rewards, ga_rewards, dqn_acc, ga_acc)
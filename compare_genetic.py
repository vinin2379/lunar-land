import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from agent import DQNAgent
from wrappers import LimitedFuelAndImpactWrapper


class GeneticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


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


def load_genetic_model(model_path, state_dim, action_dim):
    loaded_data = torch.load(model_path)

    if isinstance(loaded_data, nn.Module):
        loaded_data.eval()
        return loaded_data
    elif isinstance(loaded_data, (dict, torch.nn.modules.container.Sequential)):
        model = GeneticNetwork(state_dim, action_dim)
        try:
            model.load_state_dict(loaded_data)
        except Exception:
            pass
        model.eval()
        return model

    return loaded_data


def run_evaluation(agent_type, model_path, max_fuel=200.0, num_runs=50):
    raw_env = gym.make("LunarLander-v3")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=max_fuel)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    rewards = []
    landings_history = []  # 1 para PERFECT LANDING, 0 para falha

    if agent_type == "dqn":
        agent = load_dqn_agent(model_path, state_dim, action_dim)
    elif agent_type == "genetic":
        ga_model = load_genetic_model(model_path, state_dim, action_dim)

    print(f"📊 Running {num_runs} test runs for [{agent_type.upper()}] using '{model_path}'...")

    for run in range(num_runs):
        state, info = env.reset(seed=42 + run)
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
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    if isinstance(ga_model, nn.Module):
                        logits = ga_model(state_t)
                        action = torch.argmax(logits, dim=1).item()
                    elif isinstance(ga_model, dict):
                        w = ga_model.get("weight", ga_model.get("fc.0.weight", None))
                        if w is not None:
                            action = np.argmax(np.dot(state, w.cpu().numpy().T))
                        else:
                            action = env.action_space.sample()
                    else:
                        action = np.argmax(np.dot(state, ga_model))

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        is_perfect = "PERFECT LANDING" in info.get("impact_status", "")
        landings_history.append(1 if is_perfect else 0)

    env.close()
    return rewards, landings_history


def plot_runs_comparison(dqn_rewards, ga_rewards, dqn_landings, ga_landings, output_path="assets/dqn_vs_genetic.png"):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    runs = np.arange(1, len(dqn_rewards) + 1)
    
    dqn_cumulative = np.cumsum(dqn_landings)
    ga_cumulative = np.cumsum(ga_landings)

    dqn_total_landings = dqn_cumulative[-1]
    ga_total_landings = ga_cumulative[-1]
    total_runs = len(runs)

    # 1. Recompensa por Partida
    axs[0].plot(runs, dqn_rewards, label=f"DQN 1000", color="#4CAF50", linewidth=2.0)
    axs[0].plot(runs, ga_rewards, label=f"Genetic Alg", color="#FF9800", linewidth=1.8, linestyle="--")
    axs[0].set_title("Reward per Evaluation Run", fontsize=13, fontweight="bold")
    axs[0].set_xlabel("Run Index")
    axs[0].set_ylabel("Reward Points")
    axs[0].grid(True, linestyle=":", alpha=0.6)
    axs[0].legend(loc="upper right")

    # 2. Pousos Perfeitos Acumulados (+1 a cada acerto)
    axs[1].plot(runs, dqn_cumulative, label=f"DQN 1000 ({dqn_total_landings}/{total_runs} Landings)", color="#4CAF50", linewidth=2.5)
    axs[1].plot(runs, ga_cumulative, label=f"Genetic Alg ({ga_total_landings}/{total_runs} Landings)", color="#FF9800", linewidth=2.5, linestyle="--")
    axs[1].set_title("Accumulated Perfect Landings Count", fontsize=13, fontweight="bold")
    axs[1].set_xlabel("Run Index")
    axs[1].set_ylabel("Total Successful Landings (+1 point per landing)")
    axs[1].grid(True, linestyle=":", alpha=0.6)
    axs[1].legend(loc="upper left")

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

    dqn_rewards, dqn_landings = run_evaluation("dqn", args.dqn_model, max_fuel=args.max_fuel, num_runs=args.runs)
    ga_rewards, ga_landings = run_evaluation("genetic", args.ga_model, max_fuel=args.max_fuel, num_runs=args.runs)

    plot_runs_comparison(dqn_rewards, ga_rewards, dqn_landings, ga_landings)
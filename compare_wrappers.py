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


def prepare_state(state, expected_dim=9):
    """Garante que o estado tenha 9 dimensões (preenche fuel_ratio se rodar no env padrão)."""
    if len(state) < expected_dim:
        padding = np.ones(expected_dim - len(state), dtype=np.float32)
        state = np.append(state, padding)
    return state.astype(np.float32)


def evaluate_on_env(model_path, env_type="custom", max_fuel=200.0, num_runs=50):
    raw_env = gym.make("LunarLander-v3")
    
    if env_type == "custom":
        env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=max_fuel)
    else:
        env = raw_env

    expected_state_dim = 9
    action_dim = env.action_space.n
    agent = load_dqn_agent(model_path, expected_state_dim, action_dim)

    rewards = []
    landings_history = []

    print(f"📊 Running {num_runs} evaluation runs on [{env_type.upper()}] environment...")

    for run in range(num_runs):
        state, info = env.reset(seed=42 + run)
        done = False
        total_reward = 0.0

        while not done:
            state = prepare_state(state, expected_state_dim)
            with torch.no_grad():
                try:
                    action = agent.select_action(state)
                except TypeError:
                    action = agent.select_action(state, 0.0)

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        
        # No env customizado exige a flag PERFECT LANDING. No padrão considera recompensa >= 200
        if env_type == "custom":
            is_success = "PERFECT LANDING" in info.get("impact_status", "")
        else:
            is_success = total_reward >= 100.0
            
        landings_history.append(1 if is_success else 0)

    env.close()
    return rewards, landings_history


def plot_wrapper_comparison(custom_rewards, std_rewards, custom_landings, std_landings, output_path="assets/dqn_custom_vs_standard.png"):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    runs = np.arange(1, len(custom_rewards) + 1)
    
    custom_cumulative = np.cumsum(custom_landings)
    std_cumulative = np.cumsum(std_landings)

    custom_total = custom_cumulative[-1]
    std_total = std_cumulative[-1]
    total_runs = len(runs)

    # 1. Recompensa por Partida
    axs[0].plot(runs, std_rewards, label="Standard Env (Raw LunarLander-v3)", color="#2196F3", linewidth=1.8)
    axs[0].plot(runs, custom_rewards, label="Custom Env (Strict Wrapper)", color="#4CAF50", linewidth=2.0)
    axs[0].set_title("Reward per Evaluation Run", fontsize=13, fontweight="bold")
    axs[0].set_xlabel("Run Index")
    axs[0].set_ylabel("Reward Points")
    axs[0].grid(True, linestyle=":", alpha=0.6)
    axs[0].legend(loc="lower right")

    # 2. Pousos Bem-Sucedidos Acumulados
    axs[1].plot(runs, std_cumulative, label=f"Standard Env ({std_total}/{total_runs} Landings)", color="#2196F3", linewidth=2.5)
    axs[1].plot(runs, custom_cumulative, label=f"Custom Env ({custom_total}/{total_runs} Landings)", color="#4CAF50", linewidth=2.5)
    axs[1].set_title("Accumulated Successful Landings", fontsize=13, fontweight="bold")
    axs[1].set_xlabel("Run Index")
    axs[1].set_ylabel("Total Successful Landings (+1 point per landing)")
    axs[1].grid(True, linestyle=":", alpha=0.6)
    axs[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n📈 Wrapper comparison chart saved to '{output_path}'!")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DQN 1000 on Custom vs Standard Environment.")
    parser.add_argument("--model", type=str, default="models/dqn_1000.pt", help="Path to DQN 1000 model")
    parser.add_argument("--max_fuel", type=float, default=200.0, help="Maximum starting fuel for custom env")
    parser.add_argument("--runs", type=int, default=50, help="Number of evaluation runs")

    args = parser.parse_args()

    custom_rewards, custom_landings = evaluate_on_env(args.model, env_type="custom", max_fuel=args.max_fuel, num_runs=args.runs)
    std_rewards, std_landings = evaluate_on_env(args.model, env_type="standard", max_fuel=args.max_fuel, num_runs=args.runs)

    plot_wrapper_comparison(custom_rewards, std_rewards, custom_landings, std_landings)
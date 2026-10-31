import matplotlib.pyplot as plt
import numpy as np
from train import train_agent


def moving_average(data, window=20):
    """Calculates moving average to smooth out reward fluctuations in plots."""
    return np.convolve(data, np.ones(window) / window, mode="valid")


def main():
    episodes = 300

    print("==========================================")
    print(" Running Ablation Study: Model Comparison")
    print("==========================================")

    # 1. Train Baseline Model (Saves as baseline_no_exploration.pt)
    rewards_simple = train_agent(
        use_exploration=False,
        simple_arch=True,
        model_name="baseline_no_exploration",
        episodes=episodes,
    )

    # 2. Train Full DQN Model (Saves as dqn_full.pt)
    rewards_full = train_agent(
        use_exploration=True,
        simple_arch=False,
        model_name="dqn_full",
        episodes=episodes,
    )

    # 3. Plot Comparison Graph
    plt.figure(figsize=(10, 5))
    plt.plot(
        moving_average(rewards_simple),
        label="Simple (1 Layer, No Epsilon)",
        color="crimson",
    )
    plt.plot(
        moving_average(rewards_full),
        label="Full DQN (2 Layers, Epsilon Decay)",
        color="forestgreen",
    )

    plt.axhline(
        y=200,
        color="royalblue",
        linestyle="--",
        label="Environment Solved Threshold (+200)",
    )
    plt.title(
        "Ablation Study: Impact of Exploration & Neural Network Depth",
        fontsize=12,
    )
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Reward (20 episodes)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    output_plot = "comparison_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot successfully generated and saved to '{output_plot}'!")


if __name__ == "__main__":
    main()
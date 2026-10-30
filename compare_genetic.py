import argparse
import matplotlib.pyplot as plt
import numpy as np
from train import train_agent
from genetic_train import train_genetic


def moving_average(data, window=10):
    """Calculates moving average to smooth out reward fluctuations."""
    return np.convolve(data, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Deep Q-Learning (DQN) against Genetic Algorithm (Neuroevolution)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=150,
        help="Number of Episodes (DQN) / Generations (Genetic Algorithm). Default: 300",
    )
    args = parser.parse_args()

    print("\n==========================================")
    print(" 1/2 Training Deep Q-Network (DQN) ")
    print("==========================================")
    rewards_dqn = train_agent(
        use_exploration=True,
        simple_arch=False,
        model_name="dqn_full",
        episodes=args.episodes,
    )

    print("\n==========================================")
    print(" 2/2 Training Genetic Algorithm (Neuroevolution) ")
    print("==========================================")
    rewards_genetic = train_genetic(
        population_size=30,
        generations=args.episodes,
    )

    print("\n==========================================")
    print(" Generating Comparison Plot ")
    print("==========================================")

    plt.figure(figsize=(10, 5))
    plt.plot(
        moving_average(rewards_dqn),
        label="Deep Q-Network (DQN)",
        color="forestgreen",
        linewidth=1.8,
    )
    plt.plot(
        moving_average(rewards_genetic),
        label="Genetic Algorithm (Neuroevolution)",
        color="darkorange",
        linewidth=1.8,
    )

    plt.axhline(
        y=200,
        color="royalblue",
        linestyle="--",
        label="Environment Solved Threshold (+200)",
    )
    plt.title(
        "Paradigms Comparison: Deep Q-Learning vs. Neuroevolution",
        fontsize=12,
        pad=10,
    )
    plt.xlabel("Episodes / Generations")
    plt.ylabel("Reward / Best Fitness")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    output_plot = "dqn_vs_genetic.png"
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"\nPlot successfully generated and saved as '{output_plot}'!")


if __name__ == "__main__":
    main()
import argparse
import os
import gymnasium as gym
import torch
from agent import QNetwork
from genetic_train import RocketBrain
from wrappers import LimitedFuelAndImpactWrapper


def evaluate_best_model(model_path, model_type="dqn"):
    """Loads the best trained model and runs 5 visual evaluation episodes."""
    raw_env = gym.make("LunarLander-v3", render_mode="human")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=200.0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if not os.path.exists(model_path):
        print(f"⚠️ Model file '{model_path}' not found! Train it first.")
        return

    # Instantiate appropriate network architecture
    if model_type == "genetic":
        model = RocketBrain(state_dim, action_dim)
    else:
        simple_arch = "baseline" in model_path
        model = QNetwork(state_dim, action_dim, simple_arch=simple_arch)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"\n🚀 Running 5 evaluation episodes for [{model_type.upper()}]: '{model_path}'...\n")

    for episode in range(1, 6):
        state, _ = env.reset()
        total_reward = 0
        done = False
        last_info = {}

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            last_info = info

        print(
            f"Episode {episode}: Reward = {total_reward:6.2f} | "
            f"Fuel Left = {last_info.get('fuel_left', 0):.1f} | "
            f"{last_info.get('impact_status', '')}"
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the best trained LunarLander model visually"
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "full", "genetic"],
        default="full",
        help="Select model to evaluate: 'simple', 'full', or 'genetic' (default: 'full')",
    )
    args = parser.parse_args()

    if args.mode == "simple":
        evaluate_best_model("models/baseline_no_exploration.pt", model_type="dqn")
    elif args.mode == "full":
        evaluate_best_model("models/dqn_full.pt", model_type="dqn")
    elif args.mode == "genetic":
        evaluate_best_model("models/genetic_best.pt", model_type="genetic")
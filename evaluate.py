import argparse
import os
import gymnasium as gym
import torch
from agent import QNetwork
from wrappers import LimitedFuelAndImpactWrapper


def evaluate_milestones(model_base_name, simple_arch):
    """Loads milestone checkpoints and renders one evaluation episode for each."""
    milestones = [1, 100, 150, 200, 300]
    raw_env = gym.make("LunarLander-v3", render_mode="human")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=80.0)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"\n🚀 Running Milestone Evaluation for: '{model_base_name}'...\n")

    for ep in milestones:
        model_path = f"models/{model_base_name}_ep{ep}.pt"

        if not os.path.exists(model_path):
            print(f"⚠️ Checkpoint file '{model_path}' not found! Train the model first.")
            continue

        model = QNetwork(state_dim, action_dim, simple_arch=simple_arch)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print(f"--- Visualizing Agent at Episode {ep} ---")

        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(
            f"Result (Ep {ep:3d}): Reward = {total_reward:6.2f} | "
            f"Fuel Left = {info.get('fuel_left', 0):.1f} | "
            f"{info.get('impact_status', '')}\n"
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained LunarLander model milestones visually"
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "full"],
        default="full",
        help="Choose model to evaluate: 'simple' or 'full'",
    )
    args = parser.parse_args()

    if args.mode == "simple":
        evaluate_milestones("baseline_no_exploration", simple_arch=True)
    else:
        evaluate_milestones("dqn_full", simple_arch=False)
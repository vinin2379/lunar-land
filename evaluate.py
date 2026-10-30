import argparse
import gymnasium as gym
import torch
from agent import QNetwork


def evaluate(model_path, simple_arch):
    """Loads a trained model and renders evaluation episodes in real-time."""
    env = gym.make("LunarLander-v3", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = QNetwork(state_dim, action_dim, simple_arch=simple_arch)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"Running visual evaluation for model: '{model_path}'...")

    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(
            f"Evaluation Episode {episode + 1}: Total Reward = {total_reward:.2f}"
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained LunarLander model visually"
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "full"],
        default="full",
        help="Choose model to evaluate: 'simple' or 'full'",
    )
    args = parser.parse_args()

    if args.mode == "simple":
        evaluate("models/baseline_no_exploration.pt", simple_arch=True)
    else:
        evaluate("models/dqn_full.pt", simple_arch=False)
import os
import argparse
import torch
import numpy as np
import gymnasium as gym
import imageio
from agent import DQNAgent
from wrappers import LimitedFuelAndImpactWrapper
# Garante o parâmetro loop=0 para repetir infinitamente no navegador


def generate_best_gif(model_path="models/dqn_full.pt", output_gif="landing_demo.gif", max_fuel=200.0, max_trials=10):
    os.makedirs(os.path.dirname(output_gif) if os.path.dirname(output_gif) else ".", exist_ok=True)

    raw_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=max_fuel)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    # Load weights
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

    # FORCE EPSILON TO 0.0 (Greedy Mode)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    best_frames = []
    best_reward = -float("inf")
    best_status = "N/A"

    print(f"🎥 Searching for a PERFECT LANDING episode from '{model_path}'...")

    for trial in range(1, max_trials + 1):
        frames = []
        state, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            frames.append(env.render())

            # Select deterministic greedy action
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
            done = terminated or truncated

        status = info.get("impact_status", "N/A")
        print(f"   Trial {trial}/{max_trials} | Reward: {total_reward:.2f} | Status: {status}")

        if "PERFECT LANDING" in status and total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames
            best_status = status
            print(f"   ⭐ Captured Perfect Landing! Saving GIF.")
            break
        elif total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames
            best_status = status

    env.close()

    if best_frames:
        imageio.mimsave(output_gif, best_frames, fps=30, loop = 0)
        print(f"✅ GIF successfully saved to '{output_gif}'!")
        print(f"📊 Captured Episode Reward: {best_reward:.2f} | Status: {best_status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GIF for trained LunarLander Agent.")
    parser.add_argument("--model", type=str, default="models/dqn_full.pt", help="Path to saved model checkpoint")
    parser.add_argument("--output", type=str, default="landing_demo.gif", help="Output GIF filename")
    parser.add_argument("--max_fuel", type=float, default=500.0, help="Maximum starting fuel")

    args = parser.parse_args()
    generate_best_gif(model_path=args.model, output_gif=args.output, max_fuel=args.max_fuel)
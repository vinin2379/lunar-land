import argparse
import os
import gymnasium as gym
import numpy as np
import torch
from agent import DQNAgent
from wrappers import LimitedFuelAndImpactWrapper


def train_agent(
    use_exploration, simple_arch, model_name, episodes=300, verbose=True
):
    """Executes training loop with fuel limits and saves milestone checkpoints."""
    raw_env = gym.make("LunarLander-v3")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=100.0)
    

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim,
        action_dim,
        use_exploration=use_exploration,
        simple_arch=simple_arch,
    )
    rewards_history = []
    milestones = [1, 100, 150, 200, 300]

    os.makedirs("models", exist_ok=True)

    if verbose:
        print(f"\n--- Starting Training: {model_name} ---")

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        last_info = {}

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward
            last_info = info

        # Decay exploration rate once per episode
        if agent.use_exploration and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode % 5 == 0:
            agent.update_target_network()

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])

        # Save model checkpoint at specific milestones
        if episode in milestones:
            checkpoint_path = f"models/{model_name}_ep{episode}.pt"
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            if verbose:
                print(f" [CHECKPOINT] Saved milestone model: {checkpoint_path}")

        if verbose and (episode % 10 == 0 or episode in milestones):
            impact = last_info.get("impact_status", "N/A")
            fuel = last_info.get("fuel_left", 0)
            print(
                f"Episode: {episode:3d}/{episodes} | "
                f"Reward: {total_reward:6.1f} | "
                f"Avg (100 ep): {avg_reward:6.1f} | "
                f"Epsilon: {agent.epsilon:.2f} | "
                f"Fuel: {fuel:.1f} | "
                f"{impact}"
            )

    # Save final model state
    save_path = f"models/{model_name}.pt"
    torch.save(agent.policy_net.state_dict(), save_path)

    env.close()
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LunarLander RL Agent with customizable mode"
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "full"],
        default="full",
        help="Select model configuration: 'simple' (baseline) or 'full' (DQN)",
    )
    args = parser.parse_args()

    if args.mode == "simple":
        train_agent(
            use_exploration=False,
            simple_arch=True,
            model_name="baseline_no_exploration",
        )
    else:
        train_agent(
            use_exploration=True, simple_arch=False, model_name="dqn_full"
        )
import os
import gymnasium as gym
import numpy as np
import torch
from agent import DQNAgent


def train_agent(
    use_exploration, simple_arch, model_name, episodes=300, verbose=True
):
    """Executes the training loop for a given agent configuration and returns reward history."""
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim,
        action_dim,
        use_exploration=use_exploration,
        simple_arch=simple_arch,
    )
    rewards_history = []

    if verbose:
        print(f"\n--- Starting Training: {model_name} ---")

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        if episode % 5 == 0:
            agent.update_target_network()

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])

        if verbose and (episode % 20 == 0 or episode == 1):
            print(
                f"Episode: {episode:3d}/{episodes} | "
                f"Reward: {total_reward:6.1f} | "
                f"Avg (100 ep): {avg_reward:6.1f} | "
                f"Epsilon: {agent.epsilon:.2f}"
            )

    os.makedirs("models", exist_ok=True)
    save_path = f"models/{model_name}.pt"
    torch.save(agent.policy_net.state_dict(), save_path)
    env.close()

    return rewards_history
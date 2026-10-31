import argparse
import torch
import numpy as np
import gymnasium as gym
from agent import DQNAgent
from wrappers import LimitedFuelAndImpactWrapper
from genetic_train import GeneticNetwork


def load_agent(model_path, state_dim, action_dim):
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


def prepare_state(state, expected_dim):
    """Garante que o vetor de estado tenha a dimensão exata esperada pelo modelo (9 posições)."""
    if len(state) < expected_dim:
        padding = np.ones(expected_dim - len(state), dtype=np.float32)
        state = np.append(state, padding)
    return state.astype(np.float32)


def select_dqn_action(agent, state):
    try:
        return agent.select_action(state)
    except TypeError:
        try:
            return agent.select_action(state, 0.0)
        except TypeError:
            return agent.select_action(state, epsilon=0.0)


def evaluate(mode="dqn", model_path="models/dqn_1000.pt", env_type="custom", max_fuel=200.0, num_episodes=5, render=True):
    # Ativa o modo visual se render=True
    render_mode = "human" if render else None
    raw_env = gym.make("LunarLander-v3", render_mode=render_mode)
    
    if env_type == "custom":
        env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=max_fuel)
        print(f"\n⚙️ Running on CUSTOM Environment (LimitedFuelAndImpactWrapper, max_fuel={max_fuel})")
    else:
        env = raw_env
        print(f"\n⚙️ Running on STANDARD Environment (Raw LunarLander-v3)")

    expected_state_dim = 9
    action_dim = env.action_space.n

    if mode == "dqn":
        agent = load_agent(model_path, expected_state_dim, action_dim)
    else:
        ga_data = torch.load(model_path)
        if isinstance(ga_data, torch.nn.Module):
            agent = ga_data
            agent.eval()
        else:
            agent = GeneticNetwork(expected_state_dim, action_dim)
            agent.load_state_dict(ga_data)
            agent.eval()

    print(f"🚀 Evaluating {num_episodes} episodes for [{mode.upper()}]: '{model_path}'...\n")

    for ep in range(1, num_episodes + 1):
        state, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state = prepare_state(state, expected_state_dim)

            with torch.no_grad():
                if mode == "dqn":
                    action = select_dqn_action(agent, state)
                else:
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    logits = agent(state_t)
                    action = torch.argmax(logits, dim=1).item()

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        status = info.get("impact_status", "N/A (Standard Env)")
        fuel = info.get("fuel_left", "N/A")
        fuel_str = f"{fuel:5.1f}" if isinstance(fuel, (int, float)) else str(fuel)
        print(f"Episode {ep}: Reward = {total_reward:7.2f} | Fuel Left = {fuel_str} | Status: {status}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN or Genetic Agent on Custom or Standard Environment.")
    parser.add_argument("--mode", type=str, default="dqn", choices=["dqn", "genetic"], help="Agent type")
    parser.add_argument("--model", type=str, default="models/dqn_1000.pt", help="Path to model checkpoint")
    parser.add_argument("--env_type", type=str, default="custom", choices=["custom", "standard"], help="Environment type (custom/standard)")
    parser.add_argument("--max_fuel", type=float, default=200.0, help="Fuel limit for custom env")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--no_render", action="store_true", help="Disable graphical rendering window")

    args = parser.parse_args()
    evaluate(
        mode=args.mode, 
        model_path=args.model, 
        env_type=args.env_type, 
        max_fuel=args.max_fuel, 
        num_episodes=args.episodes,
        render=not args.no_render
    )
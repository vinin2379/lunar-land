import argparse
import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from wrappers import LimitedFuelAndImpactWrapper


class RocketBrain(nn.Module):
    """Rede Neural direta para controle por Algoritmo Genético."""
    def __init__(self, state_dim=8, action_dim=4):
        super(RocketBrain, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        return torch.argmax(q_values).item()

    def mutate(self, rate=0.10, scale=0.15):
        """Aplica mutação suave na rede."""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < rate:
                    noise = torch.randn_like(param) * scale
                    param.add_(noise)


def train_genetic(population_size=30, generations=300, max_steps=500):
    """Treino direto do Algoritmo Genético com 1 episódio por indivíduo."""
    raw_env = gym.make("LunarLander-v3")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=100.0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    population = [RocketBrain(state_dim, action_dim) for _ in range(population_size)]
    history_best_fitness = []

    print(f"\n--- Treino Genético Iniciado ({generations} Gerações | 1 Avaliação por Indivíduo) ---")

    for gen in range(1, generations + 1):
        fitness_scores = []

        for agent in population:
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < max_steps:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
                steps += 1

            fitness_scores.append(total_reward)

        # Ordena a população pelo desempenho no episódio
        sorted_indices = np.argsort(fitness_scores)[::-1]
        best_fitness = fitness_scores[sorted_indices[0]]
        avg_fitness = np.mean(fitness_scores)
        history_best_fitness.append(best_fitness)

        if gen % 10 == 0 or gen == 1:
            print(
                f"Geração: {gen:3d}/{generations} | "
                f"Melhor Fitness: {best_fitness:6.1f} | "
                f"Média População: {avg_fitness:6.1f}"
            )

        # Elitismo: preserva os melhores sem alterar
        elite_size = max(2, int(population_size * 0.15))
        new_population = [
            copy.deepcopy(population[idx]) for idx in sorted_indices[:elite_size]
        ]

        # Gera descendentes com mutação
        while len(new_population) < population_size:
            parent = random.choice([population[idx] for idx in sorted_indices[:elite_size]])
            child = copy.deepcopy(parent)
            child.mutate(rate=0.10, scale=0.15)
            new_population.append(child)

        population = new_population

    # Salva o melhor modelo da última geração
    os.makedirs("models", exist_ok=True)
    best_agent = population[sorted_indices[0]]
    torch.save(best_agent.state_dict(), "models/genetic_best.pt")
    print("\nModelo genético salvo em 'models/genetic_best.pt'!")

    env.close()
    return history_best_fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do Algoritmo Genético para o LunarLander")
    parser.add_argument(
        "--generations",
        type=int,
        default=1000,
        help="Número de gerações (padrão: 300)",
    )
    args = parser.parse_args()

    train_genetic(generations=args.generations)
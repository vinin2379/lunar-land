import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from wrappers import LimitedFuelAndImpactWrapper


# Arquitetura da Rede Neural para o Algoritmo Genético
class GeneticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

    def get_flat_weights(self):
        """Retorna todos os pesos da rede como um único vetor 1D."""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_flat_weights(self, flat_weights):
        """Aplica um vetor 1D de pesos de volta na estrutura da rede PyTorch."""
        idx = 0
        for p in self.parameters():
            shape = p.data.shape
            size = p.data.numel()
            p.data = torch.FloatTensor(flat_weights[idx:idx + size].reshape(shape))
            idx += size


def evaluate_individual(model, env, eval_episodes=3):
    """Avalia um indivíduo em múltiplos episódios e retorna a recompensa média."""
    total_reward = 0.0
    model.eval()

    with torch.no_grad():
        for ep in range(eval_episodes):
            state, info = env.reset()
            done = False

            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                logits = model(state_t)
                action = torch.argmax(logits, dim=1).item()

                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

    return total_reward / eval_episodes


def mutate(flat_weights, mutation_rate=0.1, mutation_scale=0.05):
    """Aplica mutação gaussiana suave nos pesos."""
    mutated = flat_weights.copy()
    mask = np.random.rand(*mutated.shape) < mutation_rate
    noise = np.random.normal(0, mutation_scale, size=mutated.shape)
    mutated[mask] += noise[mask]
    return mutated


def crossover(parent1_weights, parent2_weights):
    """Crossover uniforme entre dois pais."""
    mask = np.random.rand(*parent1_weights.shape) > 0.5
    child_weights = np.where(mask, parent1_weights, parent2_weights)
    return child_weights


def train_genetic(generations=1000, population_size=30, max_fuel=200.0, output_path="models/genetic_best.pt"):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    raw_env = gym.make("LunarLander-v3")
    env = LimitedFuelAndImpactWrapper(raw_env, max_fuel=max_fuel)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Inicializa a população de redes neurais
    population = [GeneticNetwork(state_dim, action_dim) for _ in range(population_size)]
    
    best_overall_reward = -float("inf")
    best_overall_model = None

    print(f"🧬 Iniciando treino por Algoritmo Genético ({generations} gerações, pop={population_size})...")

    for gen in range(1, generations + 1):
        # 1. Avaliação de Fitness de toda a população
        fitness_scores = []
        for ind in population:
            score = evaluate_individual(ind, env, eval_episodes=3)
            fitness_scores.append(score)

        # 2. Ordena a população do melhor para o pior
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        current_best_score = fitness_scores[0]

        # 3. ELITISMO ABSOLUTO: Mantém o melhor indivíduo histórico e o melhor da geração atual
        if current_best_score > best_overall_reward:
            best_overall_reward = current_best_score
            best_overall_model = copy.deepcopy(population[0])
            torch.save(best_overall_model.state_dict(), output_path)

        if gen % 10 == 0 or gen == 1:
            print(f"   Geração {gen:4d}/{generations} | Melhor Fitness Atual: {current_best_score:7.2f} | Recorde Histórico: {best_overall_reward:7.2f}")

        # 4. Construção da Nova Geração com Elitismo Preservado
        # Os TOP 2 indivíduos passam INTACTOS para a próxima geração sem mutação
        new_population = [
            copy.deepcopy(population[0]),
            copy.deepcopy(population[1])
        ]

        # Torneio para seleção dos pais para o restante da população
        def select_parent():
            tournament_idx = np.random.choice(len(population) // 2, size=3, replace=False)
            return population[min(tournament_idx)]

        while len(new_population) < population_size:
            parent1 = select_parent()
            parent2 = select_parent()

            p1_weights = parent1.get_flat_weights()
            p2_weights = parent2.get_flat_weights()

            # Gera filho por Crossover
            child_weights = crossover(p1_weights, p2_weights)

            # Aplica Mutação Suave
            child_weights = mutate(child_weights, mutation_rate=0.1, mutation_scale=0.05)

            # Cria novo indivíduo na população
            child = GeneticNetwork(state_dim, action_dim)
            child.set_flat_weights(child_weights)
            new_population.append(child)

        population = new_population

    env.close()
    print(f"\n✅ Treino Genético concluído! Melhor modelo salvo em '{output_path}' com Fitness Recorde de {best_overall_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do LunarLander via Algoritmo Genético.")
    parser.add_argument("--generations", type=int, default=1000, help="Número total de gerações")
    parser.add_argument("--pop_size", type=int, default=30, help="Tamanho da população")
    parser.add_argument("--max_fuel", type=float, default=200.0, help="Limite de combustível")
    parser.add_argument("--output", type=str, default="models/genetic_best.pt", help="Caminho de saída")

    args = parser.parse_args()
    train_genetic(generations=args.generations, population_size=args.pop_size, max_fuel=args.max_fuel, output_path=args.output)
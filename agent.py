import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, simple_arch=False):
        super(QNetwork, self).__init__()
        if simple_arch:
            # Modelo Simples: Apenas 1 camada oculta com 64 neurônios
            self.fc = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
        else:
            # Modelo Completo: 2 camadas ocultas com 128 neurônios cada
            self.fc = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, use_exploration=True, simple_arch=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_exploration = use_exploration

        # Hiperparâmetros
        self.gamma = 0.99
        self.lr = 5e-4
        
        # Se use_exploration=False, o epsilon é fixado em 0 (sem exploração aleatória)
        self.epsilon = 1.0 if use_exploration else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.batch_size = 64
        self.memory = deque(maxlen=100000)

        self.policy_net = QNetwork(state_dim, action_dim, simple_arch=simple_arch)
        self.target_net = QNetwork(state_dim, action_dim, simple_arch=simple_arch)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def select_action(self, state):
        if self.use_exploration and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_exploration and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
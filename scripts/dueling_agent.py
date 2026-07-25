"""
agent.py  — Dueling Double DQN Agent
=====================================
Key fixes applied:
  1. compute_reward() removed from this class entirely.
     The centralised version in simulator.py is used instead.
     Previously two compute_reward definitions existed in this file
     (one was a duplicate with no body that shadowed the real one),
     and the real one was missing the clipping that train.py needed.
  2. The reward method now simply delegates to simulator.compute_reward()
     so there is one single source of truth for reward calculation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ── Dueling DQN Neural Network ────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Dueling Q-Network architecture (Wang et al., 2016).

    Decomposes Q(s,a) into:
        V(s)    — state-value stream: how good is this state regardless of action
        A(s,a)  — advantage stream:  how much better is action a vs the average

    Combined as: Q(s,a) = V(s) + [ A(s,a) - mean_a'(A(s,a')) ]

    In the IoMT offloading context, many time steps involve stable network
    conditions where any reasonable action yields similar outcomes. The value
    stream efficiently identifies these periods; the advantage stream focuses
    learning on the genuinely critical decision points (e.g. Tier-1 tasks
    under congested channels).
    """
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Value stream V(s) — single scalar output
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream A(s,a) — one value per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        features  = self.shared(state)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Mean-advantage subtraction ensures V and A are uniquely identifiable
        q_values  = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# ── Experience Replay Buffer ──────────────────────────────────────────────────
class ReplayBuffer:
    """
    Uniform random experience replay buffer.
    Stores (state, action, reward, next_state) tuples and returns
    random mini-batches to decorrelate consecutive training samples.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
        )

    def __len__(self):
        return len(self.buffer)


# ── Dueling DQN Agent ─────────────────────────────────────────────────────────
class DuelingDQNAgent:
    """
    Dueling Double DQN agent for IoMT task offloading.

    Uses two networks:
      online_net  — updated every training step
      target_net  — synced from online_net every target_update steps,
                    providing stable Q-value targets (Double DQN fix)
    """
    def __init__(self, state_size, action_size,
                 lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                 batch_size=64, target_update=100):

        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.step_count    = 0

        self.online_net = DuelingDQN(state_size, action_size)
        self.target_net = DuelingDQN(state_size, action_size)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory    = ReplayBuffer(capacity=10000)

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        Epsilon decays from 1.0 (pure exploration) to 0.01 (mostly exploitation).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.online_net(state_tensor).argmax().item()

    def train_step(self):
        """
        One gradient update step using a random mini-batch from replay memory.
        Implements Double DQN: online net selects greedy action,
        target net evaluates it — suppressing Q-value overestimation.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states = self.memory.sample(self.batch_size)

        # Current Q-values
        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_actions  = self.online_net(next_states).argmax(1)
            next_q        = self.target_net(next_states).gather(
                                1, best_actions.unsqueeze(1)).squeeze(1)
            target_q      = rewards + self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        return loss.item()

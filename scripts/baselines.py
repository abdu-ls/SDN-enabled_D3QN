import numpy as np


class GreedyAgent:
    """
    Always picks the MEC server with the currently lowest load.
    MEC load values are already inside the state vector at positions
    7 to 7+n_mec, so no need to pass server objects separately.
    No consideration for energy or task priority.
    """
    def __init__(self, n_mec=3):
        self.n_mec = n_mec

    def select_action(self, state):
        # Extract MEC load values from the state vector (positions 7 to 7+n_mec)
        mec_loads = state[7 : 7 + self.n_mec]
        # Add tiny noise to break ties and prevent always selecting server 0
        noise = np.random.uniform(0, 0.001, size=len(mec_loads))
        # Pick the least loaded server; +1 because action 0 = local execution
        return int(np.argmin(mec_loads + noise)) + 1


class RoundRobinAgent:
    """
    Cycles through all available destinations in fixed order:
    local -> MEC-1 -> MEC-2 -> MEC-3 -> cloud -> local -> ...
    No awareness of network conditions, task urgency, or battery state.
    """
    def __init__(self, n_actions):
        self.counter  = 0
        self.n_actions = n_actions

    def select_action(self, state=None):
        action = self.counter % self.n_actions
        self.counter += 1
        return action


class StandardDQNAgent:
    """
    Standard single-stream DQN, identical MDP formulation to Dueling DQN
    but without the value/advantage decomposition.
    Used to isolate the specific benefit of the dueling architecture.
    """
    def __init__(self, state_size, action_size,
                 lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                 batch_size=64, target_update=100):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from collections import deque
        import random

        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.step_count    = 0
        self._torch        = torch
        self._random       = random

        # Single-stream Q-network (no dueling decomposition)
        self.online_net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, action_size),
        )
        self.target_net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, action_size),
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory    = deque(maxlen=10000)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        with self._torch.no_grad():
            t = self._torch.FloatTensor(state).unsqueeze(0)
            return self.online_net(t).argmax().item()

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        import torch
        import torch.nn as nn

        batch  = self._random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

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

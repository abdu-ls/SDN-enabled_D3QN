"""agents_v2.py -- one configurable DQN agent (PyTorch).

dueling=True,  double=True   -> Dueling Double DQN   (the proposed agent)
dueling=False, double=False  -> Standard DQN, max-target update, paper Eq. (7)
dueling=False, double=True   -> Double DQN (this is what v1 'StandardDQNAgent' really was)
dueling=True,  double=False  -> Dueling DQN

The non-dueling network uses the same 128-128 trunk and a single 64-unit head, so the
only architectural difference is the value/advantage split.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, state_size, action_size, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.shared = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU())
        if dueling:
            self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.adv   = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size))
        else:
            self.head  = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size))

    def forward(self, x):
        h = self.shared(x)
        if not self.dueling:
            return self.head(h)
        a = self.adv(h)
        return self.value(h) + (a - a.mean(dim=1, keepdim=True))


class DQNAgent:
    def __init__(self, state_size, action_size, dueling=True, double=True,
                 lr=1e-3, gamma=0.95, eps_start=1.0, eps_end=0.01, eps_decay_steps=20000,
                 batch_size=64, target_update=100, buffer_size=10000, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.A, self.gamma, self.double = action_size, gamma, double
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.batch, self.target_update = batch_size, target_update
        self.online = QNet(state_size, action_size, dueling)
        self.target = QNet(state_size, action_size, dueling)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.cap, self.n, self.ptr = buffer_size, 0, 0
        self.S  = np.zeros((buffer_size, state_size), np.float32)
        self.S2 = np.zeros((buffer_size, state_size), np.float32)
        self.Ac = np.zeros(buffer_size, np.int64)
        self.R  = np.zeros(buffer_size, np.float32)
        self.acts = 0          # number of exploratory-mode action selections
        self.learns = 0

    @property
    def epsilon(self):
        frac = min(self.acts / self.eps_decay_steps, 1.0)
        return self.eps_start * (self.eps_end / self.eps_start) ** frac

    def act(self, s, greedy=False):
        if not greedy:
            eps = self.epsilon; self.acts += 1
            if self.rng.random() < eps:
                return int(self.rng.integers(self.A))
        with torch.no_grad():
            return int(self.online(torch.from_numpy(s).float().unsqueeze(0)).argmax())

    def push(self, s, a, r, s2):
        i = self.ptr
        self.S[i], self.Ac[i], self.R[i], self.S2[i] = s, a, r, s2
        self.ptr = (self.ptr + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def learn(self):
        if self.n < self.batch:
            return None
        idx = self.rng.integers(self.n, size=self.batch)
        s, s2 = torch.from_numpy(self.S[idx]), torch.from_numpy(self.S2[idx])
        a, r = torch.from_numpy(self.Ac[idx]), torch.from_numpy(self.R[idx])
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                a_star = self.online(s2).argmax(1, keepdim=True)
                nq = self.target(s2).gather(1, a_star).squeeze(1)
            else:
                nq = self.target(s2).max(1).values
            y = r + self.gamma * nq
        loss = nn.functional.mse_loss(q, y)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()
        self.learns += 1
        if self.learns % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss)

"""policies.py -- non-learning baselines (numpy only). All act on the 13-D state vector."""
import numpy as np
from iomt_env import (KAPPA, CLOUD_CPU_HZ, BANDWIDTH_HZ, C_MAX, D_MAX, Q_REF, DEADLINE)

F_MEC_NOM, PTX_NOM, WAN_NOM = 7.5e9, 0.3, 0.05


class AlwaysLocal:
    def act(self, s): return 0

class RandomPolicy:
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def act(self, s): return int(self.rng.integers(5))

class RoundRobin:
    def __init__(self): self.i = 0
    def act(self, s):
        a = self.i % 5; self.i += 1; return a

class Greedy:
    """Least-loaded MEC server (never local, never cloud) -- same as v1."""
    def act(self, s): return int(np.argmin(s[7:10] + np.random.uniform(0, 1e-3, 3))) + 1


class FeasibleMinEnergy:
    """Strong heuristic using only what the agent observes.
    Estimate completion time and energy of every destination; among destinations whose
    estimated time <= margin * deadline pick the lowest energy; if none is feasible pick
    the fastest.
      use_priority=True : deadline from the tier P_i (priority-aware)
      use_priority=False: one common deadline for every task (priority-blind)
      use_global=False  : ignores MEC loads and link SNRs (device-local view)"""
    def __init__(self, use_priority=True, use_global=True, common_deadline=0.100, margin=0.8):
        self.up, self.ug, self.cd, self.margin = use_priority, use_global, common_deadline, margin

    def act(self, s):
        f, Qs = s[1] * 1e9, s[2] * Q_REF
        D, C = s[3] * D_MAX, s[4] * C_MAX
        P = int(round(s[5] * 3))
        dl = DEADLINE[P] if self.up else self.cd
        if self.ug:
            load, snr = np.minimum(s[7:10], 0.95), s[10:13] * 30
        else:
            load, snr = np.full(3, 0.3), np.full(3, 17.5)
        rate = BANDWIDTH_HZ * np.log2(1 + 10 ** (snr / 10))
        t = [Qs + C / f]; e = [KAPPA * f * f * C]
        for j in range(3):
            tx = D / rate[j]
            t.append(tx + C / (F_MEC_NOM * (1 - load[j]))); e.append(PTX_NOM * tx)
        tx = D / rate.max()
        t.append(tx + WAN_NOM + C / CLOUD_CPU_HZ); e.append(PTX_NOM * tx)
        t, e = np.array(t), np.array(e)
        ok = t <= self.margin * dl
        if ok.any():
            return int(np.argmin(np.where(ok, e + 1e-9 * t, np.inf)))
        return int(np.argmin(t))

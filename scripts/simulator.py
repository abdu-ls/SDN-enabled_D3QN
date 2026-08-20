"""
simulator.py  — IoMT Discrete-Event Simulation Environment
"""

import simpy
import numpy as np

# Physical constants 
BANDWIDTH_HZ  = 10e6       # Uplink channel bandwidth — 10 MHz (5G NR standard)
KAPPA         = 1e-28      # Effective switched capacitance of device CPU
CLOUD_CPU_HZ  = 50e9       # Cloud server CPU frequency — 50 GHz

# Physiological thresholds for priority classification 
THRESHOLDS = {
    "heart_rate":     {"lo": 40,  "hi": 150, "wlo": 45,  "whi": 120},
    "spo2":           {"lo": 90,  "hi": 101, "wlo": 92,  "whi": 95},
    "blood_glucose":  {"lo": 54,  "hi": 250, "wlo": 70,  "whi": 180},
    "blood_pressure": {"lo": 0,   "hi": 180, "wlo": 130, "whi": 160},
}
VITAL_SIGNS = list(THRESHOLDS.keys())


# Priority classification 
def classify_priority(vital, value):
    """
    Returns 1 (critical), 2 (urgent), or 3 (routine).
    MDP priority state P_i — Equation (2) in the paper.
    """
    t = THRESHOLDS[vital]
    if value < t["lo"] or value > t["hi"]:
        return 1
    elif value < t["wlo"] or value > t["whi"]:
        return 2
    else:
        return 3


def generate_sensor_reading(vital, critical_prob=0.05, urgent_prob=0.15):
    """
    Simulates a physiological sensor reading.
    5% chance of critical reading, 15% chance of urgent, 80% routine.
    """
    t = THRESHOLDS[vital]
    r = np.random.random()
    if r < critical_prob:
        if np.random.random() < 0.5:
            return np.random.uniform(t["lo"] * 0.5, t["lo"] - 1)
        else:
            return np.random.uniform(t["hi"] + 1, t["hi"] * 1.3)
    elif r < critical_prob + urgent_prob:
        return np.random.uniform(t["wlo"], t["lo"])
    else:
        return np.random.uniform(t["lo"], t["hi"])


# Centralised reward function 
def compute_reward(outcome, task, alpha=0.6, beta=0.4, lam=50.0):
    """
    CENTRALISED reward function — import and call this from ALL scripts.
    """
    priority_weight   = 4 - task["priority"]          # 3, 2, or 1
    time_ratio        = min(outcome["time"] / task["deadline"], 3.0)
    energy_ratio      = min(outcome["energy"] / 0.5,  1.0)
    latency_penalty   = alpha * priority_weight * time_ratio
    energy_penalty    = beta  * energy_ratio
    violation_penalty = lam   * outcome["violated"]
    return -(latency_penalty + energy_penalty + violation_penalty)


# SNR → uplink rate conversion (Shannon capacity)
def snr_to_rate(snr_linear):
    """
    Shannon capacity: R = B * log2(1 + SNR)
    Returns achievable uplink rate in bits/second.
    SNR must be in linear scale (not dB).
    """
    return BANDWIDTH_HZ * np.log2(1.0 + snr_linear)


def db_to_linear(snr_db):
    """Convert SNR from dB to linear scale."""
    return 10.0 ** (snr_db / 10.0)


# IoMT Device 
class IoMTDevice:
   
    def __init__(self, device_id, env):
        self.id        = device_id
        self.env       = env
        self.battery   = np.random.uniform(0.6, 1.0)          # 60–100% initial
        self.cpu_freq  = np.random.uniform(0.5e9, 1.0e9)      # 0.5–1.0 GHz
        self.tx_power  = np.random.uniform(0.1, 0.5)          # 0.1–0.5 W
        self.queue_depth = 0
        # current_uplink_rate is set by SDNController.get_global_state()
        # each time step based on the live SNR — no longer a fixed attribute
        self.current_uplink_rate = 15e6   # default fallback (15 Mbps)

    def generate_task(self):
      
        vital    = np.random.choice(VITAL_SIGNS)
        value    = generate_sensor_reading(vital)
        priority = classify_priority(vital, value)

        data_size = np.random.uniform(0.1e6 * 8, 5e6 * 8)   # 0.1–5 MB in bits

        if priority == 1:
            cpu_cycles = np.random.uniform(1e6, 5e6)    # small — fast alert check
            deadline   = 0.050                           # 50 ms
        elif priority == 2:
            cpu_cycles = np.random.uniform(5e6, 3e7)    # medium
            deadline   = 0.100                           # 100 ms
        else:
            cpu_cycles = np.random.uniform(3e7, 1e8)    # large — routine analysis
            deadline   = 5.0                             # 5 seconds

        return {
            "device_id":  self.id,
            "priority":   priority,
            "data_size":  data_size,
            "cpu_cycles": cpu_cycles,
            "deadline":   deadline,
            "vital":      vital,
            "value":      value,
        }

    def get_state_features(self):
        """Device-level features for the MDP state vector."""
        return {
            "battery":     self.battery,
            "cpu_freq":    self.cpu_freq / 1e9,
            "queue_depth": self.queue_depth,
            "tx_power":    self.tx_power,
        }


# MEC Server 
class MECServer:
   
    def __init__(self, server_id):
        self.id       = server_id
        self.cpu_freq = np.random.uniform(5e9, 10e9)   # 5–10 GHz
        self.load     = np.random.uniform(0.1, 0.5)    # initial load

    def available_freq(self):
        return self.cpu_freq * (1.0 - self.load)

    def get_state_features(self):
        return {"load": self.load, "cpu_freq": self.cpu_freq / 1e9}


# SDN Controller 
class SDNController:
    
    def __init__(self, devices, mec_servers):
        self.devices     = devices
        self.mec_servers = mec_servers
        self.wan_delay   = np.random.uniform(0.020, 0.080)   # 20–80 ms WAN

        # Store current SNR and derived rates — set in get_global_state()
        # and read in compute_outcome() for consistency within a time step
        self._current_snr_db    = [20.0] * len(mec_servers)  # dB
        self._current_snr_linear= [db_to_linear(20.0)] * len(mec_servers)
        self._current_rates     = [snr_to_rate(db_to_linear(20.0))] * len(mec_servers)

    def get_global_state(self, task):

        dev = self.devices[task["device_id"]]
        dev_feats = dev.get_state_features()

        # Draw SNR for each MEC link — stored for use in compute_outcome()
        snr_db_values = [np.random.uniform(5, 30) for _ in self.mec_servers]
        self._current_snr_db     = snr_db_values
        self._current_snr_linear = [db_to_linear(s) for s in snr_db_values]
        self._current_rates      = [snr_to_rate(sl) for sl in self._current_snr_linear]

        # Update the device's current uplink rate from the best available SNR
        # (the agentwill choose which MEC to use; the device rate reflects
        #  the average channel quality it currently experiences)
        dev.current_uplink_rate = float(np.mean(self._current_rates))

        mec_loads = [s.load for s in self.mec_servers]

        state = np.array(
            [
                dev_feats["battery"],
                dev_feats["cpu_freq"],
                min(dev_feats["queue_depth"] / 10.0, 1.0),   # normalise, cap at 1
                task["data_size"] / (5e6 * 8),
                task["cpu_cycles"] / 1e9,
                float(task["priority"]) / 3.0,
                task["deadline"] / 5.0,                       # normalise by 5s
            ]
            + mec_loads
            + [s / 30.0 for s in snr_db_values],             # normalise SNR to [0,1]
            dtype=np.float32,
        )
        return state

    def compute_outcome(self, task, action):
      
        dev = self.devices[task["device_id"]]
        D   = task["data_size"]     # bits
        C   = task["cpu_cycles"]    # CPU cycles

        if action == 0:
            # Local execution
            energy = KAPPA * (dev.cpu_freq ** 2) * C
            time   = C / dev.cpu_freq

        elif 1 <= action <= len(self.mec_servers):
            # MEC offloading — use the SNR-derived rate for THIS server link
            server   = self.mec_servers[action - 1]
            rate     = self._current_rates[action - 1]   # Shannon capacity for this link
            energy   = dev.tx_power * (D / rate)
            tx_time  = D / rate
            proc_time= C / server.available_freq()
            time     = tx_time + proc_time

        else:
            # Cloud offloading — WAN rate isslower and variable
            wan_rate = np.random.uniform(5e6, 15e6)
            energy   = dev.tx_power * (D / dev.current_uplink_rate)
            time     = (D / wan_rate) + self.wan_delay + (C / CLOUD_CPU_HZ)

        violated = int(time > task["deadline"])

        # Update device battery
        dev.battery = max(0.0, dev.battery - energy * 1e-5)

        # fluctuate MEC loads to simulate dynamic network conditions
        for server in self.mec_servers:
            server.load = np.clip(
                server.load + np.random.uniform(-0.03, 0.03), 0.05, 0.65
            )

        return {
            "energy":            energy,
            "time":              time,
            "violated":          violated,
            "battery_remaining": dev.battery,
        }


# Environment factory 
def build_environment(n_devices=20, n_mec=3):
    """Creates and returns the full IoMT simulation environment."""
    env         = simpy.Environment()
    devices     = [IoMTDevice(i, env) for i in range(n_devices)]
    mec_servers = [MECServer(j) for j in range(n_mec)]
    controller  = SDNController(devices, mec_servers)
    return env, devices, mec_servers, controller

import numpy as np
import pandas as pd
import torch
import time
from scipy.stats import wilcoxon
from simulator import build_environment, compute_reward
from dueling_agent import DuelingDQNAgent
from baselines import GreedyAgent, RoundRobinAgent, StandardDQNAgent

# ── Configuration ─────────────────────────────────────────────────────────────
N_MEC         = 3
N_EPISODES    = 500
TASKS_PER_EP  = 100
N_TEST_TASKS  = 1000
N_SEEDS       = 10                  
DEVICE_COUNTS = [20, 40, 60]         
STATE_SIZE  = 7 + N_MEC + N_MEC     # = 13
ACTION_SIZE = 1 + N_MEC + 1         # = 5

# ── Generic training loop ──────────────────────────────────────────────────────
def train_agent(AgentClass, n_devices, seed, algo_name, n_value):
    """
    Trains any agent for N_EPISODES using the centralised reward function.
    Returns agent, convergence logs, and wall-clock training time.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env, devices, mec_servers, controller = build_environment(n_devices, N_MEC)
    agent = AgentClass(STATE_SIZE, ACTION_SIZE)
    conv_rows = []
    
    # ── Start Timing ──────────────────────────────────────────────────
    start_train_time = time.perf_counter()
    
    for episode in range(N_EPISODES):
        episode_reward_sum = 0.0
        for _ in range(TASKS_PER_EP):
            device_id = np.random.randint(n_devices)
            devices[device_id].queue_depth += 1
            task       = devices[device_id].generate_task()
            state      = controller.get_global_state(task)
            action     = agent.select_action(state)
            outcome    = controller.compute_outcome(task, action)
            
            # ── Single centralised reward for both agents ───────────────────
            reward = compute_reward(outcome, task)
            episode_reward_sum += reward
            
            next_task  = devices[device_id].generate_task()
            next_state = controller.get_global_state(next_task)
            if hasattr(agent, "push"):
                agent.push(state, action, reward, next_state)
            else:
                agent.memory.push(state, action, reward, next_state)
            agent.train_step()
            devices[device_id].queue_depth = max(
                0, devices[device_id].queue_depth - 1)
                
        conv_rows.append({
            "N_devices": n_value,
            "Algorithm": algo_name,
            "Seed":      seed,
            "Episode":   episode + 1,
            "AvgReward": episode_reward_sum / TASKS_PER_EP,
            "Epsilon":   agent.epsilon,
        })
        if (episode + 1) % 100 == 0:
            print(f"      Episode {episode+1}/{N_EPISODES}  "
                  f"ε={agent.epsilon:.3f}  "
                  f"avg_reward={conv_rows[-1]['AvgReward']:.4f}")
                  
    # ── End Timing ────────────────────────────────────────────────────
    train_time_sec = time.perf_counter() - start_train_time
    agent.epsilon = 0.0
    return agent, conv_rows, train_time_sec

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_agent(agent, controller, devices, n_devices, n_tasks=1000):
    """
    Evaluates a trained agent over n_tasks test tasks.
    Returns a DataFrame of per-task results and mean inference time.
    """
    results = []
    inference_times_ms = []
    
    for _ in range(n_tasks):
        device_id = np.random.randint(n_devices)
        task      = devices[device_id].generate_task()
        state     = controller.get_global_state(task)
        
        # ── Measure Inference Latency ─────────────────────────────────
        t_start = time.perf_counter()
        action  = agent.select_action(state)
        t_end   = time.perf_counter()
        inference_times_ms.append((t_end - t_start) * 1000.0)
        # ──────────────────────────────────────────────────────────────
        
        outcome   = controller.compute_outcome(task, action)
        results.append({
            "energy":   outcome["energy"],
            "time_ms":  outcome["time"] * 1000,
            "violated": outcome["violated"],
            "priority": task["priority"],
        })
        
    mean_inf_time_ms = np.mean(inference_times_ms) if inference_times_ms else 0.0
    return pd.DataFrame(results), mean_inf_time_ms

# ── Per-seed metric extraction (needed for cross-seed stats + paired test) ────
def seed_level_metrics(df):
    """Collapse one seed's per-task eval DataFrame into scalar summary metrics."""
    t1 = df[df.priority == 1]
    return {
        "energy_mean":    df["energy"].mean(),
        "time_mean":      df["time_ms"].mean(),
        "viol_overall":   df["violated"].mean() * 100,
        "viol_t1":        t1["violated"].mean() * 100 if len(t1)  > 0 else np.nan,
    }

def mean_ci95(values):
    """Mean and 95% CI half-width across seeds (normal approximation)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return float(values.mean()) if n == 1 else np.nan, np.nan, np.nan
    mean = values.mean()
    std  = values.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(n)
    return mean, std, ci95

# ── Summary builder (now cross-seed, not just cross-task) ─────────────────────
def build_summary_row(name, n, combined_df, per_seed_metrics):
    """
    combined_df       : all tasks from all seeds concatenated
    per_seed_metrics  : list of dicts from seed_level_metrics(), one per seed
    """
    t1 = combined_df[combined_df.priority == 1]
    t2 = combined_df[combined_df.priority == 2]
    t3 = combined_df[combined_df.priority == 3]
    
    energy_mean, energy_seed_std, energy_ci95 = mean_ci95(
        [m["energy_mean"] for m in per_seed_metrics])
    time_mean, time_seed_std, time_ci95 = mean_ci95(
        [m["time_mean"] for m in per_seed_metrics])
    viol_mean, viol_seed_std, viol_ci95 = mean_ci95(
        [m["viol_overall"] for m in per_seed_metrics])
        
    # Calculate Computational Overhead CIs
    train_mean, _, train_ci = mean_ci95(
        [m.get("train_time_sec", np.nan) for m in per_seed_metrics])
    inf_mean, _, inf_ci = mean_ci95(
        [m.get("inference_time_ms", np.nan) for m in per_seed_metrics])

    return {
        "N_devices":         n,
        "Algorithm":         name,
        "N_seeds":           len(per_seed_metrics),
        # Cross-seed estimates
        "Avg_Energy_J":      energy_mean,
        "Seed_Std_Energy_J": energy_seed_std,
        "Seed_CI95_Energy":  energy_ci95,
        "Avg_Time_ms":       time_mean,
        "Seed_Std_Time_ms":  time_seed_std,
        "Seed_CI95_Time":    time_ci95,
        "Viol_Overall_pct":  viol_mean,
        "Seed_Std_Viol_pct": viol_seed_std,
        "Seed_CI95_Viol":    viol_ci95,
        # Computational Overhead Metrics
        "Avg_Train_Time_sec":       train_mean,
        "Seed_CI95_Train_Time":     train_ci,
        "Avg_Inference_Time_ms":    inf_mean,
        "Seed_CI95_Inference_Time": inf_ci,
        # Within-run (task-level) spread
        "Std_Energy_J":      combined_df["energy"].std(),
        "Std_Time_ms":       combined_df["time_ms"].std(),
        "Viol_T1_pct":       t1["violated"].mean() * 100 if len(t1) > 0 else 0.0,
        "Viol_T2_pct":       t2["violated"].mean() * 100 if len(t2) > 0 else 0.0,
        "Viol_T3_pct":       t3["violated"].mean() * 100 if len(t3) > 0 else 0.0,
        "Avg_T1_Time_ms":    t1["time_ms"].mean() if len(t1) > 0 else 0.0,
        "Avg_T2_Time_ms":    t2["time_ms"].mean() if len(t2) > 0 else 0.0,
    }

def paired_wilcoxon_row(n, dueling_metrics, standard_metrics):
    """
    Paired-by-seed Wilcoxon signed-rank test comparing Dueling DQN vs
    Standard DQN, for both mean energy and overall violation rate.
    """
    row = {"N_devices": n}
    for metric_key, label in [("energy_mean", "Energy"),
                              ("viol_overall", "Violation")]:
        a = np.array([m[metric_key] for m in dueling_metrics])
        b = np.array([m[metric_key] for m in standard_metrics])
        row[f"Dueling_mean_{label}"]  = a.mean()
        row[f"Standard_mean_{label}"] = b.mean()
        diff = a - b
        if np.allclose(diff, 0):
            row[f"Wilcoxon_p_{label}"] = 1.0
        else:
            try:
                stat, p = wilcoxon(a, b)
                row[f"Wilcoxon_p_{label}"] = p
            except ValueError:
                row[f"Wilcoxon_p_{label}"] = np.nan
    return row

# ── Main experiment loop ──────────────────────────────────────────────────────
all_summary_rows = []
all_significance_rows = []
all_convergence_rows = []

for N in DEVICE_COUNTS:
    print(f"\n{'='*60}")
    print(f"  N = {N} DEVICES")
    print(f"{'='*60}")
    
    all_results = {
        "Dueling DQN":  [],
        "Standard DQN": [],
        "Greedy":       [],
        "Round-Robin":  [],
    }
    per_seed_metrics = {
        "Dueling DQN":  [],
        "Standard DQN": [],
        "Greedy":       [],
        "Round-Robin":  [],
    }
    
    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed + 1}/{N_SEEDS}")
        
        # ── Train Dueling DQN (seeds 0 .. N_SEEDS-1) ──────────────────────
        print("    Training Dueling DQN...")
        d_agent, d_conv, d_train_time = train_agent(
            DuelingDQNAgent, N, seed=seed,
            algo_name="Dueling DQN", n_value=N)
        all_convergence_rows.extend(d_conv)
        
        # ── Train Standard DQN (seeds 0 .. N_SEEDS-1) ─────────────────────
        # FIXED: Changed from seed+50 to seed to ensure valid paired testing
        print("    Training Standard DQN...")
        s_agent, s_conv, s_train_time = train_agent(
            StandardDQNAgent, N, seed=seed, 
            algo_name="Standard DQN", n_value=N)
        all_convergence_rows.extend(s_conv)
        
        # ── Fresh evaluation environment (seed+100) ────────────────────────
        np.random.seed(seed + 100)
        _, devices_eval, _, controller_eval = build_environment(N, N_MEC)
        
        # Copy weights into fresh agent instances for evaluation
        d_eval = DuelingDQNAgent(STATE_SIZE, ACTION_SIZE)
        d_eval.online_net.load_state_dict(d_agent.online_net.state_dict())
        d_eval.epsilon = 0.0
        
        s_eval = StandardDQNAgent(STATE_SIZE, ACTION_SIZE)
        s_eval.online_net.load_state_dict(s_agent.online_net.state_dict())
        s_eval.epsilon = 0.0
        
        greedy_agent = GreedyAgent(n_mec=N_MEC)
        rr_agent     = RoundRobinAgent(n_actions=ACTION_SIZE)
        
        eval_agents = {
            "Dueling DQN":  d_eval,
            "Standard DQN": s_eval,
            "Greedy":       greedy_agent,
            "Round-Robin":  rr_agent,
        }
        
        print("    Evaluating...")
        for name, agent in eval_agents.items():
            df, inf_time = evaluate_agent(agent, controller_eval,
                                          devices_eval, N, N_TEST_TASKS)
            all_results[name].append(df)
            
            # Get base metrics and inject computational overhead metrics
            metrics = seed_level_metrics(df)
            metrics["inference_time_ms"] = inf_time
            if name == "Dueling DQN":
                metrics["train_time_sec"] = d_train_time
            elif name == "Standard DQN":
                metrics["train_time_sec"] = s_train_time
            else:
                metrics["train_time_sec"] = np.nan
                
            per_seed_metrics[name].append(metrics)
            
            t1 = df[df.priority == 1]
            print(f"      {name:<15}: "
                  f"E={df['energy'].mean():.4f}J  "
                  f"T={df['time_ms'].mean():.1f}ms  "
                  f"viol={df['violated'].mean()*100:.2f}%  "
                  f"T1={t1['violated'].mean()*100:.2f}%  "
                  f"Inf={inf_time:.4f}ms")
                  
    # ── Save per-N CSV files and build summary ─────────────────────────────
    print(f"\n  Saving N={N} results...")
    for name, dfs in all_results.items():
        combined = pd.concat(dfs, ignore_index=True)
        fname = f"results_N{N}_{name.replace(' ', '_')}.csv"
        combined.to_csv(fname, index=False)
        print(f"    Saved {fname}")
        
        all_summary_rows.append(
            build_summary_row(name, N, combined, per_seed_metrics[name]))
            
    # ── Paired significance test: Dueling DQN vs Standard DQN ──────────────
    all_significance_rows.append(
        paired_wilcoxon_row(N,
                            per_seed_metrics["Dueling DQN"],
                            per_seed_metrics["Standard DQN"]))

# ── Save master summary ───────────────────────────────────────────────────────
summary_df = pd.DataFrame(all_summary_rows)
summary_df = summary_df.sort_values(
    ["N_devices", "Algorithm"]).reset_index(drop=True)
summary_df.to_csv("results_summary.csv", index=False)

# ── Save significance results ──────────────────────────────────────────────────
significance_df = pd.DataFrame(all_significance_rows)
significance_df.to_csv("significance.csv", index=False)

# ── Save convergence log ────────────────────────────────────────────────────────
convergence_df = pd.DataFrame(all_convergence_rows)
for N in DEVICE_COUNTS:
    n_conv = convergence_df[convergence_df.N_devices == N]
    n_conv.to_csv(f"convergence_N{N}.csv", index=False)
    print(f"  Saved convergence_N{N}.csv")

print("\n" + "="*60)
print(f"  SCALABILITY SUMMARY (cross-seed means, n_seeds={N_SEEDS})")
print("="*60)
cols = ["N_devices", "Algorithm", "Avg_Energy_J", "Seed_CI95_Energy",
        "Avg_Time_ms", "Viol_Overall_pct", "Seed_CI95_Viol",
        "Avg_Train_Time_sec", "Avg_Inference_Time_ms"]
print(summary_df[cols].to_string(index=False, float_format="%.4f"))

print("\n" + "="*60)
print("  DUELING vs STANDARD DQN — PAIRED WILCOXON SIGNIFICANCE")
print("="*60)
print(significance_df.to_string(index=False, float_format="%.4f"))

print("\nDone. Run plot_all_results.py to generate figures.")
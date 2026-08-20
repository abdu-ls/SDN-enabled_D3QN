"""
plot_all_results.py - run to generate all graphs
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS  = {"Dueling DQN": "#028090", "Standard DQN": "#ff7f0e",
           "Greedy": "#d62728", "Round-Robin": "#2ca02c"}
AGENTS  = ["Dueling DQN", "Standard DQN", "Greedy", "Round-Robin"]
STYLES  = {"Dueling DQN": "-", "Standard DQN": "--",
           "Greedy": "-.", "Round-Robin": ":"}
MARKERS = {"Dueling DQN": "o", "Standard DQN": "s",
           "Greedy": "^", "Round-Robin": "D"}
N_TEST_TASKS = 1000  

# Maps algorithm display name -> filename fragment actually used on disk
# (note "Standard DQN" -> "Std_DQN", not "Standard_DQN")
FILE_TAG = {
    "Dueling DQN":  "Dueling_DQN",
    "Standard DQN": "Std_DQN",
    "Greedy":       "Greedy",
    "Round-Robin":  "Round-Robin",
}

DEVICE_COUNTS = [20, 40, 60]


def save(fig, name):
    fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return (values.mean() if n else np.nan), 0.0
    mean = values.mean()
    ci95 = 1.96 * values.std(ddof=1) / np.sqrt(n)
    return mean, ci95


# LOAD DATA
print("Loading summary + significance...")
summary_df = pd.read_csv("summary_results.csv")
sig_df     = pd.read_csv("significance.csv")

print("Loading per-task eval results for all N...")
results = {}          # results[N][agent] = DataFrame, with a Seed column added
for N in DEVICE_COUNTS:
    results[N] = {}
    for agent in AGENTS:
        fname = f"results_N{N}_{FILE_TAG[agent]}.csv"
        df = pd.read_csv(fname)
        df["Seed"] = df.index // N_TEST_TASKS
        results[N][agent] = df

print("Loading convergence logs...")
conv_df = pd.concat(
    [pd.read_csv(f"convergence_N{N}.csv") for N in DEVICE_COUNTS],
    ignore_index=True)

print()

# FIGURE 1 — CONVERGENCE, one subplot per N, mean +/- 95% CI ACROSS SEEDS
print("Generating Figure 1: Convergence (per N, cross-seed CI)...")


def smooth(series, w=15):
    return series.rolling(window=w, min_periods=1, center=True).mean()


fig, axes = plt.subplots(1, len(DEVICE_COUNTS), figsize=(15, 4.3), sharey=True)

for ax, N in zip(axes, DEVICE_COUNTS):
    sub_n = conv_df[conv_df.N_devices == N]
    for algo in ["Dueling DQN", "Standard DQN"]:
        sub = sub_n[sub_n.Algorithm == algo]
        # pivot to (episode x seed) then take mean/CI across seeds per episode
        pivot = sub.pivot(index="Episode", columns="Seed", values="AvgReward")
        mean_per_ep = pivot.mean(axis=1)
        ci_per_ep   = 1.96 * pivot.std(axis=1) / np.sqrt(pivot.shape[1])
        mean_smooth = smooth(mean_per_ep)
        ci_smooth   = smooth(ci_per_ep)

        ax.plot(mean_smooth.index, mean_smooth.values,
                color=COLORS[algo], linestyle=STYLES[algo],
                linewidth=2.0, label=algo)
        ax.fill_between(mean_smooth.index,
                        mean_smooth - ci_smooth,
                        mean_smooth + ci_smooth,
                        color=COLORS[algo], alpha=0.20)

    ax.set_title(f"N = {N} devices", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_xlim(1, sub_n["Episode"].max())

axes[0].set_ylabel("Mean Episode Reward\n(smoothed, shaded = 95% CI across seeds)",
                    fontsize=10.5)
axes[0].legend(fontsize=10, loc="lower right")
fig.suptitle("Figure 1: Training Convergence — Dueling vs Standard DQN "
             f"(n={pivot.shape[1]} seeds per curve)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig1_convergence")

# FIGURE 2 — ENERGY vs N  (cross-seed 95% CI error bars)
print("Generating Figure 2: Energy scaling vs N...")
fig, ax = plt.subplots(figsize=(8, 4.5))
for agent in AGENTS:
    sub = summary_df[summary_df.Algorithm == agent].sort_values("N_devices")
    ax.errorbar(sub["N_devices"], sub["Avg_Energy_J"],
                yerr=sub["Seed_CI95_Energy"],
                label=agent, color=COLORS[agent], marker=MARKERS[agent],
                linestyle=STYLES[agent], linewidth=2.0, markersize=7, capsize=4)
ax.set_xticks(DEVICE_COUNTS)
ax.set_xticklabels([f"N = {n}" for n in DEVICE_COUNTS])
ax.set_xlabel("Number of Devices")
ax.set_ylabel("Average Energy per Task (J)")
ax.set_title("Figure 2: Energy Consumption vs Device Count\n"
             "(error bars: 95% CI across seeds)", fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "fig2_energy_scaling")

# FIGURE 3 — OVERALL VIOLATION RATE vs N  (cross-seed 95% CI error bars)
print("Generating Figure 3: Violation rate scaling vs N...")
fig, ax = plt.subplots(figsize=(8, 4.5))
for agent in AGENTS:
    sub = summary_df[summary_df.Algorithm == agent].sort_values("N_devices")
    ax.errorbar(sub["N_devices"], sub["Viol_Overall_pct"],
                yerr=sub["Seed_CI95_Viol"],
                label=agent, color=COLORS[agent], marker=MARKERS[agent],
                linestyle=STYLES[agent], linewidth=2.0, markersize=7, capsize=4)
ax.set_xticks(DEVICE_COUNTS)
ax.set_xticklabels([f"N = {n}" for n in DEVICE_COUNTS])
ax.set_xlabel("Number of Devices")
ax.set_ylabel("Overall Deadline Violation Rate (%)")
ax.set_title("Figure 3: Deadline Violations vs Device Count\n"
             "(error bars: 95% CI across seeds)", fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "fig3_violation_scaling")

# FIGURE 4 — LATENCY vs N  (cross-seed 95% CI error bars)
print("Generating Figure 4: Latency scaling vs N...")
fig, ax = plt.subplots(figsize=(8, 4.5))
for agent in AGENTS:
    sub = summary_df[summary_df.Algorithm == agent].sort_values("N_devices")
    ax.errorbar(sub["N_devices"], sub["Avg_Time_ms"],
                yerr=sub["Seed_CI95_Time"],
                label=agent, color=COLORS[agent], marker=MARKERS[agent],
                linestyle=STYLES[agent], linewidth=2.0, markersize=7, capsize=4)
ax.set_xticks(DEVICE_COUNTS)
ax.set_xticklabels([f"N = {n}" for n in DEVICE_COUNTS])
ax.set_xlabel("Number of Devices")
ax.set_ylabel("Average Task Completion Time (ms)")
ax.set_title("Figure 4: Latency vs Device Count\n"
             "(error bars: 95% CI across seeds)", fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "fig4_latency_scaling")

# FIGURE 5 — TIER-1 VIOLATION RATE vs N  (recomputed per-seed CI from raw data)
print("Generating Figure 5: Tier-1 violation scaling vs N...")
fig, ax = plt.subplots(figsize=(8, 4.5))
for agent in AGENTS:
    means, cis = [], []
    for N in DEVICE_COUNTS:
        df = results[N][agent]
        t1 = df[df.priority == 1]
        per_seed = t1.groupby("Seed")["violated"].mean() * 100
        mean, ci = mean_ci95(per_seed.values)
        means.append(mean)
        cis.append(ci)
    ax.errorbar(DEVICE_COUNTS, means, yerr=cis,
                label=agent, color=COLORS[agent], marker=MARKERS[agent],
                linestyle=STYLES[agent], linewidth=2.0, markersize=7, capsize=4)
ax.set_xticks(DEVICE_COUNTS)
ax.set_xticklabels([f"N = {n}" for n in DEVICE_COUNTS])
ax.set_xlabel("Number of Devices")
ax.set_ylabel("Tier-1 (Critical) Violation Rate (%)")
ax.set_title("Figure 5: Tier-1 Deadline Violations vs Device Count\n"
             "(error bars: 95% CI across seeds)", fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(-2, 105)
fig.tight_layout()
save(fig, "fig5_tier1_scaling")

# FIGURE 6 — ENERGY BAR CHART, snapshot at N = 20
print("Generating Figure 6: Energy bar chart (N=20 snapshot)...")
N_SNAP = 20
means, cis = [], []
for agent in AGENTS:
    per_seed = results[N_SNAP][agent].groupby("Seed")["energy"].mean()
    m, c = mean_ci95(per_seed.values)
    means.append(m)
    cis.append(c)

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(AGENTS, means, yerr=cis,
              color=[COLORS[a] for a in AGENTS],
              capsize=5, width=0.55, alpha=0.9,
              error_kw={"linewidth": 1.4, "ecolor": "black"})
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.02,
            f"{m:.4f} J", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Average Energy per Task (J)")
ax.set_title(f"Figure 6: Average Energy Consumption per Task (N = {N_SNAP})\n"
             "(error bars: 95% CI across seeds)", fontweight="bold")
fig.tight_layout()
save(fig, "fig6_energy_bar_N20")

# FIGURE 7 — LATENCY CDF, snapshot at N = 60 (most-loaded condition)
print("Generating Figure 7: Latency CDF (N=60 snapshot)...")
N_SNAP = 60
fig, ax = plt.subplots(figsize=(8, 4.5))
for agent in AGENTS:
    data = np.sort(results[N_SNAP][agent]["time_ms"].values)
    cdf  = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, cdf, label=agent, color=COLORS[agent],
            linestyle=STYLES[agent], linewidth=1.8)
ax.axvline(50,  color="dimgray", linestyle="--", linewidth=1.1)
ax.axvline(100, color="black",   linestyle=":",  linewidth=1.1)
ax.text(53,  0.05, "Tier-1\n(50 ms)",  fontsize=9, color="dimgray")
ax.text(103, 0.05, "Tier-2\n(100 ms)", fontsize=9, color="black")
ax.set_xlim(0, 1200)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Task Completion Time (ms)")
ax.set_ylabel("Cumulative Probability")
ax.set_title(f"Figure 7: CDF of Task Completion Times (N = {N_SNAP})", fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
fig.tight_layout()
save(fig, "fig7_latency_cdf_N60")

# FIGURE 8 — VIOLATIONS BY TIER, snapshot at N = 60
print("Generating Figure 8: Tier violations bar chart (N=60 snapshot)...")
N_SNAP = 60
tier_labels = ["Tier 1\n(Critical, 50 ms)", "Tier 2\n(Urgent, 100 ms)", "Tier 3\n(Routine, 5 s)"]
x, width = np.arange(3), 0.2

fig, ax = plt.subplots(figsize=(9, 4.5))
for i, agent in enumerate(AGENTS):
    df = results[N_SNAP][agent]
    viol = [df[df.priority == p]["violated"].mean() * 100 for p in [1, 2, 3]]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, viol, width, label=agent, color=COLORS[agent], alpha=0.9)
    for bar, v in zip(bars, viol):
        if v > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tier_labels, fontsize=10)
ax.set_ylabel("Deadline Violation Rate (%)")
ax.set_title(f"Figure 8: Deadline Violations by Priority Tier (N = {N_SNAP})", fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "fig8_tier_violations_N60")

# FIGURE 9 — SIGNIFICANCE: Dueling vs Standard DQN, paired Wilcoxon p-values
print("Generating Figure 9: Significance summary...")
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.axis("off")

table_rows = []
for _, row in sig_df.iterrows():
    for metric, plabel in [("Energy", "Energy"), ("Violation", "Violation rate")]:
        p = row[f"Wilcoxon_p_{metric}"]
        sig = "significant (p<0.05)" if p < 0.05 else "not significant"
        table_rows.append([
            f"N={int(row['N_devices'])}", plabel,
            f"{row[f'Dueling_mean_{metric}']:.4g}",
            f"{row[f'Standard_mean_{metric}']:.4g}",
            f"{p:.3f}", sig,
        ])

col_labels = ["N", "Metric", "Dueling mean", "Standard mean", "Wilcoxon p", "Result"]
tbl = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#028090")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i, row in enumerate(table_rows, start=1):
    if "not" not in row[-1]:
        tbl[i, 5].set_facecolor("#d4f4dd")

ax.set_title("Figure 9: Dueling DQN vs Standard DQN — Paired Wilcoxon Test "
             f"(n={sig_df.shape[0]} device counts, paired by seed)",
             fontweight="bold", pad=20)
fig.tight_layout()
save(fig, "fig9_significance")

# PRINT DATA TABLES
print("\n" + "="*65)
print("  DATA TABLES")
print("="*65)

print("\n--- Table: Scalability Summary (cross-seed) ---")
cols = ["N_devices", "Algorithm", "Avg_Energy_J", "Seed_CI95_Energy",
        "Avg_Time_ms", "Seed_CI95_Time", "Viol_Overall_pct", "Seed_CI95_Viol"]
print(summary_df[cols].to_string(index=False, float_format="%.4f"))

print("\n--- Table: Significance (Dueling vs Standard DQN) ---")
print(sig_df.to_string(index=False, float_format="%.4f"))

print("\nAll done. Check your folder for fig1-fig9 PDF and PNG files.")

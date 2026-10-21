"""Merge the raw per-run outputs into the two tables used by every later step.
   raw/*/per_seed.csv   -> results/all_per_seed.csv   (one row per method x N x seed)
   raw/*/convergence.csv -> results/all_conv.csv      (one row per method x N x seed x episode)
Duplicates (same arm, N, seed) are dropped keeping the first file in the order below.
Run from the package root:  python analysis/merge_raw.py
"""
import pandas as pd
R = "raw/"
SEEDF = ["ablation_lab_N20", "ablation_lab_N60", "nonlearning_baselines", "main_A", "main_B1", "main_B2", "arch_C"]
CONVF = ["ablation_lab_N20", "ablation_lab_N60", "main_B1", "main_A", "arch_C"]
d = pd.concat([pd.read_csv(f"{R}{k}/per_seed.csv") for k in SEEDF], ignore_index=True)
d = d.drop_duplicates(["arm", "N", "seed"])
d.to_csv("results/all_per_seed.csv", index=False)
c = pd.concat([pd.read_csv(f"{R}{k}/convergence.csv") for k in CONVF], ignore_index=True)
c = c.drop_duplicates(["arm", "N", "seed", "episode"])
c.to_csv("results/all_conv.csv", index=False)
print(len(d), "per-seed rows;", len(c), "convergence rows")

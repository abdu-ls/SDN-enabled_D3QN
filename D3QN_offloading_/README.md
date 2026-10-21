# Reproducibility package: SDN-enabled Dueling Double DQN Mechanism for energy-efficient task offloading in IoMT
NOTE: This README file applies to the original thesis as well, therefore certain unapplied files or explanations is to be expected.

This package holds the simulator, the agents and policies, the raw run outputs, the analysis scripts, the final result tables and the figures reported in Chapters 3 and 4. It lets a reader (1) re-run the experiments, (2) regenerate every table and figure from the raw outputs, and (3) trace each number in the thesis to a file.

## 1. Contents

| Folder / file | Content |
|---|---|
| `code/iomt_env.py` | Event-driven simulator, state, action, task generation, `compute_reward` |
| `code/policies.py` | Non-learning policies: always-local, random, round-robin, greedy (least-loaded MEC), priority-aware feasible-min-energy rule (and a priority-blind version, not reported) |
| `code/agents_v2.py` | PyTorch DQN with switches for the dueling head and the double-estimator target |
| `code/run_v2.py` | Experiment runner (parallel, resumable); writes `per_seed.csv`, `convergence.csv`, `summary.csv`, `paired_tests.csv` per output folder |
| `raw/` | Raw outputs of every run (7 folders), exactly as produced by `run_v2.py` |
| `analysis/merge_raw.py` | Merges `raw/` into `results/all_per_seed.csv` and `results/all_conv.csv` |
| `analysis/summarise_and_test.py` | Means, t-based 95% CIs, paired Wilcoxon tests with Holm correction and bootstrap CIs (main and architecture families) |
| `analysis/ablation_tests.py` | Paired tests of the five ablation variants against the full agent |
| `analysis/convergence_stats.py` | Training-convergence statistics (Table 4.4) |
| `analysis/make_figures.py` | Figures 4.1-4.6 |
| `results/` | Final tables used in the thesis |
| `figures/` | Figures 4.1-4.6 (PNG). Figure 3.1 is the author's own drawing and is not included |
| `run_commands.sh` | The exact commands for every raw folder and for the analysis chain |
| `requirements.txt` | Software versions (as given in Table 3.1) |
| `SHA256SUMS.txt` | Checksums of all files |

## 2. Experimental design (summary of Chapter 3)

- **System:** wearable devices (N = 20 baseline; 40 and 60 for scaling), 3 MEC servers (5-10 GHz, other-tenant share following a bounded random walk around 0.25 in [0.05, 0.5], FIFO work backlog), a cloud (50 GHz, 20-80 ms wide-area delay, no queue). Wireless rate from the Shannon capacity over 10 MHz with SNR drawn uniformly between 5 and 30 dB per link and decision. Device CPU 0.5-1.0 GHz, transmit power 0.1-0.5 W, battery 2,664 J, switched capacitance 1e-28.
- **Tasks:** Poisson arrivals at 6N tasks/s. Priority tier mix 5/15/80% (Tier 1/2/3) with deadlines 50/100/5,000 ms. Task type light/medium/heavy with probabilities 0.3/0.4/0.3, independent of tier.
- **MDP:** 13-dimensional state (B, f, Q, D, C, P, R, L1-L3, SNR1-SNR3), 5 actions (local, MEC 1-3, cloud). Reward = -(alpha * w * min(T/Tmax, 3) + beta * min(E/0.02, 1) + lambda * 1[T > Tmax]) with alpha = 0.6, beta = 0.4, lambda = 50, w = 4 - P.
- **Agents:** two shared hidden layers of 128 ReLU units, 64-unit value and advantage streams for the dueling networks; Adam, learning rate 1e-3, gamma = 0.95, replay 10,000, batch 64, target update every 100 gradient steps, gradient-norm clip 1.0; epsilon decays geometrically from 1.0 to 0.01 over 20,000 decisions.
- **Protocol:** 500 episodes of 100 decisions (environment not reset between episodes); greedy evaluation on 2,000 decisions in a new ward (environment seed 10000 + training seed); 10 seeds (0-9); common random numbers across methods for the same (N, seed).
- **Methods:** learned: `dueling_ddqn` (proposed, Dueling Double DQN / D3QN), `standard_dqn`, `double_dqn`, `dueling_dqn`. Policies: `always_local`, `random`, `round_robin`, `greedy`, `heuristic_pa` (priority-aware rule). Ablations: `no_priority_reward`, `no_priority_state`, `tier_blind`, `no_sdn` (thesis label "No per-server state": 7-D state B, f, Q, D, C, P, R, which keeps the mean link rate R), `no_sdn_loads` (10-D state without L1-L3).
- **Statistics:** mean and t-based 95% CI (9 degrees of freedom); exact two-sided Wilcoxon signed-rank test on paired per-seed differences (comparison arm minus proposed agent); Holm correction within each family and (N, metric): main = 6 comparisons, architectures = 3, ablation = 5; bootstrap 95% CI of the mean difference (5,000 resamples, generator seed 0). With 10 pairs the smallest attainable p is 0.002.

## 3. Which raw folder holds what

| Folder | Methods | N | Seeds |
|---|---|---|---|
| `raw/ablation_lab_N20`, `raw/ablation_lab_N60` | dueling_ddqn, no_priority_reward, no_priority_state, tier_blind, no_sdn, no_sdn_loads, heuristic_pa | 20 / 60 | 0-9 |
| `raw/nonlearning_baselines` | always_local, greedy, heuristic_pa, random, round_robin | 20, 60 | 0-9 |
| `raw/main_A` | standard_dqn | 20, 40, 60 | 0-9 |
| `raw/main_B1` | dueling_ddqn | 40 | 0-9 |
| `raw/main_B2` | greedy, round_robin, random, always_local, heuristic_pa | 40 | 0-9 |
| `raw/arch_C` | dueling_dqn, double_dqn | 20, 60 | 0-9 |

`heuristic_pa` appears in two folders at N = 20 and 60. It is deterministic given the seed, and `merge_raw.py` keeps the first copy. Total: 350 method x N x seed rows.

## 4. Reproducing the results

```
pip install -r requirements.txt
# Option A: use the supplied raw outputs (minutes)
python analysis/merge_raw.py
python analysis/summarise_and_test.py
python analysis/convergence_stats.py
python analysis/ablation_tests.py
python analysis/make_figures.py
# Option B: re-run the experiments (see run_commands.sh; the learned agents take several core-hours per configuration)
```

Checks made when the package was assembled:
- `merge_raw.py` reproduces `results/all_per_seed.csv` and `results/all_conv.csv` exactly from `raw/`.
- The statistics scripts reproduce `summary_main.csv`, `paired_main.csv`, `conv_stats.csv` and `paired_ablation.csv` (differences below 1e-14, which is floating-point rounding).
- `make_figures.py` reproduces the six figures pixel for pixel.
- Re-running the non-learning policies (greedy, round-robin, priority-aware rule, N = 20, seeds 0-1) with `run_v2.py` gives values identical to the stored ones.
- The learned agents were not re-run when the package was assembled. Their results depend on the PyTorch version, the CPU and the thread count, so a re-run is expected to agree statistically but not bit for bit.

## 5. Where each thesis table and figure comes from

| Thesis item | Source |
|---|---|
| Table 4.1 energy, N = 20, and paired differences | `summary_main.csv` (`energy_mean`), `paired_main.csv` (family `main`, metric `energy_mean`) |
| Tables 4.2, 4.3 time and violations by tier, N = 20 | `summary_main.csv` (`time_t*_mean`, `viol_*`) |
| Table 4.4 convergence | `conv_stats.csv` |
| Section 4.5.1 decision time | `summary_main.csv` (`infer_ms`, per-seed in `all_per_seed.csv`) |
| Tables 4.5, 4.6 architectures | `summary_main.csv` (four learned arms), `paired_main.csv` (family `arch`) |
| Tables 4.7, 4.8 scaling | `summary_main.csv` (N = 20, 40, 60) |
| Tables 4.9, 4.10 ablation | `summary_main.csv`, `paired_ablation.csv` |
| Table 4.11 related work | qualitative, no data file |
| Figures 4.1-4.6 | `figures/`, from `analysis/make_figures.py` |
| Training times (Table 4.4) | `all_per_seed.csv` (`train_time_s`); these depend on batch size and machine load and are only indicative |

Column dictionary for `all_per_seed.csv`: `energy_mean` (J per task), `time_mean`, `time_t1_mean` ... (ms), `viol_overall`, `viol_t1` ... (%), `eval_reward` (canonical full reward, identical for every arm), `t1_p99_ms`, `t1_n`, `t1_viol_n` (Tier-1 task counts), `act_t{tier}_a{action}` (action shares by tier), `train_time_s`, `infer_ms` (ms per decision).

## 6. Known limitations to keep in mind

- All results come from simulation with author-chosen parameters, deadlines and thresholds. The SDN control plane is idealised.
- The `no_sdn` ablation keeps the mean link rate R, so it is a reduced network view and not a strictly device-local one.
- `run_v2.py` lists `heuristic_np` (priority-blind rule) among its default `main` arms; it is not part of the reported results. The ablation default list omits `tier_blind` and `no_sdn_loads`, which were run with explicit `--arms`. `run_commands.sh` states the explicit arms used.
- Training and decision times were measured on shared machines and in parallel jobs, so they are not benchmarks.


- the experiment was run on Google Colab Platform

- Every data need for reproduction can be found here: https://github.com/abdu-ls/SDN-enabled_D3QN.git

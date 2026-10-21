#!/usr/bin/env bash
# Commands that produced the raw runs in raw/. Run from code/. Each command is resumable (same --out folder).
# Settings common to all runs (defaults of run_v2.py): 10 seeds (0-9), 500 episodes x 100 decisions,
# 2,000 test decisions per seed, arrival rate 6 tasks/s/device, epsilon decay over 20,000 decisions.
cd "$(dirname "$0")/code"

# raw/ablation_lab_N20 and raw/ablation_lab_N60 : proposed agent, five ablation variants and the priority-aware rule
python run_v2.py --suite ablation --arms dueling_ddqn,no_priority_reward,no_priority_state,tier_blind,no_sdn,no_sdn_loads,heuristic_pa --N 20 --seeds 10 --jobs 4 --out ../raw/ablation_lab_N20
python run_v2.py --suite ablation --arms dueling_ddqn,no_priority_reward,no_priority_state,tier_blind,no_sdn,no_sdn_loads,heuristic_pa --N 60 --seeds 10 --jobs 4 --out ../raw/ablation_lab_N60

# raw/nonlearning_baselines : non-learning policies at N = 20 and 60
python run_v2.py --suite main --arms always_local,greedy,heuristic_pa,random,round_robin --N 20,60 --seeds 10 --jobs 4 --out ../raw/nonlearning_baselines

# raw/main_A, main_B1, main_B2 : Standard DQN (N = 20, 40, 60), proposed agent at N = 40, non-learning policies at N = 40
python run_v2.py --suite main --arms standard_dqn --N 20,40,60 --seeds 10 --jobs 2 --out ../raw/main_A
python run_v2.py --suite main --arms dueling_ddqn --N 40 --seeds 10 --jobs 2 --out ../raw/main_B1
python run_v2.py --suite main --arms greedy,round_robin,random,always_local,heuristic_pa --N 40 --seeds 10 --jobs 2 --out ../raw/main_B2

# raw/arch_C : Double DQN and Dueling DQN (N = 20, 60)
python run_v2.py --suite arch --arms dueling_dqn,double_dqn --N 20,60 --seeds 10 --jobs 2 --out ../raw/arch_C

# Analysis (from the package root)
cd ..
python analysis/merge_raw.py
python analysis/summarise_and_test.py
python analysis/convergence_stats.py
python analysis/ablation_tests.py
python analysis/make_figures.py

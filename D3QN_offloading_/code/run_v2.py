"""
run_v2.py -- main comparison + limited ablation on the repaired environment.

Examples
  python run_v2.py --suite ablation --seeds 10 --jobs 4
  python run_v2.py --suite main     --seeds 30 --jobs 8
  python run_v2.py --suite ablation --arms dueling_ddqn,tier_blind,no_sdn_loads --N 20,60
  python run_v2.py --suite ablation --episodes 5 --seeds 2 --n-test 300     # smoke test

Design points
  * Every arm with the same (N, seed) is trained on the same hardware draw and evaluated on the
    SAME exogenous stream (common random numbers) -> paired comparisons are valid.
  * Evaluation env uses a different seed (10000+seed): a new ward / new hardware.
  * eval_reward is always the canonical full reward, so ablation arms are comparable even though
    they were trained with different rewards.
  * Results are appended to <out>/per_seed.csv after every job; re-running resumes.
"""
import argparse, os, time, itertools
import numpy as np, pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed

from iomt_env import IoMTEnv, compute_reward, STATE_SIZE, ACTION_SIZE
from policies import AlwaysLocal, RandomPolicy, RoundRobin, Greedy, FeasibleMinEnergy

ALL = list(range(STATE_SIZE))
NO_P   = [i for i in ALL if i != 5]                 # drop priority P_i
LOCAL7 = list(range(7))                             # B,f,Q,D,C,P,R  = device-local view
NO_L   = [i for i in ALL if i not in (7, 8, 9)]     # drop MEC loads only
EVAL_OFFSET = 10_000

ARMS = {
  # learned agents ------------------------------------------------------------
  "dueling_ddqn":       dict(kind="drl", dueling=True,  double=True,  keep=ALL,    reward={}),
  "standard_dqn":       dict(kind="drl", dueling=False, double=False, keep=ALL,    reward={}),   # Eq. (7)
  "double_dqn":         dict(kind="drl", dueling=False, double=True,  keep=ALL,    reward={}),
  "dueling_dqn":        dict(kind="drl", dueling=True,  double=False, keep=ALL,    reward={}),
  # ablations -----------------------------------------------------------------
  "no_priority_reward": dict(kind="drl", dueling=True, double=True, keep=ALL,    reward=dict(uniform_weight=True)),
  "no_priority_state":  dict(kind="drl", dueling=True, double=True, keep=NO_P,   reward={}),
  "tier_blind":         dict(kind="drl", dueling=True, double=True, keep=NO_P,
                             reward=dict(uniform_weight=True, deadline_override=0.100)),
  "no_sdn":             dict(kind="drl", dueling=True, double=True, keep=LOCAL7, reward={}),  # device-local view
  "no_sdn_loads":       dict(kind="drl", dueling=True, double=True, keep=NO_L,   reward={}),
  # non-learning baselines ------------------------------------------------------
  "greedy":       dict(kind="heur", make=lambda s: Greedy()),
  "round_robin":  dict(kind="heur", make=lambda s: RoundRobin()),
  "random":       dict(kind="heur", make=lambda s: RandomPolicy(s)),
  "always_local": dict(kind="heur", make=lambda s: AlwaysLocal()),
  "heuristic_pa": dict(kind="heur", make=lambda s: FeasibleMinEnergy(use_priority=True)),
  "heuristic_np": dict(kind="heur", make=lambda s: FeasibleMinEnergy(use_priority=False)),
}
SUITES = {
  "main":     (["dueling_ddqn", "standard_dqn", "greedy", "round_robin", "random", "always_local",
                "heuristic_pa", "heuristic_np"], [20, 40, 60]),
  "ablation": (["dueling_ddqn", "no_priority_reward", "no_priority_state", "no_sdn", "heuristic_pa"], [20, 60]),
  "arch":     (["dueling_ddqn", "dueling_dqn", "double_dqn", "standard_dqn"], [20, 60]),
}

# ------------------------------------------------------------------------------------
def train_drl(cfg, N, seed, a):
    from agents_v2 import DQNAgent                       # lazy: heuristics need no torch
    try:
        import torch; torch.set_num_threads(1)   # one thread per worker process
    except ImportError:
        pass
    keep = cfg["keep"]
    env = IoMTEnv(N, seed=seed, arrival_rate=a["arrival_rate"]); s = env.reset()
    agent = DQNAgent(len(keep), ACTION_SIZE, dueling=cfg["dueling"], double=cfg["double"],
                     eps_decay_steps=a["eps_decay_steps"], seed=seed)
    conv, t0 = [], time.perf_counter()
    for ep in range(a["episodes"]):
        tot = 0.0
        for _ in range(a["tasks_per_ep"]):
            act = agent.act(s[keep])
            s2, info = env.step(act)
            r = compute_reward(info, **cfg["reward"])
            agent.push(s[keep], act, r, s2[keep]); agent.learn()
            tot += r; s = s2
        conv.append(dict(episode=ep + 1, avg_reward=tot / a["tasks_per_ep"], epsilon=agent.epsilon))
    return agent, conv, time.perf_counter() - t0


def evaluate(act_fn, N, seed, a):
    env = IoMTEnv(N, seed=EVAL_OFFSET + seed, arrival_rate=a["arrival_rate"]); s = env.reset()
    rows, inf = [], []
    for _ in range(a["n_test"]):
        t0 = time.perf_counter(); act = act_fn(s); inf.append(time.perf_counter() - t0)
        s, info = env.step(act)
        rows.append(dict(priority=info["priority"], energy=info["energy"], time_ms=info["time"] * 1e3,
                         violated=info["violated"], action=info["action"], reward=compute_reward(info)))
    return pd.DataFrame(rows), 1e3 * float(np.mean(inf))


def seed_row(df):
    r = dict(energy_mean=df.energy.mean(), time_mean=df.time_ms.mean(), viol_overall=100 * df.violated.mean(),
             eval_reward=df.reward.mean())
    for p in (1, 2, 3):
        d = df[df.priority == p]
        r[f"viol_t{p}"] = 100 * d.violated.mean() if len(d) else np.nan
        r[f"time_t{p}_mean"] = d.time_ms.mean() if len(d) else np.nan
        for k in range(ACTION_SIZE):
            r[f"act_t{p}_a{k}"] = (d.action == k).mean() if len(d) else np.nan
    d1 = df[df.priority == 1]
    r.update(t1_n=len(d1), t1_viol_n=int(d1.violated.sum()),
             t1_p99_ms=float(np.percentile(d1.time_ms, 99)) if len(d1) else np.nan)
    return r


def run_job(arm, N, seed, a):
    cfg = ARMS[arm]; np.random.seed(seed)
    row = dict(arm=arm, N=N, seed=seed); conv = []
    if cfg["kind"] == "drl":
        agent, conv, tt = train_drl(cfg, N, seed, a); keep = cfg["keep"]
        df, inf_ms = evaluate(lambda s: agent.act(s[keep], greedy=True), N, seed, a)
        row.update(train_time_s=tt, infer_ms=inf_ms)
    else:
        pol = cfg["make"](seed)
        df, inf_ms = evaluate(pol.act, N, seed, a); row.update(train_time_s=np.nan, infer_ms=inf_ms)
    row.update(seed_row(df))
    for c in conv: c.update(arm=arm, N=N, seed=seed)
    return row, conv

# ------------------------------------------------------------------------------------
def tci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 2: return (x.mean() if n else np.nan), np.nan, np.nan
    sd = x.std(ddof=1); return x.mean(), sd, stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)

def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

METRICS = ["energy_mean", "time_mean", "time_t1_mean", "time_t2_mean", "time_t3_mean", "viol_overall",
           "viol_t1", "viol_t2", "viol_t3", "eval_reward", "t1_p99_ms"]

def summarise(ps):
    out = []
    for (N, arm), g in ps.groupby(["N", "arm"]):
        r = dict(N=N, arm=arm, n_seeds=len(g))
        for m in METRICS:
            mean, sd, ci = tci(g[m]); r[m], r[m + "_sd"], r[m + "_ci95"] = mean, sd, ci
        n1, k1 = int(g.t1_n.sum()), int(g.t1_viol_n.sum())
        lo, hi = clopper_pearson(k1, n1)
        r.update(t1_pooled_n=n1, t1_pooled_viol=k1, t1_viol_pct_ub95=100 * hi)
        out.append(r)
    return pd.DataFrame(out).sort_values(["N", "arm"])

def paired(ps, ref, n_boot=5000, seed=0):
    rng, rows = np.random.default_rng(seed), []
    for N, gN in ps.groupby("N"):
        base = gN[gN.arm == ref].set_index("seed")
        if base.empty: continue
        for m in ["energy_mean", "viol_overall", "viol_t1", "time_mean", "eval_reward"]:
            block = []
            for arm, g in gN[gN.arm != ref].groupby("arm"):
                g = g.set_index("seed"); common = base.index.intersection(g.index)
                if len(common) < 3: continue
                d = (g.loc[common, m] - base.loc[common, m]).values        # arm minus reference
                p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(d).pvalue
                bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
                block.append(dict(N=N, metric=m, arm=arm, ref=ref, n_pairs=len(d), mean_diff=d.mean(),
                                  boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5), wilcoxon_p=p))
            if block:
                adj = holm([b["wilcoxon_p"] for b in block])
                for b, q in zip(block, adj): b["holm_p"] = q
                rows += block
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="ablation", choices=list(SUITES) + ["all"])
    ap.add_argument("--arms", default=None); ap.add_argument("--N", default=None)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=500); ap.add_argument("--tasks-per-ep", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=2000); ap.add_argument("--arrival-rate", type=float, default=6.0)
    ap.add_argument("--eps-decay-steps", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--out", default="results_v2")
    ap.add_argument("--ref", default="dueling_ddqn")
    args = ap.parse_args()
    a = dict(episodes=args.episodes, tasks_per_ep=args.tasks_per_ep, n_test=args.n_test,
             arrival_rate=args.arrival_rate, eps_decay_steps=args.eps_decay_steps)

    pairs = []
    for name in (SUITES if args.suite == "all" else [args.suite]):
        arms, Ns = SUITES[name]
        if args.arms: arms = args.arms.split(",")
        if args.N: Ns = [int(x) for x in args.N.split(",")]
        pairs += list(itertools.product(arms, Ns))
    pairs = list(dict.fromkeys(pairs))
    for arm, _ in pairs:
        assert arm in ARMS, f"unknown arm {arm}; choices: {list(ARMS)}"

    os.makedirs(args.out, exist_ok=True)
    f_seed, f_conv = f"{args.out}/per_seed.csv", f"{args.out}/convergence.csv"
    done = set()
    if os.path.exists(f_seed):
        d = pd.read_csv(f_seed); done = set(zip(d.arm, d.N, d.seed))
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    jobs = [(arm, N, s) for (arm, N) in pairs for s in seeds if (arm, N, s) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done)")

    def save(row, conv):
        pd.DataFrame([row]).to_csv(f_seed, mode="a", header=not os.path.exists(f_seed), index=False)
        if conv: pd.DataFrame(conv).to_csv(f_conv, mode="a", header=not os.path.exists(f_conv), index=False)

    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(run_job, *j, a): j for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                save(*f.result()); print(f"[{i}/{len(jobs)}] {futs[f]}  {time.time()-t0:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            save(*run_job(*j, a)); print(f"[{i}/{len(jobs)}] {j}  {time.time()-t0:.0f}s", flush=True)

    ps = pd.read_csv(f_seed)
    wanted = {(arm, N) for arm, N in pairs}
    ps = ps[[ (r.arm, r.N) in wanted for r in ps.itertuples()]]
    summ, pt = summarise(ps), paired(ps, args.ref)
    summ.to_csv(f"{args.out}/summary.csv", index=False); pt.to_csv(f"{args.out}/paired_tests.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    cols = ["N", "arm", "n_seeds", "energy_mean", "time_mean", "viol_overall", "viol_t1", "viol_t2",
            "t1_viol_pct_ub95", "eval_reward"]
    print("\n=== SUMMARY (means over seeds) ===\n", summ[cols].round(4).to_string(index=False))
    if len(pt):
        print(f"\n=== PAIRED vs {args.ref} (arm minus ref; Holm-adjusted Wilcoxon) ===\n",
              pt.round(4).to_string(index=False))

if __name__ == "__main__":
    main()

"""Summary table (mean and t-based 95% CI over seeds) and paired tests for the main and architecture families.
Input : results/all_per_seed.csv (from merge_raw.py)
Output: results/summary_main.csv, results/paired_main.csv
Paired test: exact two-sided Wilcoxon signed-rank on per-seed differences (comparison arm minus dueling_ddqn),
Holm correction within each (family, N, metric), bootstrap 95% CI (5,000 resamples, rng seed 0).
Run from the package root:  python analysis/summarise_and_test.py
"""
import pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv('results/all_per_seed.csv')
M=['energy_mean','time_mean','time_t1_mean','time_t2_mean','time_t3_mean','viol_overall','viol_t1','viol_t2','viol_t3','eval_reward','t1_p99_ms','train_time_s','infer_ms']
rows=[]
for (N,a),g in d.groupby(['N','arm']):
    r=dict(N=N,arm=a,n=len(g))
    for m in M:
        x=g[m].dropna().values; r[m]=x.mean(); r[m+'_ci']=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    rows.append(r)
S=pd.DataFrame(rows); S.to_csv('results/summary_main.csv',index=False)
def holm(p):
    o=np.argsort(p); adj=np.empty(len(p)); run=0
    for i,k in enumerate(o):
        run=max(run,(len(p)-i)*p[k]); adj[k]=min(run,1)
    return adj
def paired(ref,arms,fam):
    out=[]
    for N in (20,40,60):
        for m in M:
            tmp=[]
            for a in arms:
                x=d[(d.arm==ref)&(d.N==N)].set_index('seed')[m]; y=d[(d.arm==a)&(d.N==N)].set_index('seed')[m]
                j=x.index.intersection(y.index)
                if len(j)<10: continue
                df=(y[j]-x[j]).values
                p=stats.wilcoxon(df).pvalue if np.any(df!=0) else 1.0
                rng=np.random.default_rng(0); b=[rng.choice(df,len(df)).mean() for _ in range(5000)]
                tmp.append(dict(family=fam,N=N,metric=m,arm=a,mean_diff=df.mean(),lo=np.percentile(b,2.5),hi=np.percentile(b,97.5),p=p))
            if tmp:
                h=holm(np.array([t['p'] for t in tmp]))
                for t,hh in zip(tmp,h): t['holm_p']=hh
                out+=tmp
    return out
P=pd.DataFrame(paired('dueling_ddqn',['standard_dqn','heuristic_pa','greedy','round_robin','random','always_local'],'main')+paired('dueling_ddqn',['standard_dqn','dueling_dqn','double_dqn'],'arch'))
P.to_csv('results/paired_main.csv',index=False)
pd.set_option('display.width',250)
print(S[['N','arm','n','energy_mean','time_mean','time_t1_mean','viol_overall','viol_t1','viol_t1_ci','viol_t2']].assign(energy_mean=lambda x:x.energy_mean*1e3).round(3).to_string())
print(P[P.metric.isin(['viol_t1','energy_mean'])].round(4).to_string())

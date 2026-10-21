"""Paired Wilcoxon tests of each ablation variant against the full agent (arm minus dueling_ddqn),
Holm-adjusted over the five variants within each (N, metric). Input: results/all_per_seed.csv. Output: results/paired_ablation.csv.
Run from the package root."""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
d=pd.read_csv('results/all_per_seed.csv')
abl=['no_priority_reward','no_priority_state','tier_blind','no_sdn','no_sdn_loads']
out=[]
for N in (20,60):
  for m in ['energy_mean','time_mean','time_t1_mean','viol_overall','viol_t1','eval_reward','t1_p99_ms']:
    ref=d[(d.N==N)&(d.arm=='dueling_ddqn')].set_index('seed')[m]
    ps=[]
    for a in abl:
        x=d[(d.N==N)&(d.arm==a)].set_index('seed')[m]; df=(x-ref).dropna()
        try: p=wilcoxon(df).pvalue if (df!=0).any() else 1.0
        except Exception: p=1.0
        ps.append((a,df.mean(),p))
    # Holm
    order=sorted(range(len(ps)),key=lambda i:ps[i][2]); k=len(ps); adj=[0]*k; run=0
    for r,i in enumerate(order):
        run=max(run,min(1,(k-r)*ps[i][2])); adj[i]=run
    for i,(a,md,p) in enumerate(ps): out.append(dict(N=N,metric=m,arm=a,mean_diff=md,p=p,holm_p=adj[i]))
o=pd.DataFrame(out); o.to_csv('results/paired_ablation.csv',index=False)
pd.set_option('display.width',200); pd.set_option('display.max_rows',100)
print(o[o.metric.isin(['energy_mean','viol_t1','viol_overall','eval_reward','time_mean'])].round(5).to_string())

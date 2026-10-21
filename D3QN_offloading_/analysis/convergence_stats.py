"""Convergence statistics per run (first-10 and last-50 episode mean reward, episode of convergence).
Input: results/all_per_seed.csv, results/all_conv.csv. Output: results/conv_stats.csv. Run from the package root."""
import pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv('results/all_per_seed.csv'); print(d.columns.tolist()); print(d.groupby(['N','arm']).size().unstack(0))
c=pd.read_csv('results/all_conv.csv')
rows=[]
for (arm,N,seed),g in c.groupby(['arm','N','seed']):
    g=g.sort_values('episode'); r=g.avg_reward.values
    first=r[:10].mean(); last=r[-50:].mean()
    ma=pd.Series(r).rolling(20,min_periods=20).mean().values
    thr=0.05*abs(last-first)+1e-9
    ok=np.abs(ma-last)<=max(thr,0.02*abs(last))
    # first index from which ok stays True
    idx=None
    for i in range(len(ok)):
        if ok[i:].all() if not np.isnan(ma[i:]).any() else False: idx=i+1;break
    rows.append(dict(arm=arm,N=N,seed=seed,ep1=r[0],first10=first,last50=last,conv=idx if idx else np.nan,last50sd=r[-50:].std()))
cs=pd.DataFrame(rows); cs.to_csv('results/conv_stats.csv',index=False)
print(cs.groupby(['N','arm']).agg(first10=('first10','mean'),last50=('last50','mean'),conv=('conv','mean'),convmed=('conv','median'),sd=('last50sd','mean')).round(3))

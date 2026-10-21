"""Regenerates Figures 4.1-4.6 from results/summary_main.csv and results/all_conv.csv. Run from the package root: python analysis/make_figures.py"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='results/'
S=pd.read_csv(R+'summary_main.csv').set_index(['arm','N'])
C=pd.read_csv(R+'all_conv.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#555','axes.linewidth':.6})
COL={20:'#9ECAD0',40:'#3F9AA6',60:'#0B5563'}
NAMES={'dueling_ddqn':'D3QN','standard_dqn':'Standard\nDQN','heuristic_pa':'Priority-\naware rule','greedy':'Greedy','round_robin':'Round-\nRobin','random':'Random','always_local':'Always-\nlocal',
 'no_priority_reward':'No priority\nreward','no_priority_state':'No priority\nin state','tier_blind':'Tier-\nblind','no_sdn':'No per-\nserver state','no_sdn_loads':'No MEC\nloads'}
def grouped(arms,Ns,key,scale,ylabel,fn,size,log=False,hline=None,ylim=None,fmt='{:.1f}'):
    fig,ax=plt.subplots(figsize=size); w=0.8/len(Ns); x=np.arange(len(arms))
    for j,N in enumerate(Ns):
        v=[];e=[]
        for a in arms:
            if (a,N) in S.index: v.append(S.loc[(a,N),key]*scale); e.append(S.loc[(a,N),key+'_ci']*scale)
            else: v.append(np.nan); e.append(0)
        v=np.array(v); e=np.array(e); lo=np.minimum(e,np.nan_to_num(v)-(1e-9 if log else 0)) if not log else np.minimum(e,v*0.9)
        ax.bar(x+(j-(len(Ns)-1)/2)*w,v,w*0.92,color=COL[N],label=f'N = {N}',yerr=[lo,e],error_kw=dict(lw=.6,capsize=1.2,ecolor='#222'))
    ax.set_xticks(x); ax.set_xticklabels([NAMES[a] for a in arms],fontsize=6.5)
    ax.set_ylabel(ylabel)
    if log: ax.set_yscale('log')
    if hline: ax.axhline(hline[0],ls='--',lw=.8,color='#C0392B'); ax.text(len(arms)-0.5,hline[0]*1.08,hline[1],ha='right',fontsize=6.5,color='#C0392B')
    if ylim: ax.set_ylim(*ylim)
    ax.legend(frameon=False,fontsize=7,ncol=len(Ns)); ax.yaxis.grid(True,lw=.3,color='#ddd'); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig('figures/'+fn,dpi=220); plt.close(fig)
ALG=['dueling_ddqn','standard_dqn','heuristic_pa','greedy','round_robin','random','always_local']
grouped(ALG,[20,40,60],'energy_mean',1000,'Energy per task (mJ)','Fig4.1_energy.png',(5.27,2.88))
grouped(ALG,[20,40,60],'time_t1_mean',1,'Tier-1 mean completion time (ms, log)','Fig4.2_tier1_time.png',(5.51,3.3),log=True,hline=(50,'Tier-1 deadline (50 ms)'))
grouped(ALG,[20,40,60],'viol_t1',1,'Tier-1 deadline violations (%)','Fig4.3_tier1_violation.png',(5.15,3.06),ylim=(0,105))
# convergence
fig,axs=plt.subplots(1,2,figsize=(6.35,2.36),sharey=True)
for ax,N in zip(axs,[20,60]):
    for arm,col,lab in [('dueling_ddqn','#0B5563','D3QN'),('standard_dqn','#C0792B','Standard DQN')]:
        g=C[(C.arm==arm)&(C.N==N)].pivot(index='episode',columns='seed',values='avg_reward').rolling(10,min_periods=1).mean()
        m=g.mean(1); s=g.std(1)
        ax.plot(m.index,m,color=col,lw=1,label=lab); ax.fill_between(m.index,m-s,m+s,color=col,alpha=.15,lw=0)
    ax.set_title(f'N = {N}',fontsize=8); ax.set_xlabel('Training episode'); ax.set_ylim(-3,0.1); ax.yaxis.grid(True,lw=.3,color='#ddd')
axs[0].set_ylabel('Mean reward per decision\n(10-episode average)'); axs[0].legend(frameon=False,fontsize=7,loc='lower right')
fig.tight_layout(); fig.savefig('figures/Fig4.4_training_reward.png',dpi=220); plt.close(fig)
ABL=['dueling_ddqn','no_priority_reward','no_priority_state','tier_blind','no_sdn','no_sdn_loads']
grouped(ABL,[20,60],'viol_t1',1,'Tier-1 deadline violations (%)','Fig4.5_ablation_violation.png',(5.22,2.90),ylim=(0,40))
grouped(ABL,[20,60],'energy_mean',1000,'Energy per task (mJ)','Fig4.6_ablation_energy.png',(6.81,2.57),ylim=(0,2.0))
print('done')

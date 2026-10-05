from pathlib import Path
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
P = ROOT/'data/processed'
OUT = P/'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False,'lines.linewidth':1.2})
def read(n): return json.loads((P/n).read_text(encoding='utf-8'))
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=400,bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(6.69,5.2))
ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
boxes=[(.06,.85,.88,.11,'Observed baseline extract\n2985 contacts in 924 households'),
       (.02,.65,.48,.13,'Recorded TST: 2725 contacts\n877 households; 359 positive\nTST diameter ≥10 mm'),
       (.56,.65,.42,.13,'Missing TST: 260 contacts\n47 households wholly unobserved'),
       (.06,.45,.88,.12,'Assumed workflow: random order within each household\nEarlier members’ results assumed returned before ranking\nNo observed screening or result-return timestamps'),
       (.02,.24,.48,.13,'Household-grouped five-fold CV\n20 seeds; fold-specific preprocessing\nSame household always in one fold'),
       (.56,.24,.42,.13,'Paired random-forest models\n29 baseline versus 29 + 4 features\nTarget’s own result excluded'),
       (.06,.03,.88,.12,'Separate evaluations: raw ranking, nested calibration and charged budget\nConditional intervals distinguished from full-pipeline refit intervals\nNo observed clinical implementation')]
for x,y,w,h,t in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor='white',edgecolor='#333333',linewidth=.9))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8.4)
for a,b in [((.3,.85),(.26,.79)),((.7,.85),(.77,.79)),((.26,.65),(.32,.58)),((.3,.45),(.26,.38)),((.7,.45),(.77,.38)),((.26,.24),(.35,.16)),((.77,.24),(.65,.16))]:
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#333333','lw':.9})
save(fig,'Figure_1_workflow')

primary=read('sop_unified_ci_leakfree_homeacf_20261001.json')
controls=read('samecohort_controls_homeacf_20261001.json')['results']
budget=read('homeacf_dpr_budget_audit_v2.json')['results']
metrics=read('dpr_same_information_baseline_v1/result.json')['metrics']
fig,(a,b)=plt.subplots(2,1,figsize=(6.69,5.25),layout='constrained')
x=np.arange(2)
for j,(arm,label,color) in enumerate([('rf_base','Baseline','#8798aa'),('rf_sop','Baseline + SOP','#2473a6')]):
    vals=[metrics[arm][m]['mean'] if isinstance(metrics[arm][m],dict) else metrics[arm][m] for m in ['auroc','auprc']]
    bars=a.bar(x+(j-.5)*.32,vals,.32,label=label,color=color)
    a.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
a.set_xticks(x,['AUROC','AUPRC']);a.set_ylim(0,.84)
a.set_title('A  Raw discrimination: means over 20 seeds',loc='left');a.legend(loc='upper right',fontsize=9)
for i,kind in enumerate(['RF','LR']):
    rr=[budget[f'{kind}_{q}'] for q in [.1,.2,.3]]
    yy=np.array([r['delta'] for r in rr])*100;ci=np.array([r['conditional_ci95'] for r in rr])*100
    b.errorbar(np.array([10,20,30])+(i-.5)*.7,yy,yerr=[yy-ci[:,0],ci[:,1]-yy],fmt='o-',capsize=3,label=kind)
b.axhline(0,color='gray',ls='--',lw=.8);b.set_xticks([10,20,30]);b.legend()
b.set_xlabel('Total test budget (%)');b.set_ylabel('Capture gain (percentage points)')
b.set_title('B  Charged two-wave rule versus its own static baseline',loc='left')
save(fig,'Figure_2_performance_budget')

fig,(a,b,c)=plt.subplots(3,1,figsize=(6.69,6.15))
rows=[('Primary SOP gain',primary['effects']['ep_minus_exp']),('Global permutation',controls['b1a_global_perm']),('Within-household permutation',controls['b1b_within_hh_perm']),('Future-result gain',primary['effects']['fut_minus_exp'])]
rows2=[('Future minus SOP',primary['effects']['fut_minus_prior']),('Past minus future: window 1',controls['b1c_matched_windows']['k1']['past_minus_fut']),('Past minus future: window 2',controls['b1c_matched_windows']['k2']['past_minus_fut'])]
for ax,rr,color in [(a,rows,'#2473a6'),(b,rows2,'#b36c2d')]:
    for j,(label,r) in enumerate(rr):
        m=r['mean'];lo,hi=r['ci95_unified'];ax.errorbar(m,j,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=color)
    ax.set_yticks(range(len(rr)),[r[0] for r in rr]);ax.invert_yaxis();ax.axvline(0,color='gray',ls='--',lw=.8)
a.set_xlabel('ΔAUROC versus baseline');a.set_title('A  Outcome-structure controls',loc='left')
b.set_xlabel('Direction contrast (ΔAUROC)');b.set_title('B  Distinct order contrasts',loc='left')
rr=[controls['b1d_pseudo_granularity'][f'k{k}'] for k in [1,2,4,8]]
yy=np.array([r['mean'] for r in rr]);ci=np.array([r['ci95_unified'] for r in rr])
c.errorbar(range(4),yy,yerr=[yy-ci[:,0],ci[:,1]-yy],fmt='o-',capsize=3)
c.set_xticks(range(4),['1','2','4','8']);c.set_xlabel('Households merged per pseudo-group');c.set_ylabel('ΔAUROC')
c.set_title('C  Pseudo-group analysis',loc='left')
fig.subplots_adjust(left=.35,right=.98,top=.96,bottom=.08,hspace=.8)
save(fig,'Figure_3_controls')

sim=read('dpr_d1_strengthened_v1/summary.json')
cells=[dict(counts={k:v['count'] for k,v in row['outputs'].items()},
            mc_ci95={k:v['monte_carlo_exact_ci95'] for k,v in row['outputs'].items()})
       for row in sim['summaries_lr']]
labels=list(cells[0]['counts'])
order_label=next(x for x in labels if 'order' in x.lower())
ind_label=next(x for x in labels if 'indeterminate' in x.lower())
fig,axes=plt.subplots(1,2,figsize=(6.69,3.0),layout='constrained')
for ax,label,title in zip(axes,[order_label,ind_label],['A  Order-specific output (%)','B  Indeterminate output (%)']):
    vals=np.array([x['counts'][label] for x in cells[:9]]).reshape(3,3)
    im=ax.imshow(vals,vmin=0,vmax=100,cmap='Blues')
    for i in range(3):
        for j in range(3): ax.text(j,i,str(vals[i,j]),ha='center',va='center',color='white' if vals[i,j]>55 else 'black',fontsize=10)
    ax.set_xticks(range(3),['0','1','2']);ax.set_yticks(range(3),['0','0.8','1.6'])
    ax.set_xlabel('Time-trigger strength (γ)');ax.set_ylabel('Household strength (σ)');ax.set_title(title,fontsize=9)
    fig.colorbar(im,ax=ax,shrink=.75)
save(fig,'Figure_4_simulation')

cal=read('homeacf_dpr_nested_calibration_v1.json')
fig,axs=plt.subplots(1,2,figsize=(6.69,3.15),layout='constrained')
for ax,kind,title in zip(axs,['raw','nested_calibrated'],['A  Uncalibrated scores','B  Nested calibration']):
    for arm,label,color in [('exposure','Baseline','#3568a8'),('exposure_prior','Baseline + SOP','#cf7333')]:
        bins=cal['rows'][0]['arms'][arm][kind]['bins']
        ax.plot([b['mean_predicted'] for b in bins],[b['observed'] for b in bins],'-',color=color,label=label,lw=1)
        ax.scatter([b['mean_predicted'] for b in bins],[b['observed'] for b in bins],s=[max(10,min(70,b['n']/12)) for b in bins],color=color,zorder=3)
        for i,bn in enumerate(bins):
            if bn['n']<20: ax.annotate('n='+str(bn['n']),(bn['mean_predicted'],bn['observed']),xytext=(3,5 if i%2 else -11),textcoords='offset points',fontsize=7.5,color=color)
    ax.plot([0,.8],[0,.8],'--',color='gray',lw=.8);ax.set(xlim=(0,.8),ylim=(0,.8),xlabel='Mean predicted probability',ylabel='Observed TST-positive fraction',title=title)
    ax.legend(fontsize=7.8,loc='upper left')
save(fig,'Figure_S1_calibration')

pts=np.array([float(x) for x in cal['dca']]);rr=list(cal['dca'].values())
fig,(a,b)=plt.subplots(2,1,figsize=(6.69,5.05),layout='constrained')
for i,label,color in [(0,'Baseline','#3568a8'),(1,'Baseline + SOP','#cf7333')]:a.plot(pts,[v['arms_mean'][i] for v in rr],'o-',label=label,color=color)
a.plot(pts,[v['treat_all_nb'] for v in rr],'--',label='Test all',color='gray');a.axhline(0,color='black',ls=':',label='Test none');a.legend(fontsize=8)
a.set(xlabel='Illustrative threshold probability',ylabel='Net benefit',title='A  Exploratory nested-calibrated DCA')
vals=np.array([v['delta_mean'] for v in rr]);ci=np.array([v['conditional_ci95'] for v in rr])
b.errorbar(pts,vals,yerr=[vals-ci[:,0],ci[:,1]-vals],fmt='o',capsize=3,color='#cf7333')
b.axhline(0,color='gray',ls='--',lw=.8);b.set(xlabel='Illustrative threshold probability',ylabel='SOP minus baseline net benefit',title='B  Paired conditional household-bootstrap intervals')
save(fig,'Figure_S2_dca')

ids=[6,19,2,24,8];names=['Household only / random','Household only / age sorted','Time only / 13% positive','Time only / 30% positive','Both mechanisms / random']
ys=np.array([cells[i]['counts'][order_label] for i in ids]);cis=np.array([cells[i]['mc_ci95'][order_label] for i in ids])*100
fig,ax=plt.subplots(figsize=(6.69,3.5))
ax.barh(names,ys,color=['#4575b4','#4575b4','#d95f02','#d95f02','#7570b3'])
ax.errorbar(ys,np.arange(5),xerr=np.vstack([ys-cis[:,0],cis[:,1]-ys]),fmt='none',ecolor='black',capsize=3)
for y,k in enumerate(ys):ax.text(min(cis[y,1]+1.5,96),y,f'{k}/100',va='center',fontsize=9)
ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('Order-specific output (%) with exact 95% MC intervals')
fig.subplots_adjust(left=.35,right=.98,top=.97,bottom=.18)
save(fig,'Figure_S3_simulation_boundaries')
print('Seven figures redrawn from original archived results; no estimates recalculated.')

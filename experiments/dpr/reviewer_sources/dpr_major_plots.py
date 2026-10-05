from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(r'D:\tb_risk_DPR\data\processed/dpr_major_review_v011_20261003');(R/'figures').mkdir(exist_ok=True)
ROOT=Path(r'D:\tb_risk_DPR');plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,n):
 fig.savefig(R/'figures'/f'{n}.png',dpi=300,bbox_inches='tight');fig.savefig(R/'figures'/f'{n}.pdf',bbox_inches='tight');plt.close(fig)
si=json.loads((ROOT/'data/processed/dpr_same_information_baseline_v1/result.json').read_text(encoding="utf8"));bud=json.loads((ROOT/'data/processed/homeacf_dpr_budget_audit_v2.json').read_text(encoding="utf8"))
fig,ax=plt.subplots(1,2,figsize=(11,4.2));colors=['#607889','#1b705c']
for j,n in enumerate(['rf_base','rf_sop']):
 vals=[si['metrics'][n][m] for m in ['auroc','auprc']];allv=np.array([[s['metrics'][n][m] for m in ['auroc','auprc']] for s in si['records']]);xx=np.arange(2)+(j-.5)*.28
 ax[0].bar(xx,vals,width=.26,color=colors[j],label=['Baseline','Four-feature SOP'][j]);ax[0].errorbar(xx,vals,yerr=[np.array(vals)-allv.min(0),allv.max(0)-np.array(vals)],fmt='none',ecolor='black',capsize=4)
 for x,y in zip(xx,vals):ax[0].text(x,y+.027,f'{y:.3f}',ha='center',fontsize=9)
ax[0].set_xticks([0,1],['AUROC','AUPRC']);ax[0].set_ylim(0,.84);ax[0].set_ylabel('Mean performance');ax[0].legend(frameon=False);ax[0].set_title('A  Random-order prediction comparison',loc='left',fontsize=11)
ax[0].text(.02,-.24,'Whiskers: 20-seed minima and maxima, not confidence intervals.',transform=ax[0].transAxes,fontsize=8)
for j,name in enumerate(['RF','LR']):
 ks=[.1,.2,.3];v=np.array([bud['results'][f'{name}_{k}']['delta']*100 for k in ks]);ci=np.array([bud['results'][f'{name}_{k}']['conditional_ci95'] for k in ks])*100
 ax[1].errorbar(np.array(ks)*100+(j-.5)*.6,v,yerr=[v-ci[:,0],ci[:,1]-v],marker=['o','s'][j],capsize=4,color=colors[j],label=name,linewidth=1.3)
ax[1].axhline(0,color='#888',ls='--',lw=.9);ax[1].set_xticks([10,20,30]);ax[1].set_xlabel('Total test budget (%)');ax[1].set_ylabel('Capture difference (percentage points)');ax[1].legend(frameon=False);ax[1].set_title('B  Exploratory charged rule (k = 2, w = 7)',loc='left',fontsize=11)
ax[1].text(.02,-.24,'Conditional intervals omit historical parameter-selection uncertainty.',transform=ax[1].transAxes,fontsize=8)
fig.subplots_adjust(bottom=.25,wspace=.33);save(fig,'Figure_2_performance_budget')
grid=json.loads((R/'analysis/primary_grid_verified.json').read_text(encoding="utf8"))[1:]
fig,axs=plt.subplots(2,2,figsize=(9,7.4),sharex=True,sharey=True)
for ax,col,title in zip(axs.ravel(),[3,4,5,6],['A  No detectable gain','B  Order-specific signal','C  Household-structure consistent','D  Indeterminate']):
 arr=np.zeros((3,3))
 for row in grid:arr[[0,.8,1.6].index(float(row[0])),[0,1,2].index(float(row[1]))]=int(row[col])
 im=ax.imshow(arr,vmin=0,vmax=100,cmap='Blues',origin='lower');ax.set_title(title,loc='left',fontsize=11)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{arr[i,j]:.0f}%',ha='center',va='center',color='white' if arr[i,j]>=55 else 'black',fontsize=12)
 ax.set_xticks([0,1,2],['0','1','2']);ax.set_yticks([0,1,2],['0','0.8','1.6']);ax.set_xlabel('Previous-positive trigger strength γ');ax.set_ylabel('Household-effect strength σ')
fig.suptitle('LR diagnostic outputs: 100 datasets per cell, not causal classifications',fontsize=12);fig.tight_layout(rect=[0,0,1,.96]);save(fig,'Figure_S4_all_diagnostic_outputs')
o=json.loads((R/'analysis/order_sensitivity.json').read_text(encoding="utf8"))['results'];sc=list(o);labels=['Random','Age ↑','Age ↓','Baseline risk ↓'];fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for j,(n,c) in enumerate([('rf_four','#1b705c'),('lr_eb','#5269a9'),('lr_four','#c48735')]):
 rows=[next(r for r in o[s] if r['model']==n) for s in sc];x=np.arange(4)+(j-1)*.12;v=np.array([r['delta_auroc'] for r in rows]);ci=np.array([r['conditional_ci95'] for r in rows]);name={'rf_four':'RF, four features','lr_eb':'LR, single EB rate','lr_four':'LR, four features'}[n]
 axs[0].errorbar(x,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='o',capsize=3,color=c,label=name)
 cap=np.array([100*r['free_capture_10'] for r in rows]);ranges=np.array([r['capture_seed_range'] for r in rows])*100
 axs[1].errorbar(x,cap,yerr=[cap-ranges[:,0],ranges[:,1]-cap],fmt='o',capsize=3,color=c,label=name)
axs[0].axhline(0,lw=.8,color='gray',ls='--');axs[0].set_ylabel('AUROC difference versus own baseline');axs[1].set_ylabel('TST-positive capture at 10% (%)')
for a in axs:a.set_xticks(range(4),labels);a.set_xlabel('Assumed household order')
axs[0].set_title('A  Conditional 95% household intervals',loc='left',fontsize=11);axs[1].set_title('B  Optimistic free-evidence scenario',loc='left',fontsize=11);axs[1].legend(fontsize=8,frameon=False)
fig.text(.51,.02,'Capture whiskers: 20-seed ranges, not confidence intervals.',fontsize=8);fig.tight_layout(rect=[0,.04,1,1]);save(fig,'Figure_S5_order_sensitivity')
b=json.loads((R/'analysis/charged_parameter_sensitivity.json').read_text(encoding="utf8"))['records'];fig,axs=plt.subplots(1,2,figsize=(10,4.2));qs=[.01,.025,.05,.075];ws=[1,3,5,7,9,12]
for ax,model in zip(axs,['RF','LR']):
 arr=np.array([[next(r['delta_pp'] for r in b if r['model']==model and r['family']=='w_q_surface' and r['total_budget']==.1 and r['q']==q and r['w']==w) for w in ws] for q in qs])
 ax.imshow(arr,origin='lower',vmin=0,vmax=4,cmap='YlGnBu',aspect='auto')
 for i in range(4):
  for j in range(6):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if arr[i,j]>2.7 else 'black')
 ax.set_xticks(range(6),ws);ax.set_yticks(range(4),[1,2.5,5,7.5]);ax.set_xlabel('Update weight w');ax.set_ylabel('First-wave fraction (%)');ax.set_title(f'{model}: capture difference (percentage points)',fontsize=11)
fig.suptitle('10% total budget, k = 2; descriptive sensitivity, not policy selection',fontsize=12);fig.tight_layout(rect=[0,0,1,.93]);save(fig,'Figure_S6_charged_parameter_surface')
print('Four figures regenerated from verified results.')

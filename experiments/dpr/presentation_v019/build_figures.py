"""Redraw accepted aggregate results; no model fitting or new inference."""
from pathlib import Path
import json, shutil, hashlib, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=Path(__file__).resolve().parent
BUNDLED=(HERE/'inputs').exists()
BASE=HERE/'inputs' if BUNDLED else Path('D:/tb_risk_DPR')
E=BASE/'experiments/dpr/evidence/results'
OLD=BASE/'data/processed/dpr_final_revision_v017_20261005'
OUT=Path(os.environ.get('DPR_FIGURE_OUTPUT',str(HERE/'regenerated' if BUNDLED else Path('D:/tb_risk/dpr_v019_revision_20261102'))))
F=OUT/'figures'
F.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
C=['#365A7A','#217D6A','#8B5B9A']
provenance=[]
def save(fig,name,sources,detail):
 for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
 provenance.append({'figure':name,'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'detail':detail,'new_model_fits':0})

# Illustration, not participant data. Coordinates specify a designed diagram.
fig,ax=plt.subplots(figsize=(10.5,7.2));ax.set_xlim(0,10);ax.set_ylim(0,7);ax.axis('off')
def box(x,y,w,h,text,fill='#EFF4F7',size=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',facecolor=fill,edgecolor='#52616D',linewidth=1))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.45)
def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=13,color='#52616D',linewidth=1.1))
ax.text(0,6.8,'A  Information available at a hypothetical ranking decision',fontsize=12,weight='bold')
box(.1,5.3,2.65,1.05,'Illustrative household\nEarlier members: + and -\nTarget member: result unknown')
box(3.35,5.3,2.65,1.05,'Four SOP summaries\nCount; positive count;\npositive rate; availability')
box(6.6,5.3,3.15,1.05,'Simpler comparator\nOne empirical-Bayes\nhousehold shrinkage rate')
arrow(2.85,5.85,3.25,5.85)
ax.add_patch(FancyArrowPatch((2.75,6.22),(6.6,6.22),connectionstyle='arc3,rad=-0.15',arrowstyle='-|>',mutation_scale=13,color='#52616D',linewidth=1.1))
ax.text(.12,4.91,'Only other members\' results assumed returned before ranking may enter; the target\'s own TST is excluded.',fontsize=9.2)
ax.text(0,4.42,'B  Paired model comparisons and evaluation',fontsize=12,weight='bold')
box(.1,2.85,2.65,1.15,'HomeACF: 2985 contacts / 924 households\n260 excluded: missing TST\nAnalysis: 2725 contacts / 877 households\n359 TST-positive contacts',size=8.5)
box(3.35,2.85,2.65,1.15,'Same baseline predictors\nBaseline / four SOP features /\nEB representations in RF or LR')
box(6.6,2.85,3.15,1.15,'Household-grouped CV\nTraining-fold preprocessing\nSame folds and assumed orders')
arrow(2.85,3.42,3.25,3.42);arrow(6.1,3.42,6.5,3.42)
box(.1,.9,4.4,1.2,'Predictive information and representation\nRaw AUROC / AUPRC; paired contrasts\n300 full-refit samples; 20-seed descriptions')
box(5.15,.9,4.6,1.2,'Operating assumptions and interpretation\nOrder sensitivity; acquisition-cost scenarios\nSeparate nested calibration; retrospective controls')
arrow(7.9,2.77,7.6,2.18);arrow(7.2,2.77,2.8,2.18)
ax.text(.1,.35,'Screening order and timely return are assumed, not observed. The diagram does not establish deployment effectiveness.',fontsize=9.2)
save(fig,'Figure_1_research_framework',[E/'homeacf_dpr_f1_final_performance_20261002_v3.json'],'Schematic of performed analyses; household outcomes are illustrative, not observed records. EB and four-feature arms are parallel representations of the same information.')

same=load(E/'dpr_same_information_baseline_v1/result.json')
full=load(E/'homeacf_dpr_a1_fullrefit_20261001.json')['delta_auroc']
weighted=load(E/'homeacf_dpr_e1_fullrefit_v2.json')['summary']
fig,axs=plt.subplots(1,2,figsize=(10.5,4.1),gridspec_kw={'width_ratios':[1.15,1]})
for y,(v,lohi,label,c) in enumerate([(full['mean'],full['ci95'],'Primary unweighted',C[1]),(weighted['mean'],weighted['ci95_percentile'],'IPW sensitivity',C[0])]):
 lo,hi=lohi;axs[0].errorbar(v,1-y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=c,markersize=7)
 axs[0].text(.0,1-y+.21,f'{label}: {v:+.3f} [{lo:+.3f}, {hi:+.3f}]',fontsize=9)
axs[0].axvline(0,ls='--',lw=.9,color='gray');axs[0].set_xlim(-.004,.116);axs[0].set_ylim(-.55,1.65);axs[0].set_yticks([]);axs[0].set_xlabel('Mean AUROC difference');axs[0].set_title('A  Full-refit household bootstrap',loc='left',fontsize=11)
for j,(arm,color,label) in enumerate([('rf_base',C[0],'Baseline RF'),('rf_sop',C[1],'RF + four SOP features')]):
 vals=np.array([same['metrics'][arm][m] for m in ['auroc','auprc']]);arr=np.array([[r['metrics'][arm][m] for m in ['auroc','auprc']] for r in same['records']]);x=np.arange(2)+(j-.5)*.3
 axs[1].bar(x,vals,width=.27,color=color,label=label);axs[1].errorbar(x,vals,yerr=[vals-arr.min(0),arr.max(0)-vals],fmt='none',ecolor='#333333',capsize=3)
 for xx,yy in zip(x,vals):axs[1].text(xx,yy+.035,f'{yy:.3f}',ha='center',fontsize=9)
axs[1].set_xticks([0,1],['AUROC','AUPRC']);axs[1].set_ylim(0,.88);axs[1].legend(frameon=False,fontsize=8,loc='upper right');axs[1].set_title('B  Descriptive 20-seed means',loc='left',fontsize=11)
fig.text(.02,.015,'A: 300 refits with CV seed 0 fixed. B: whiskers are seed minima/maxima, not confidence intervals.',fontsize=8.5)
fig.tight_layout(rect=[0,.07,1,1]);save(fig,'Figure_2_primary_performance',[E/'homeacf_dpr_a1_fullrefit_20261001.json',E/'homeacf_dpr_e1_fullrefit_v2.json',E/'dpr_same_information_baseline_v1/result.json'],'Full-refit and repeated-seed procedures in distinct panels; no mixing of estimands.')

fig,axs=plt.subplots(2,1,figsize=(9.3,6.8))
groups=[([('rf_sop_vs_base','RF: four features'),('rf_eb_vs_base','RF: single EB rate'),('rf_eb_counts_vs_base','RF: rate, count, availability'),('lr_sop_vs_base','LR: four features'),('lr_eb_vs_base','LR: single EB rate')],'A  Gain versus each algorithm\'s baseline'),([('rf_sop_vs_eb','RF: four minus single rate'),('rf_sop_vs_eb_counts','RF: four minus rate, count, availability'),('lr_sop_vs_eb','LR: four minus single rate')],'B  Additional gain from four features')]
for ax,(rows,title) in zip(axs,groups):
 for y,(key,label) in enumerate(rows):
  r=same['comparisons'][key];v=r['delta_auroc'];lo,hi=r['conditional_ci95'];ax.errorbar(v,len(rows)-1-y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=C[0] if key.startswith('lr') else C[1])
 ax.set_yticks(range(len(rows)),[x[1] for x in rows][::-1],fontsize=9);ax.axvline(0,ls='--',color='gray',lw=.9);ax.set_title(title,loc='left',fontsize=11);ax.set_xlabel('Paired AUROC difference');ax.set_ylim(-.65,len(rows)-.35)
axs[0].set_xlim(-.004,.103);axs[1].set_xlim(-.0035,.013)
fig.subplots_adjust(left=.41,right=.97,hspace=.65,bottom=.12,top=.94)
fig.text(.03,.02,'Conditional 95% intervals: 2000 paired household resamples of fixed predictions. No equivalence test was performed.',fontsize=8.5)
save(fig,'Figure_3_same_information',[E/'dpr_same_information_baseline_v1/result.json'],'Existing paired comparisons, conditional intervals preserved.')

order=load(OLD/'analysis/order_sensitivity.json')['results']
fig,axs=plt.subplots(1,2,figsize=(10.5,4.5));labels=['Random','Age ascending','Age descending','Baseline risk\ndescending']
for j,(arm,col,label) in enumerate([('rf_four',C[1],'RF: four features'),('lr_eb',C[0],'LR: single EB rate'),('lr_four',C[2],'LR: four features')]):
 rows=[next(r for r in rs if r['model']==arm) for rs in order.values()];x=np.arange(4)+(j-1)*.14
 v=np.array([r['delta_auroc'] for r in rows]);ci=np.array([r['conditional_ci95'] for r in rows]);axs[0].errorbar(x,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='o',color=col,capsize=3,label=label)
 v=np.array([r['free_capture_10']*100 for r in rows]);rng=np.array([r['capture_seed_range'] for r in rows])*100;axs[1].errorbar(x,v,yerr=[v-rng[:,0],rng[:,1]-v],fmt='o',color=col,capsize=3,label=label)
axs[0].axhline(0,color='gray',ls='--',lw=.8);axs[0].set_ylabel('AUROC gain versus own baseline');axs[0].set_title('A  Conditional 95% intervals',loc='left',fontsize=11)
axs[1].set_ylabel('TST-positive contacts captured (%)');axs[1].set_title('B  Capture at 10% test budget',loc='left',fontsize=11)
for ax in axs:ax.set_xticks(range(4),labels,fontsize=8.5);ax.set_xlabel('Assumed household order')
axs[1].legend(frameon=False,fontsize=8,loc='lower left');fig.text(.02,.015,'Earlier-result acquisition is uncharged. Capture whiskers show 20-seed ranges, not confidence intervals.',fontsize=8.5)
fig.tight_layout(rect=[0,.07,1,1]);save(fig,'Figure_4_order_sensitivity',[OLD/'analysis/order_sensitivity.json'],'All four accepted scenarios; archived 2000-resample conditional intervals preserved.')

cal=load(E/'homeacf_dpr_nested_calibration_v1.json')
fig,axs=plt.subplots(1,3,figsize=(11,3.7),gridspec_kw={'width_ratios':[1,1,.95]})
for ax,kind,title in zip(axs,['raw','nested_calibrated'],['A  Before calibration','B  Nested calibration']):
 for arm,c,label in [('exposure',C[0],'Baseline'),('exposure_prior',C[1],'SOP')]:
  bins=cal['rows'][0]['arms'][arm][kind]['bins'];ax.plot([b['mean_predicted'] for b in bins],[b['observed'] for b in bins],'-',color=c,label=label,linewidth=1)
  ax.scatter([b['mean_predicted'] for b in bins],[b['observed'] for b in bins],s=[max(10,min(65,b['n']/12)) for b in bins],color=c,zorder=3)
  for b in bins:
   if b['n']<30:ax.annotate('n='+str(b['n']),(b['mean_predicted'],b['observed']),xytext=(3,5 if arm=='exposure' else -11),textcoords='offset points',fontsize=7,color=c)
 ax.plot([0,1],[0,1],ls='--',color='gray',lw=.8);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed positive fraction');ax.set_title(title,loc='left',fontsize=11)
axs[0].legend(frameon=False,fontsize=8)
ax=axs[2];ax.axis('off');ax.set_title('C  Means across 20 seeds',loc='left',fontsize=11)
ax.text(.0,.86,'Measure            Baseline     SOP',family='DejaVu Sans Mono',fontsize=8.5)
for y,key,label in [(.66,'brier','Brier'),(.48,'ece_10_equal_width','ECE'),(.30,'calibration_in_large_offset','CITL'),(.12,'calibration_slope','Slope')]:
 vals=[cal['summary'][a]['nested_calibrated'][key]['mean'] for a in ['exposure','exposure_prior']]
 ax.text(.0,y,f'{label:<15} {vals[0]:>7.4f}  {vals[1]:>7.4f}',family='DejaVu Sans Mono',fontsize=8.5)
fig.text(.02,.005,'A-B: illustrative CV seed 0, fixed 0.1-wide bins; C: all 20 seeds. Lower Brier does not imply better calibration on every measure.',fontsize=8)
fig.tight_layout(rect=[0,.06,1,1]);save(fig,'Figure_5_calibration',[E/'homeacf_dpr_nested_calibration_v1.json'],'Seed-0 bin coordinates and 20-seed metrics; preserves unfavourable ECE/CITL results.')

for p in (OLD/'figures').glob('Figure_S*'):shutil.copy2(p,F/p.name)
for ext in ['png','pdf']:shutil.copy2(OLD/'figures'/f'Figure_1_workflow.{ext}',F/f'Figure_S10_cohort_flow.{ext}')
# Preserve the selected-rule figure explicitly as supplementary, without CIs.
bud=load(BASE/'data/processed/homeacf_dpr_budget_audit_v2.json')
fig,ax=plt.subplots(figsize=(6.8,3.8))
for model,c,m in [('RF',C[1],'o'),('LR',C[0],'s')]:
 x=[10,20,30];y=[100*bud['results'][f'{model}_{k}']['delta'] for k in [.1,.2,.3]];ax.plot(x,y,marker=m,color=c,label=model)
ax.axhline(0,color='gray',ls='--',lw=.8);ax.set_xticks([10,20,30]);ax.set_xlabel('Total test budget (%)');ax.set_ylabel('Capture difference (percentage points)');ax.legend(frameon=False)
ax.set_title('Historically selected two-wave rule',loc='left',fontsize=11)
fig.text(.05,.015,'k = 2, w = 7; same-cohort selection; descriptive means only. Both waves count.',fontsize=8.5);fig.tight_layout(rect=[0,.07,1,1])
save(fig,'Figure_S9_selected_charged_rule',[BASE/'data/processed/homeacf_dpr_budget_audit_v2.json'],'Same-cohort selected rule; no inferential intervals or deployment claim.')
(OUT/'checks').mkdir(exist_ok=True)
(OUT/'checks/figure_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
print('Created five main figures and S9; preserved S1-S8. Zero model fits.')

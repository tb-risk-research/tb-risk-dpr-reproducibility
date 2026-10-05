from pathlib import Path
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(r'D:\tb_risk_DPR');R=ROOT/'data/processed/dpr_major_review_v011_20261003';d=json.loads((ROOT/'data/processed/dpr_same_information_baseline_v1/result.json').read_text(encoding='utf8'))['comparisons']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,1,figsize=(7.5,6.8));colors=['#1b705c','#1b705c','#1b705c','#5269a9','#5269a9']
left=[('rf_sop_vs_base','RF: four features'),('rf_eb_vs_base','RF: single EB rate'),('rf_eb_counts_vs_base','RF: rate/count/availability'),('lr_sop_vs_base','LR: four features'),('lr_eb_vs_base','LR: single EB rate')]
right=[('rf_sop_vs_eb','RF: four minus rate'),('rf_sop_vs_eb_counts','RF: four minus rate/count/availability'),('lr_sop_vs_eb','LR: four minus rate')]
for ax,rows,title in zip(axes,[left,right],['A  Gain versus each algorithm’s baseline','B  Residual gain from four features']):
 for j,(key,label) in enumerate(rows):
  v=d[key]['delta_auroc'];lo,hi=d[key]['conditional_ci95'];c=colors[j] if ax is axes[0] else ('#5269a9' if key.startswith('lr') else '#1b705c')
  ax.errorbar(v,len(rows)-j-1,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=c)
 ax.set_yticks(range(len(rows)),[lab for _,lab in rows][::-1]);ax.axvline(0,color='gray',ls='--',lw=.8);ax.set_xlabel('Paired AUROC difference');ax.set_title(title,loc='left',fontsize=10.5);ax.set_ylim(-.6,len(rows)-.4)
axes[0].set_xlim(-.005,.103);axes[1].set_xlim(-.004,.014)
axes[1].set_xticks([-.0025,0,.005,.01],['−0.0025','0','0.005','0.010'])
fig.subplots_adjust(left=.45,right=.97,hspace=.65,bottom=.13,top=.92)
fig.text(.04,.035,'2000 fixed-prediction household resamples; non-significance is not equivalence.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(R/'figures'/f'Figure_3_same_information.{ext}',dpi=300,bbox_inches='tight')
print('Same-information figure generated from archived paired comparisons.')

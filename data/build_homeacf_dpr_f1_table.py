#!/usr/bin/env python3
import json, os, time
HERE=os.path.dirname(os.path.abspath(__file__)); PROC=os.path.join(HERE,'processed'); DOCS=os.path.join(os.path.dirname(HERE),'docs')
def load(n):
    with open(os.path.join(PROC,n),encoding='utf8') as f:return json.load(f)
a=load('sop_unified_ci_leakfree_homeacf_20261001.json'); e=load('homeacf_dpr_e1_tst_missingness_ipw_20261002_v2.json'); f=load('homeacf_dpr_f1_operating_metrics_20261002.json')
refit=load('homeacf_dpr_e1_fullrefit_v2.json')
assert refit['status']=='complete' and len(refit['replicates'])==refit['n_requested']==300
nested=load('homeacf_dpr_nested_calibration_v1.json')
assert nested['status']=='complete' and nested['completed_seeds']==20
def arm(x):return [r for r in f['rows'] if r['arm']==x]
def mean(xs):return sum(xs)/len(xs)
def op(x,q,k):return mean([r['budgets'][q][k] for r in arm(x)])
out={'name':'homeacf_dpr_f1_final_performance','date':time.strftime('%Y%m%d'),'sample':f['sample'],'primary':{'exposure_auroc':a['arm_stats']['exposure']['mean'],'sop_auroc':a['arm_stats']['exposure_prior']['mean'],'delta_auroc':a['effects']['ep_minus_exp'],'brier_delta_crossfit':a['brier']['delta']['crossfit'],'ipw_delta_auroc':e['summary']},'operating':{x:{'auprc':mean([r['auprc'] for r in arm(x)]),'budgets':{q:{'capture_rate':op(x,q,'capture_rate'),'nns':op(x,q,'nns')} for q in ['0.05','0.1','0.2','0.3']}} for x in ['exposure','exposure_prior']},'dca_exploratory':a['dca']}
out['primary']['ipw_fullrefit']=refit['summary']
out['primary'].pop('brier_delta_crossfit')
out['primary']['brier_delta_nested']={'mean':nested['brier_delta']['mean'],'ci95_unified':nested['brier_delta']['conditional_ci95']}
out['calibration_nested']=nested['summary']
out['dca_exploratory']=nested['dca']
out['primary']['ipw_fullrefit']['seed_fixed']=refit['seed_fixed']
out['limitations']=['Capture and NNS exclude acquisition costs of earlier household results.','IPW assumes adequate observation modelling and missing at random; it does not reconstruct missing household trajectories.','Conditional 20-seed CI and fixed-seed full-refit bootstrap describe different uncertainty estimates.']
with open(os.path.join(PROC,'homeacf_dpr_f1_final_performance_%s_v3.json'%time.strftime('%Y%m%d')),'w',encoding='utf8') as z:json.dump(out,z,ensure_ascii=False,indent=2)
p=out['primary']; o=out['operating']; md=['# HomeACF DPR final performance table','',f"n={f['sample']['n']}; events={f['sample']['events']}; households={f['sample']['households']}.",'','| Metric | Exposure | Exposure + SOP |','|---|---:|---:|',f"| AUROC | {p['exposure_auroc']:.3f} | {p['sop_auroc']:.3f} |",f"| AUROC difference (conditional household-bootstrap 95% CI) | — | +{p['delta_auroc']['mean']:.3f} [{p['delta_auroc']['ci95_unified'][0]:.3f}, +{p['delta_auroc']['ci95_unified'][1]:.3f}] |",f"| AUPRC | {o['exposure']['auprc']:.3f} | {o['exposure_prior']['auprc']:.3f} |",f"| Brier difference (nested calibration; conditional CI) | — | {p['brier_delta_nested']['mean']:.4f} [{p['brier_delta_nested']['ci95_unified'][0]:.4f}, {p['brier_delta_nested']['ci95_unified'][1]:.4f}] |"]
for q in ['0.05','0.1','0.2','0.3']:md.append(f"| Capture rate at {int(float(q)*100)}% screening budget | {o['exposure']['budgets'][q]['capture_rate']:.1%} | {o['exposure_prior']['budgets'][q]['capture_rate']:.1%} |")
md+=['',f"At a 10% budget, NNS was {o['exposure']['budgets']['0.1']['nns']:.2f} versus {o['exposure_prior']['budgets']['0.1']['nns']:.2f}.",f"IPW sensitivity delta: +{p['ipw_delta_auroc']['weighted_delta_mean']:.3f} [{p['ipw_delta_auroc']['weighted_delta_conditional_household_bootstrap_ci95'][0]:.3f}, +{p['ipw_delta_auroc']['weighted_delta_conditional_household_bootstrap_ci95'][1]:.3f}] (conditional household bootstrap).",'','DCA is exploratory; this table does not claim clinical deployment benefit.']
md += ['',f"IPW full-refit bootstrap: mean +{refit['summary']['mean']:.3f}, percentile 95% CI [{refit['summary']['ci95_percentile'][0]:.3f}, +{refit['summary']['ci95_percentile'][1]:.3f}]; 300/300 replicates; CV seed 0 fixed. Each replicate resamples all raw households and refits the observation model and both RF arms.",'','The weighted analysis assumes missing at random given included covariates. It reweights complete-case predictions and does not reconstruct missing household results. Capture and NNS exclude the costs of obtaining earlier household results.']
with open(os.path.join(DOCS,'homeacf_dpr_f1_final_performance.md'),'w',encoding='utf8') as z:z.write('\n'.join(md)+'\n')
print('F1 complete')

#!/usr/bin/env python3
import json,os,time
HERE=os.path.dirname(os.path.abspath(__file__)); PROC=os.path.join(HERE,'processed'); DOCS=os.path.join(os.path.dirname(HERE),'docs')
with open(os.path.join(PROC,'homeacf_dpr_nested_calibration_v1.json'),encoding='utf8') as f:d=json.load(f)
assert d['status']=='complete' and d['completed_seeds']==20
rows=[]
for key in ['0.05','0.1','0.2','0.3']:
 x=d['dca'][key]; rows.append({'threshold':float(key),'exposure_nb':x['arms_mean'][0],'sop_nb':x['arms_mean'][1],'delta_nb':x['delta_mean'],'ci95':x['conditional_ci95']})
out={'name':'homeacf_dpr_f2_dca_nested','date':time.strftime('%Y%m%d'),'design':'Fully nested Platt calibration: outer test households excluded from ALL model and calibrator training; conditional household-bootstrap CI of 20-seed mean difference. Thresholds are illustrative, not observed clinical action thresholds.','rows':rows,'interpretation':'Exploratory only; prior-result acquisition costs omitted. Does not establish real-world clinical utility.','supersedes':'homeacf_dpr_f2_dca_exploratory_20261002.json'}
with open(os.path.join(PROC,'homeacf_dpr_f2_dca_nested_v1.json'),'w',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
md=['# HomeACF DPR F2: exploratory DCA after nested calibration','',out['design'],'','| Threshold | Exposure net benefit | Exposure + SOP net benefit | Difference, conditional 95% CI |','|---|---:|---:|---|']
for r in rows:md.append(f"| {r['threshold']:.0%} | {r['exposure_nb']:.4f} | {r['sop_nb']:.4f} | {r['delta_nb']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] |")
md+=['','These results replace earlier non-nested calibration estimates. Intervals fix all fitted models and calibrators; acquisition costs are omitted. Simulated order and illustrative thresholds do not establish clinical deployment benefit.']
with open(os.path.join(DOCS,'homeacf_dpr_f2_dca_exploratory.md'),'w',encoding='utf8') as f:f.write('\n'.join(md)+'\n')
print('F2 complete')

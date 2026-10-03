"""Rebuild diagnostics from the complete synthetic journal, without editing manuscripts."""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/dpr_d1_strengthened_v1'
meta=json.loads((OUT/'metadata.json').read_text(encoding='utf-8'))
rows=[json.loads(line) for line in (OUT/'datasets.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
labels=['no detectable gain','order-specific signal present','household-structure-consistent','mixed/indeterminate']
assert len(rows)==3708 and len({(r['config']['config_id'],r['rep'],r['kind']) for r in rows})==3708
assert len({r['data_seed'] for r in rows})==3708
assert hashlib.sha256((ROOT/'experiments/dpr/run_d1_strengthened_v1.py').read_bytes()).hexdigest()==meta['script_sha256']
assert hashlib.sha256((ROOT/'docs/dpr_d1_strengthened_protocol_v1.md').read_bytes()).hexdigest()==meta['protocol_sha256']
def classify(r,margin):
    if r['contrasts']['gain']['conditional_ci95'][0]<=0:return labels[0]
    if min(r['holm_adjusted_p'].values())<=.05:return labels[1]
    if all(lo>-margin and hi<margin for lo,hi in r['equivalence_ci96_667'].values()):return labels[2]
    return labels[3]
for r in rows:
    assert not r['pilot']
    assert r['config']==meta['configurations'][r['config']['config_id']]
    assert r['sklearn_auc_gate_max_error']<1e-12
    assert all(np.isfinite(v['mean']) and np.isfinite(v['conditional_ci95']).all() for v in r['contrasts'].values())
    assert classify(r,.01)==r['diagnostic_output']
    for m in [.005,.02]:assert classify(r,m)==r['margin_sensitivity'][str(m)]
lr=[r for r in rows if r['kind']=='lr'];rf=[r for r in rows if r['kind']=='rf']
assert len(lr)==3700 and collections.Counter(r['config']['config_id'] for r in rf)=={0:2,2:2,6:2,8:2}
cells=[]
for c in meta['configurations']:
    rr=[r for r in lr if r['config']==c]
    assert len(rr)==100 and {r['rep'] for r in rr}==set(range(100))
    outputs={}
    for label in labels:
        k=sum(r['diagnostic_output']==label for r in rr)
        outputs[label]={'count':k,'proportion':k/100,'monte_carlo_exact_ci95':[0 if k==0 else float(beta.ppf(.025,k,101-k)),1 if k==100 else float(beta.ppf(.975,k+1,100-k))]}
    cells.append({'config':c,'n_datasets':100,'outputs':outputs,'mean_gain':float(np.mean([r['contrasts']['gain']['mean'] for r in rr])),
      'mean_achieved_pi':float(np.mean([r['achieved_pi'] for r in rr])),
      'margin_sensitivity':{str(m):dict(collections.Counter(classify(r,m) for r in rr)) for m in [.005,.02]}})
result={'status':'complete','design':meta,'summaries_lr':cells,'rf_records':rf,
  'limits':['Predictive contrasts are not causal attribution.','Fixed-prediction conditional intervals.','Only trigger-style time mechanisms; one-at-a-time boundary variants.','RF checks limited to eight datasets.','Retrospective rule; not preregistered.']}
if (OUT/'summary.json').exists():
    old=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    for a,b in zip(old['summaries_lr'],cells):
        assert a['config']==b['config'] and a['outputs']==b['outputs'] and abs(a['mean_gain']-b['mean_gain'])<1e-12
(OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
(OUT/'reconstruction_audit.json').write_text(json.dumps({'status':'passed','records':len(rows),'cells':len(cells),'all_labels_recomputed':True,'script_and_protocol_hashes_match':True},indent=2),encoding='utf-8')
print('Reconstructed all 3708 synthetic records and 37 cells; no manuscript modified.')

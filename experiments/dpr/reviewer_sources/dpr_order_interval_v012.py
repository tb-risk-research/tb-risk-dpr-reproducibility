"""Recalculate conditional order intervals using saved predictions, without fitting models."""
from pathlib import Path
import os,sys,json,hashlib,csv,time,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=Path(r'D:\tb_risk_DPR'))
parser.add_argument('--input',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
root=args.project.resolve();src=args.input or root/'data/processed/dpr_major_review_v011_20261003'
dst=args.output or root/'data/processed/dpr_reviewer_followup_v012_20261004'
sys.path[:0]=[str(root/'data'),str(root/'experiments/dpr')]
import run_same_information_baseline_v1 as si
from run_d1_strengthened_v1 import bootstrap_auc
old=json.loads((src/'analysis/order_sensitivity.json').read_text(encoding='utf8'))
_,y,g,_,_=si.load();_,gid=np.unique(g,return_inverse=True);hh=gid.max()+1
draws=np.random.default_rng(91303).multinomial(hh,np.full(hh,1/hh),size=2000)
result={};rows=[];start=time.time();hashes={}
names=['rf_base','rf_four','lr_eb','lr_base','lr_four']
for sc,records in old['results'].items():
 boot=[]
 for seed in range(20):
  path=si.OUT/f'seed_{seed:02d}.npz' if sc=='random' else src/'analysis'/f'{sc}_{seed:02d}.npz'
  hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
  z=np.load(path,allow_pickle=True);scores=z['scores'][[0,1,6,4,5]] if sc=='random' else z['scores']
  assert scores.shape==(5,len(y))
  boot.append(np.vstack([bootstrap_auc(scores.T,y,gid,draws[i:i+250]) for i in range(0,2000,250)]))
 b=np.mean(boot,axis=0);out=[]
 for i,row in enumerate(records):
  assert row['model']==names[i]
  baseline=0 if row['model'].startswith('rf') else 3
  contrast=b[:,i]-b[:,baseline]
  assert np.allclose(np.quantile(contrast[:1000],[.025,.975]),row['conditional_ci95'],atol=1e-12),('Original interval cannot be reproduced',sc,row['model'])
  fresh=dict(row);fresh['conditional_ci95']=np.quantile(contrast,[.025,.975]).tolist();out.append(fresh);rows.append(fresh)
 result[sc]=out
 print(sc,'complete; saved-model point estimates unchanged',flush=True)
analysis=dst/'analysis';analysis.mkdir(parents=True,exist_ok=True)
new=dict(old);new.update({'results':result,'interval':'2000 paired household resamples of fixed OOF predictions; seed 91303; same draws across orders/seeds/arms',
 'resamples':2000,'resampling_seed':91303,'revision':'v0.12 interval-count harmonization; no model refitting',
 'original_first_1000_intervals_reproduced':True,'input_prediction_sha256':hashes,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
(analysis/'order_sensitivity.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf8')
with (analysis/'order_sensitivity.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Complete:',round(time.time()-start,1),'seconds; no models trained',flush=True)

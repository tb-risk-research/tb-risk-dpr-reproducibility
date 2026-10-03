"""Frozen D1 grid; tie-safe conditional bootstrap, resumable dataset records."""
import os,sys,json,time,hashlib,platform,traceback,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'data'))
_saved_argv=sys.argv[:];sys.argv=['known_truth','full']
import run_known_truth_simulation as old
sys.argv=_saved_argv
OUTDIR=ROOT/'data/processed/dpr_d1_strengthened_v1';OUTDIR.mkdir(exist_ok=True)
N_BOOT=500;N_CV=3;REPS=100
ARMS=['exp','ep','fut','past1','fut1','past2','fut2']
CONTRASTS={'gain':('ep','exp'),'future_gain':('fut','exp'),
 'order_k1':('past1','fut1'),'order_k2':('past2','fut2'),'order_full':('ep','fut')}
ORDER_NAMES=['order_k1','order_k2','order_full']
_INTERCEPTS={}

def configs():
 out=[]
 for sigma in [0.,.8,1.6]:
  for gamma in [0.,1.,2.]:
   out.append(dict(sigma=sigma,gamma=gamma,pi=.13,hh='large',order='random',delay=0,error=0.,missing=0.,family='primary'))
 variants=[('pi',.05),('pi',.30),('hh','small'),('order','age_sorted'),('delay',1),('error',.10),('missing',.10)]
 for sigma,gamma in [(0.,0.),(1.6,0.),(0.,2.),(1.6,2.)]:
  for variable,value in variants:
   c=dict(sigma=sigma,gamma=gamma,pi=.13,hh='large',order='random',delay=0,error=0.,missing=0.,family='boundary')
   c[variable]=value;c['variant']=variable+'='+str(value);out.append(c)
 for i,c in enumerate(out):c['config_id']=i
 return out

def intercept(c):
 key=json.dumps({k:c[k] for k in ['sigma','gamma','pi','hh','order']},sort_keys=True)
 if key not in _INTERCEPTS:
  seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
  base=old._draw_base(np.random.RandomState(seed),c['hh'],4000,c['order'])
  lo,hi=-9.,5.
  for _ in range(20):
   mid=(lo+hi)/2
   if old._gen_y(base,mid,c['sigma'],c['gamma']).mean()>c['pi']:hi=mid
   else:lo=mid
  _INTERCEPTS[key]=(lo+hi)/2
 return _INTERCEPTS[key]

def blocks(base,y,c,rng):
 observed=rng.random(len(y))>=c['missing']
 # Flip mask is drawn once; independent errors only in auxiliary revealed results.
 noisy=y.copy();flip=rng.random(len(y))<c['error'];noisy[flip]=1-noisy[flip]
 result={a:np.zeros((len(y),4),float) for a in ARMS[1:]}
 for seq in base['seq']:
  for t,target in enumerate(seq):
   past=seq[:max(0,t-c['delay'])];future=seq[t+1+c['delay']:]
   past=past[observed[past]];future=future[observed[future]]
   for a,available in [('ep',past),('fut',future),('past1',past[-1:]),('fut1',future[:1]),('past2',past[-2:]),('fut2',future[:2])]:
    assert target not in available
    n=len(available);positive=int(noisy[available].sum())
    result[a][target]=[n,positive,positive/n if n else np.nan,int(n>0)]
 data=base['cov'].copy()
 for a,mat in result.items():
  for j,col in enumerate(old.ARM_COLS[a][-4:]):data[col]=mat[:,j]
 return data

def bootstrap_auc(scores,y,groups,multiplicities):
 """Each column uses fixed original score order with exact tie-group weights."""
 out=np.empty((len(multiplicities),scores.shape[1]))
 for j in range(scores.shape[1]):
  order=np.argsort(scores[:,j],kind='mergesort');ss=scores[order,j]
  start=np.r_[0,np.flatnonzero(np.diff(ss)!=0)+1]
  weights=multiplicities[:,groups[order]].astype(float)
  pos=np.add.reduceat(weights*y[order],start,axis=1)
  neg=np.add.reduceat(weights*(1-y[order]),start,axis=1)
  den=pos.sum(axis=1)*neg.sum(axis=1)
  if np.any(den==0):raise ValueError('Single-class bootstrap: do not silently discard')
  out[:,j]=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1)/den
 return out

def holm(p):
 order=np.argsort(p);adjusted=np.empty(len(p));last=0.
 for rank,ix in enumerate(order):
  last=max(last,min(1.,p[ix]*(len(p)-rank)));adjusted[ix]=last
 return adjusted

def verdict(gain_ci,adjusted_p,order_ci,margin):
 if gain_ci[0]<=0:return 'no detectable gain'
 if np.any(adjusted_p<=.05):return 'order-specific signal present'
 if np.all((order_ci[:,0]>-margin)&(order_ci[:,1]<margin)):return 'household-structure-consistent'
 return 'mixed/indeterminate'

def run_one(task):
 c,rep,kind,pilot=task;start=time.time()
 data_seed=20300000+1000*c['config_id']+rep+(700000 if pilot else 0)+(900000 if kind=='rf' else 0)
 rng=np.random.RandomState(data_seed);base=old._draw_base(rng,c['hh'],2725,c['order'])
 b0=intercept(c);y=old._gen_y(base,b0,c['sigma'],c['gamma']);g=base['hh']
 data=blocks(base,y,c,np.random.default_rng(data_seed+1))
 scores=[];point=np.zeros((len(ARMS),N_CV))
 for arm,a in enumerate(ARMS):
  for seed in range(N_CV):
   folds=old._group_cv_indices(g,y,5,seed)
   for tr,te in folds:assert not set(g[tr])&set(g[te])
   p=old._fit_oof(old.ARM_COLS[a],folds,y,data,seed,kind)
   scores.append(p);point[arm,seed]=roc_auc_score(y,p)
 scores=np.column_stack(scores)
 rng_b=np.random.default_rng(data_seed+2)
 draws=rng_b.multinomial(int(g.max()+1),np.full(int(g.max()+1),1/(g.max()+1)),size=N_BOOT)
 aucs=bootstrap_auc(scores,y,g,draws)
 # Weighted, tie-safe formula checked against sklearn on fresh resamples.
 max_error=0.
 for b in [0,1]:
  for j in [0,N_CV,2*N_CV]:
   gold=roc_auc_score(y,scores[:,j],sample_weight=draws[b,g]);max_error=max(max_error,abs(gold-aucs[b,j]))
 assert max_error<1e-12
 aucs=aucs.reshape(N_BOOT,len(ARMS),N_CV);estimates={};delta_boot={}
 for name,(a,b) in CONTRASTS.items():
  ia,ib=ARMS.index(a),ARMS.index(b);d=float((point[ia]-point[ib]).mean())
  bd=(aucs[:,ia,:]-aucs[:,ib,:]).mean(axis=1);delta_boot[name]=bd
  estimates[name]={'mean':d,'conditional_ci95':np.quantile(bd,[.025,.975]).tolist()}
 pvalues=[];equiv_ci=[]
 for name in ORDER_NAMES:
  d=estimates[name]['mean'];bd=delta_boot[name]
  pvalues.append((1+int(np.sum(bd-d>=d)))/(N_BOOT+1))
  equiv_ci.append(np.quantile(bd,[.05/3,1-.05/3]))
 adjusted=holm(np.asarray(pvalues));equiv_ci=np.asarray(equiv_ci)
 cat=verdict(estimates['gain']['conditional_ci95'],adjusted,equiv_ci,.01)
 legacy='none' if estimates['gain']['conditional_ci95'][0]<=0 else ('order_specific' if any(estimates[n]['conditional_ci95'][0]>0 for n in ORDER_NAMES) else 'clustering_consistent')
 return {'config':c,'rep':rep,'kind':kind,'pilot':pilot,'data_seed':data_seed,
  'n':len(y),'households':int(g.max()+1),'events':int(y.sum()),'achieved_pi':float(y.mean()),'beta0':b0,
  'arm_mean_auc':{a:float(point[i].mean()) for i,a in enumerate(ARMS)},'contrasts':estimates,
  'one_sided_centered_bootstrap_p':dict(zip(ORDER_NAMES,pvalues)),
  'holm_adjusted_p':dict(zip(ORDER_NAMES,adjusted.tolist())),
  'equivalence_ci96_667':dict(zip(ORDER_NAMES,equiv_ci.tolist())),
  'diagnostic_output':cat,'margin_sensitivity':{str(m):verdict(estimates['gain']['conditional_ci95'],adjusted,equiv_ci,m) for m in [.005,.02]},
  'historical_rule_on_corrected_scores':legacy,'sklearn_auc_gate_max_error':max_error,
  'runtime_seconds':time.time()-start}

def atomic_json(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def preflight():
 # Exact ties, unequal household multiplicities, and expanded-copy equivalence.
 y=np.array([0,1,0,1,0,1]);g=np.array([0,0,1,1,2,2])
 scores=np.array([[.2,.5],[.2,.5],[.5,.5],[.8,.5],[.8,.5],[.8,.5]])
 w=np.array([[1,1,1],[2,0,1],[0,2,1]])
 a=bootstrap_auc(scores,y,g,w)
 for b in range(len(w)):
  expanded=np.repeat(np.arange(len(y)),w[b,g])
  for j in range(2):assert abs(a[b,j]-roc_auc_score(y[expanded],scores[expanded,j]))<1e-12
 assert np.allclose(holm(np.array([.01,.04,.20])),[.03,.08,.20])
 cc=configs();results=[run_one((cc[i],0,'lr',True)) for i in [0,8]]
 protocol=ROOT/'docs/dpr_d1_strengthened_protocol_v1.md'
 out={'status':'preflight_passed','auc_ties_and_duplicates_gate':True,'holm_gate':True,
  'datasets':results,'median_dataset_seconds':float(np.median([r['runtime_seconds'] for r in results])),
  'estimated_lr_wall_seconds_4_workers':sum(r['runtime_seconds'] for r in results)/2*3700/4,
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest()}
 atomic_json(OUTDIR/'preflight.json',out);print(json.dumps(out,indent=2),flush=True)

def main(workers):
 cc=configs();tasks=[(c,r,'lr',False) for c in cc for r in range(REPS)]
 corners=[c for c in cc if c['family']=='primary' and c['sigma'] in [0.,1.6] and c['gamma'] in [0.,2.]]
 tasks += [(c,r,'rf',False) for c in corners for r in range(2)]
 journal=OUTDIR/'datasets.jsonl';records=[]
 if journal.exists():
  for line in journal.read_text(encoding='utf-8').splitlines():records.append(json.loads(line))
 keys={(r['config']['config_id'],r['rep'],r['kind']) for r in records}
 assert len(keys)==len(records)
 todo=[t for t in tasks if (t[0]['config_id'],t[1],t[2]) not in keys]
 start=time.time();baseline=len(records)
 metadata={'design':'37 LR configurations x 100 independent datasets + 8 limited RF checks',
  'configurations':cc,'target':len(tasks),'cv_seeds':[0,1,2],'bootstrap_replicates':N_BOOT,
  'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'protocol_sha256':hashlib.sha256((ROOT/'docs/dpr_d1_strengthened_protocol_v1.md').read_bytes()).hexdigest()}
 def status(state,error=None):
  elapsed=time.time()-start;finished=len(records)-baseline
  atomic_json(OUTDIR/'progress.json',{'status':state,'completed':len(records),'target':len(tasks),
   'completed_lr':sum(r['kind']=='lr' for r in records),'completed_rf':sum(r['kind']=='rf' for r in records),
   'elapsed_session_seconds':elapsed,'estimated_remaining_seconds':elapsed/finished*(len(tasks)-len(records)) if finished else None,
   'workers':workers,'pid':os.getpid(),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S'),'error':error})
 atomic_json(OUTDIR/'metadata.json',metadata);status('running')
 try:
  with ProcessPoolExecutor(max_workers=workers) as pool, journal.open('a',encoding='utf-8') as f:
   pending={};iterator=iter(todo)
   for _ in range(workers*2):
    t=next(iterator,None)
    if t is not None:pending[pool.submit(run_one,t)]=t
   while pending:
    done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
    if not done:status('running');continue
    for future in done:
     pending.pop(future);r=future.result();f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();records.append(r)
     if len(records)%25==0:print('completed',len(records),'/',len(tasks),flush=True)
     status('running');t=next(iterator,None)
     if t is not None:pending[pool.submit(run_one,t)]=t
  assert len(records)==len(tasks)
  summaries=[]
  for c in cc:
   rr=[r for r in records if r['config']['config_id']==c['config_id'] and r['kind']=='lr'];assert len(rr)==100
   proportions={}
   for category in ['no detectable gain','order-specific signal present','household-structure-consistent','mixed/indeterminate']:
    k=sum(r['diagnostic_output']==category for r in rr)
    proportions[category]={'count':k,'proportion':k/100,
      'monte_carlo_exact_ci95':[0. if k==0 else float(beta.ppf(.025,k,101-k)),1. if k==100 else float(beta.ppf(.975,k+1,100-k))]}
   summaries.append({'config':c,'n_datasets':100,'outputs':proportions,
     'mean_gain':float(np.mean([r['contrasts']['gain']['mean'] for r in rr])),
     'mean_achieved_pi':float(np.mean([r['achieved_pi'] for r in rr]))})
  atomic_json(OUTDIR/'summary.json',{'status':'complete','design':metadata,'summaries_lr':summaries,
   'rf_records':[r for r in records if r['kind']=='rf'],
   'limits':['Diagnostics are not causal classification.','Conditional dataset intervals omit model refits.',
    'Boundary variants are one-at-a-time, not a factorial grid.','Only trigger-style time mechanism evaluated.',
    'RF checks limited to two datasets per corner.','Analysis rule was frozen after seeing historical results, not preregistered.']})
  status('complete');print('ALL COMPLETE',flush=True)
 except Exception:
  error=traceback.format_exc();status('failed',error);raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['preflight','full'],default='preflight');parser.add_argument('--workers',type=int,default=4)
 args=parser.parse_args()
 if args.mode=='preflight':preflight()
 else:main(args.workers)

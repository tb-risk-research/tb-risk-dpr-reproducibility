"""Portable, isolated entry point for the frozen DPR analyses (Python 3.12)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/dpr'
EVIDENCE = HERE / 'evidence'
SHA = '748694c26c80ed7dfd275c8a7d57d226b432ac133c2c336113085db5d17cbf8b'
SOURCE = 'https://raw.githubusercontent.com/petermacp/tstsa/9e26014096bedaa2cd552c44d91459796d1df0a7/data/tstsa.rda'
TASKS = {
 'primary': ('data/run_sop_unified_ci_leakfree.py', [], 'sop_unified_ci_leakfree_homeacf_*.json', 'sop_unified_ci_leakfree_homeacf_20261001.json'),
 'controls': ('data/run_samecohort_controls.py', ['sop_unified_ci_leakfree_homeacf_20261001.json'], 'samecohort_controls_homeacf_*.json', 'samecohort_controls_homeacf_20261001.json'),
 'fullrefit': ('data/run_sop_fullrefit_bootstrap.py', [], 'homeacf_dpr_a1_fullrefit_*.json', 'homeacf_dpr_a1_fullrefit_20261001.json'),
 'availability': ('experiments/dpr/run_availability_sensitivity_v1.py', [], None, None),
 'missingness': ('data/run_homeacf_tst_missingness_ipw.py', [], 'homeacf_dpr_e1_tst_missingness_ipw_*_v2.json', 'homeacf_dpr_e1_tst_missingness_ipw_20261002_v2.json'),
 'ipw-fullrefit': ('data/run_homeacf_e1_fullrefit.py', [], None, None),
 'operating': ('data/run_homeacf_f1_operating_metrics.py', [], 'homeacf_dpr_f1_operating_metrics_*.json', 'homeacf_dpr_f1_operating_metrics_20261002.json'),
 'calibration': ('experiments/dpr/run_nested_calibration_v1.py', ['sop_unified_ci_leakfree_homeacf_20261001.json'], None, None),
 'same-information': ('experiments/dpr/run_same_information_baseline_v1.py', ['homeacf_dpr_nested_calibration_v1.npz'], None, None),
 'coding-sensitivity': ('experiments/dpr/run_feature_audit_sensitivity_v1.py', [], None, None),
 'simulation': ('experiments/dpr/run_d1_strengthened_v1.py', [], None, None),
 'simulation-report': ('experiments/dpr/summarize_simulation.py', ['dpr_d1_strengthened_v1/datasets.jsonl'], None, None),
 'performance': ('data/build_homeacf_dpr_f1_table.py', ['sop_unified_ci_leakfree_homeacf_20261001.json','homeacf_dpr_e1_tst_missingness_ipw_20261002_v2.json','homeacf_dpr_e1_fullrefit_v2.json','homeacf_dpr_f1_operating_metrics_20261002.json','homeacf_dpr_nested_calibration_v1.json'], None, None),
 'dca': ('data/build_homeacf_dpr_f2_dca.py', ['homeacf_dpr_nested_calibration_v1.json'], None, None),
 'cohort': ('experiments/dpr/cohort_report.py', [], None, None),
 'figures': ('experiments/dpr/plot_figures.py', ['sop_unified_ci_leakfree_homeacf_20261001.json','samecohort_controls_homeacf_20261001.json','homeacf_dpr_budget_audit_v2.json','dpr_same_information_baseline_v1/result.json','dpr_d1_strengthened_v1/summary.json','homeacf_dpr_nested_calibration_v1.json'], None, None),
 'budget': ('experiments/dpr/run_budget_audit_v2.py', [], None, None),
}
ORDER = ['cohort','primary','controls','fullrefit','availability','missingness','ipw-fullrefit','operating','calibration','same-information','coding-sensitivity','budget','simulation','simulation-report','performance','dca','figures']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def inventory():
    return json.loads((HERE/'release_manifest.json').read_text(encoding='utf-8'))

def audit():
    manifest = inventory()
    failures = []
    for row in manifest['files']:
        p = ROOT / row['path']
        if not p.is_file() or sha(p) != row['sha256']:
            failures.append(row['path'])
    raw = ROOT/'data/raw/homeacf_tstsa.rda'
    result = {'status':'passed' if not failures else 'failed','files_checked':len(manifest['files']), 'failures':failures,
       'raw_data_present':raw.is_file(), 'raw_data_verified':raw.is_file() and sha(raw)==SHA,
       'data_recovery':'fetch-data downloads pinned original file and fails on checksum mismatch',
       'full_experiment_rerun':False}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures:
        raise RuntimeError('Release contents changed; refresh the manifest before packaging.')
    return result

def fetch():
    p = ROOT/'data/raw/homeacf_tstsa.rda'
    if p.exists():
        if sha(p) != SHA:
            raise RuntimeError('Existing raw file differs from the frozen source; refusing to overwrite it.')
        print('Verified existing HomeACF data.'); return
    p.parent.mkdir(parents=True,exist_ok=True)
    request = urllib.request.Request(SOURCE,headers={'User-Agent':'tb-risk-dpr-reproduction'})
    with urllib.request.urlopen(request,timeout=60) as response:
        data=response.read()
    if hashlib.sha256(data).hexdigest()!=SHA:
        raise RuntimeError('Upstream checksum mismatch; no file saved.')
    tmp=p.with_suffix('.download.tmp');tmp.write_bytes(data);tmp.replace(p)
    print('Downloaded and verified pinned HomeACF data.')

def prepare(run_dir, references=False):
    run_dir=Path(run_dir).resolve()
    forbidden=[ROOT.resolve(),(ROOT/'data/raw').resolve(),(ROOT/'submission_private').resolve()]
    if any(run_dir==p or run_dir in p.parents for p in forbidden):
        raise ValueError('Run directory must not be the source project or an ancestor of protected directories.')
    if any(p in run_dir.parents for p in forbidden[1:]):
        raise ValueError('Run directory cannot be inside raw data or private submission directories.')
    snapshot=run_dir/'tb_risk'
    manifest=inventory();fingerprint=sha(HERE/'release_manifest.json')
    marker=run_dir/'workspace.json'
    if marker.exists():
        meta=json.loads(marker.read_text(encoding='utf-8'))
        if meta['source_manifest_sha256']!=fingerprint or meta['reference_outputs_seeded']!=references:
            raise RuntimeError('Run workspace belongs to a different source version or reference mode; use a new directory.')
        for row in manifest['files']:
            p=snapshot/row['path']
            if not p.exists() or sha(p)!=row['sha256']:
                raise RuntimeError(f'Run source changed: {row["path"]}; use a new directory.')
        if sha(snapshot/'data/raw/homeacf_tstsa.rda')!=SHA:
            raise RuntimeError('Run raw-data checksum changed; refusing to reuse the workspace.')
        return snapshot
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError('Unrecognised nonempty run directory; refusing to merge files.')
    raw=ROOT/'data/raw/homeacf_tstsa.rda'
    if not raw.exists() or sha(raw)!=SHA:
        raise RuntimeError('Run fetch-data first: frozen HomeACF data is absent or has a different checksum.')
    for row in manifest['files']:
        source=ROOT/row['path']; dest=snapshot/row['path']
        if sha(source)!=row['sha256']:raise RuntimeError(f'Source changed: {source}')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    (snapshot/'data/raw').mkdir(parents=True,exist_ok=True)
    shutil.copy2(raw,snapshot/'data/raw/homeacf_tstsa.rda')
    p=snapshot/'data/processed';p.mkdir(parents=True,exist_ok=True)
    # Historical summaries are regression-test comparators, never fitted scores.
    for n in ['sop_unified_ci_homeacf_20260928.json','sop_robustness_battery_homeacf_20260927.json']:
        shutil.copy2(EVIDENCE/'historical_gates'/n,p/n)
    if references:
        for source in (EVIDENCE/'results').rglob('*'):
            if source.is_file():
                dest=p/source.relative_to(EVIDENCE/'results');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    write(marker,{'source_manifest_sha256':fingerprint,'reference_outputs_seeded':references,
       'raw_sha256':SHA,'scope':'isolated source and raw-data copy; no participant prediction cache copied',
       'created':time.strftime('%Y-%m-%d %H:%M:%S'),'python':sys.version})
    return snapshot

def child(snapshot, script, arguments, label):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUTF8']='1'
    env['PYTHONPATH']=os.pathsep.join([str(snapshot.parent),str(snapshot/'data'),str(snapshot/'experiments/dpr')])
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
    log_dir=snapshot.parent/'logs';log_dir.mkdir(exist_ok=True)
    log=log_dir/(re.sub(r'[^a-zA-Z0-9_.-]','_',label)+'.log')
    start=time.time()
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([sys.executable,'-B',str(snapshot/script),*arguments],cwd=snapshot,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            print(line,end='',flush=True);stream.write(line);stream.flush()
        code=process.wait()
    state={'task':label,'exit_code':code,'seconds':time.time()-start,'log':str(log)}
    write(snapshot.parent/'last_task.json',state)
    if code:raise RuntimeError(f'{label} failed (exit {code}); see {log}')
    return state

def run_task(snapshot, task, args):
    script, dependencies, pattern, canonical=TASKS[task]
    p=snapshot/'data/processed'
    if task=='controls':
        source=p/'sop_unified_ci_leakfree_homeacf_20261001.json'
        if source.exists():shutil.copy2(source,p/'sop_unified_ci_leakfree_homeacf_20260929.json')
    missing=[n for n in dependencies if not (p/n).is_file()]
    if missing:raise RuntimeError(f'{task} requires earlier regenerated outputs: {missing}; follow run-all order.')
    smoke='smoke' in args
    state=child(snapshot,script,args,task+('_smoke' if smoke else ''))
    if pattern and not smoke:
        candidates=sorted(p.glob(pattern),key=lambda q:q.stat().st_mtime_ns)
        if not candidates:raise RuntimeError(f'{task} did not produce its declared output')
        source=candidates[-1]
        result=json.loads(source.read_text(encoding='utf-8'))
        if result.get('smoke'):raise RuntimeError('A smoke result cannot satisfy a full-run dependency.')
        validate_result(result)
        if canonical and source.name!=canonical:shutil.copy2(source,p/canonical)
        state['generated_file']=str(source.relative_to(snapshot));state['canonical_alias']=canonical
    write(snapshot.parent/'tasks'/f'{task}.json',state)
    return state

def validate_result(result):
    if result.get('status') in ['failed','running','issues_found']:
        raise RuntimeError('Result is incomplete or failed; refusing to promote it to a full-run dependency.')
    if result.get('failures'):
        raise RuntimeError('Result contains failed replicates; refusing to promote it.')
    if 'n_effective' in result and result['n_effective']!=result.get('n_requested'):
        raise RuntimeError('Not all requested replicates completed.')
    def gates(obj):
        if isinstance(obj,dict):
            if obj.get('passed') is False:raise RuntimeError('An archived regression gate failed.')
            for value in obj.values():gates(value)
        elif isinstance(obj,list):
            for value in obj:gates(value)
    design=result.get('design',{})
    if isinstance(design,dict):
        gates(design.get('anchor_gate',{}));gates(design.get('gates',{}))

def pack(output):
    audit(); target=Path(output).resolve()
    if target.exists():raise FileExistsError('Archive already exists; choose a new filename.')
    target.parent.mkdir(parents=True,exist_ok=True)
    entries=[r['path'] for r in inventory()['files']]+['experiments/dpr/release_manifest.json']
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for relative in sorted(entries):z.write(ROOT/relative,'tb_risk/'+relative)
    print(json.dumps({'archive':str(target),'sha256':sha(target),'files':len(entries),'raw_data':'obtained through pinned fetch-data command; not rehosted in archive'},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('audit');sub.add_parser('fetch-data')
    for name in ['check','validate','run-all','figures','tables']:
        p=sub.add_parser(name);p.add_argument('--run-dir',required=True)
    p=sub.add_parser('run');p.add_argument('task',choices=TASKS);p.add_argument('--run-dir',required=True)
    p=sub.add_parser('pack');p.add_argument('--output',required=True)
    args,extra=parser.parse_known_args()
    if extra and args.command!='run':parser.error('Unexpected arguments: '+str(extra))
    if args.command=='audit':audit();return
    if args.command=='fetch-data':fetch();return
    if args.command=='pack':pack(args.output);return
    reference=args.command in ['figures','tables']
    snapshot=prepare(args.run_dir,references=reference)
    if args.command in ['check','validate']:
        child(snapshot,'experiments/dpr/validate_reproduction.py',['--scope',args.command],args.command)
    elif args.command=='run':
        if extra and extra[0]=='--':extra=extra[1:]
        run_task(snapshot,args.task,extra)
    elif args.command=='run-all':
        for task in ORDER:run_task(snapshot,task,['--mode','full'] if task=='simulation' else [])
    elif args.command=='figures':run_task(snapshot,'figures',[])
    elif args.command=='tables':child(snapshot,'experiments/dpr/export_tables.py',[], 'tables')

if __name__=='__main__':main()

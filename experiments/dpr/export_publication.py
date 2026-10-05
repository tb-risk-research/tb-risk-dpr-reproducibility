"""Export accepted v0.17 display snapshots. This does NOT fit models or regenerate plots."""
from pathlib import Path
import argparse,json,csv,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/dpr/evidence/publication_v017'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['all','tables','figures'],default='all');p.add_argument('--output',type=Path,default=ROOT/'data/processed/publication_v017');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);report={'mode':'accepted snapshots; zero model fits','version':'0.17','tables':[],'figures':[]}
 if a.kind in ['all','tables']:
  for group,items in json.loads((SOURCE/'tables.json').read_text(encoding='utf-8')).items():
   for i,item in enumerate(items,1):
    target=a.output/(group+'_table_'+str(i)+'.csv')
    with target.open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(item['cells'])
    report['tables'].append({'file':target.name,'caption':item['caption'],'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
 if a.kind in ['all','figures']:
  for entry in json.loads((SOURCE/'figures.json').read_text(encoding='utf-8')):
   src=SOURCE/entry['path'];assert hashlib.sha256(src.read_bytes()).hexdigest()==entry['sha256'];dst=a.output/src.name;shutil.copy2(src,dst);report['figures'].append({'file':dst.name,'sha256':entry['sha256']})
 (a.output/('export_report_'+a.kind+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'mode':report['mode'],'tables':len(report['tables']),'figure_files':len(report['figures'])}))
if __name__=='__main__':main()

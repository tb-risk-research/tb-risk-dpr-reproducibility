"""Export immutable accepted Word table cells, labelled as archived (not refitted)."""
from pathlib import Path
import csv
import json
ROOT=Path(__file__).resolve().parents[2]
tables=json.loads((ROOT/'experiments/dpr/evidence/manuscript_tables.json').read_text(encoding='utf-8'))
out=ROOT/'data/processed/tables';out.mkdir(exist_ok=True)
for group,items in tables.items():
    for i,rows in enumerate(items,1):
        with (out/f'{group}_{i}.csv').open('w',encoding='utf-8-sig',newline='') as stream:csv.writer(stream).writerows(rows)
(out/'README.md').write_text('These are the accepted v0.8 editable table cells exported from the manuscript, not newly fitted results. Recompute experiment results with run-all and compare them before changing paper tables. See experiment_registry.json for each result source.\n',encoding='utf-8')
print('Exported seven main and seven supplementary tables from accepted cell snapshots.')

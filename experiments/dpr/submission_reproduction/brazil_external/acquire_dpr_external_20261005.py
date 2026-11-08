"""Acquire pinned public inputs; run with pandas and openpyxl installed.
Does not overwrite an existing file with an unexpected SHA-256.
"""
from pathlib import Path
from io import StringIO
import os, urllib.request, hashlib, json
from datetime import datetime, timezone
import pandas as pd
ROOT=Path(os.environ.get('DPR_ROOT','D:/tb_risk_DPR'))
OUT=ROOT/'data/processed/dpr_external_validation_20261005'

def fetch_checked(path,url,expected):
    if path.exists(): blob=path.read_bytes()
    else:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as r: blob=r.read()
    assert hashlib.sha256(blob).hexdigest()==expected, 'Source hash mismatch; refusing overwrite: '+str(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): path.write_bytes(blob)
    return dict(url=url,sha256=expected,bytes=len(blob),checked_utc=datetime.now(timezone.utc).isoformat())

def main():
    (OUT/'audit').mkdir(parents=True,exist_ok=True)
    xlsx=OUT/'raw/brazil2018_aerosol_s002.xlsx'
    target=fetch_checked(xlsx,'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0206384.s002&type=supplementary',
        '7cd901f6e4a3d23f15d6234ec0ff8cadbf696875b507e0bdeddd064e5d72ca94')
    source=fetch_checked(ROOT/'data/raw/homeacf_tstsa.rda',
        'https://raw.githubusercontent.com/petermacp/tstsa/9e26014096bedaa2cd552c44d91459796d1df0a7/data/tstsa.rda',
        '748694c26c80ed7dfd275c8a7d57d226b432ac133c2c336113085db5d17cbf8b')
    data=pd.read_excel(xlsx,sheet_name='HHCs')
    assert data.shape==(230,39) and data.indexid.nunique()==48 and data.contactid.is_unique
    derived=OUT/'raw/brazil2018_hhc_derived.csv'
    if derived.exists():
        # Excel contains mixed numeric/text smear columns (e.g. 2 and '2+').
        # CSV necessarily normalizes these to strings; compare the complete
        # Excel-to-CSV round trip, while independently checking model fields.
        normalized=pd.read_csv(StringIO(data.to_csv(index=False)))
        pd.testing.assert_frame_equal(normalized,pd.read_csv(derived),check_dtype=False,check_exact=False)
        used=['indexid','contactid','HHCage','HHCgender','ageindex','genderindex','ppd1','outcome']
        pd.testing.assert_frame_equal(data[used],pd.read_csv(derived)[used],check_dtype=False,check_exact=False)
    else: data.to_csv(derived,index=False)
    target.update(article='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0206384',
        doi='10.1371/journal.pone.0206384.s002',worksheet='HHCs',outcome='ppd1 >=10 mm; baseline only',
        attribution='Acuña-Villaorduña C et al., PLOS ONE 2018;13(10):e0206384.',
        article_license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
        availability='Article states all relevant data in paper and supporting information.',
        derived_csv_sha256=hashlib.sha256(derived.read_bytes()).hexdigest())
    source.update(attribution='Peter MacPherson, tstsa package, 2019',
        package_license='MIT + file LICENSE; package-level evidence, no separate data license found',
        provenance_commit='9e26014096bedaa2cd552c44d91459796d1df0a7',
        license_url='https://github.com/petermacp/tstsa/blob/9e26014096bedaa2cd552c44d91459796d1df0a7/LICENSE.md')
    obj={'source_homeacf':source,'external_brazil2018':target,'redistribution':'No participant-level data supplied in desktop handoff or remotely published.'}
    (OUT/'audit/validated_acquisition.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.read_excel(xlsx,sheet_name='Dictionary').to_csv(OUT/'audit/brazil_dictionary.csv',index=False)
    print('PASS: pinned source and target inputs, Excel/CSV equality, baseline endpoint and dictionary.')
if __name__=='__main__': main()

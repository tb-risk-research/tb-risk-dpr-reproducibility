from pathlib import Path
import json,argparse
parser=argparse.ArgumentParser()
parser.add_argument('--artifact-dir',type=Path,default=Path(r'D:\tb_risk_DPR\data\processed\dpr_reviewer_followup_v012_20261004'))
args=parser.parse_args()
# The plotting function is isolated from Word-processing dependencies.
source=(Path(__file__).parent/'dpr_v012_finalize.py').read_text(encoding='utf8')
start=source.index('def plot():');end=source.index('def edit():',start)
namespace={'R':args.artifact_dir,'json':json}
exec(compile(source[start:end],'order_sensitivity_plot','exec'),namespace)
namespace['plot']()

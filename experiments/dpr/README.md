# DPR 独立论文仓库运行入口

这套入口复现 DPR 论文的 HomeACF 分析，保留整个 tb_risk 项目源码。测试基准为 Python 3.12.14 / Windows / CPU，无需 GPU 或 GNN。其他 GUI、Cox、SEIR 等模块仍保留，但本交接不声称已验证它们全部的可选环境或训练数据。

## 新机器安装

在解压后的 tb_risk 根目录执行（Windows）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/dpr/requirements-dpr.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py audit
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py fetch-data
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py check --run-dir data/processed/dpr_runs/quick
```

Linux 使用 python3.12、.venv/bin/python 对应替换；依赖锁来自 Windows 实测，Linux 安装及结果尚未验证。安装需网络；原始 HomeACF 文件 63,431 字节，可自动下载并按固定 SHA-256 检查。已有文件哈希不一致时拒绝覆盖。MIT 许可和原作者版权说明保留在 licenses/HomeACF；不在 Git 中另行托管参与者记录。

## 检查和完整运行

```powershell
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py validate --run-dir data/processed/dpr_runs/validation
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py run primary --run-dir data/processed/dpr_runs/full
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py run controls --run-dir data/processed/dpr_runs/full
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py run-all --run-dir data/processed/dpr_runs/full
```

check 从原始数据重新拟合种子 0 的两臂模型。validate 重新拟合主分析种子 0/19、七臂强基线种子 0、嵌套校准种子 0、预算核查、A1/E1 各首个重拟合、四个 D1 配置首个数据集。它不是全部 20 种子、300 重拟合及 3,708 模拟的全量重跑。比较仅使用汇总参考数值，不复制旧个体预测缓存用于拟合。

run-all 按完整依赖顺序运行；simulation 会自动采用 --mode full。计算量很大，具体历史耗时与工作量见 RUN_ORDER.md；无法保证不同电脑完成时间。程序前台运行，关闭终端可能中断；logs/ 和 last_task.json 可查看状态，长任务另有进度 JSON。

每个 --run-dir 生成独立 tb_risk 源码和原始数据快照。所有输出均写在该快照的 data/processed 下，不覆盖唯一归档。当前证据不会预填进 full/validation；两个历史汇总只用于主程序的回归门。变更源码后必须换运行目录。同一目录仅复用同一版本；各脚本断点支持见 RUN_ORDER.md。不要把 smoke 结果当作论文结果。

## 图表与结果

```powershell
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py figures --run-dir data/processed/dpr_runs/accepted_figures
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py tables --run-dir data/processed/dpr_runs/accepted_tables
```

这两条命令明确使用已接受的汇总证据：figures 生成七张 PDF/PNG；tables 导出 14 张已接受 Word 表的 CSV 快照，**不是重新拟合后自动生成的论文表**。全量重跑的输出须对照 experiment_registry.json 和 accepted_results 后逐项核查，不能未经审核替换文稿。

data_dictionary.json 保存本稿的结局、冻结变量定义及已知编码问题；experiment_registry.json 对应全部当前实验、参数规模、结果与论文位置。第三方来源与项目许可状态见 THIRD_PARTY_NOTICES.md。清单与打包范围见 release_manifest.json。

## 独立仓库交付

```powershell
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py pack --output data/processed/tb_risk_DPR_source.zip
```

pack 只按已审核的文件清单打包，不包含 .git、Python 环境、私人投稿材料、伦理照片、原始参与者记录、NPZ 缓存、个人运行日志或旧论文草稿。不覆写已有压缩包。本包交付独立 DPR 论文复现仓库；不以向源项目提交合并请求为默认目标。名称、归属及公开时机由用户确认；本次没有创建远程、推送、公开归档或投稿。以后确有通用改进再选择性贡献。

旧的直接写固定路径入口 run.py 和会修改旧稿的 analyze_d1_strengthened_v1.py 留在本地历史目录，不纳入交接压缩包；新复现统一使用 delivery.py 和 summarize_simulation.py。论文 v0.8 与签章伦理材料仍在 submission_private，不纳入公开代码包。需要最新稿件时使用桌面 DPR_格式修订_v0.8_20261003。


来源项目：https://github.com/tiande888/tb_risk；基础提交：3cc3c82f42dbf1e2bb8c3c4209e078a99db7c5c7。本地 .git 历史与旧来源远程保留，新仓库地址尚未决定。交付政策见 INDEPENDENT_HANDOFF.md，准确上传范围见 upload_manifest.json。旧运行目录绑定旧文件清单指纹；若需要后续运行请使用新 --run-dir，不能把旧指纹目录当作更新后的快照续跑。

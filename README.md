# DPR 论文独立复现仓库

本仓库复现拟投 **Diagnostic and Prognostic Research（DPR）** 论文中的 HomeACF 家庭接触者分析、SOP 特征比较、诊断对照、模拟与筛查预算评价。它是独立论文复现仓库，不以合并回源项目为默认目标。

## 来源和新增工作

- 源项目：[https://github.com/tiande888/tb_risk](https://github.com/tiande888/tb_risk)。
- 基础提交：[`3cc3c82f42dbf1e2bb8c3c4209e078a99db7c5c7`](https://github.com/tiande888/tb_risk/tree/3cc3c82f42dbf1e2bb8c3c4209e078a99db7c5c7)。
- 保留来源项目的模型及应用源码、软件包布局和本地 Git 历史，没有改写来源提交或修改源项目远程仓库。
- 新增 DPR 实验协议、完整运行入口、隔离输出、固定依赖、数据版本/校验、汇总证据台账、变量字典和复现检查；绘图去除本机绝对路径，模拟报告不再依赖旧稿。
- 具体来源、修改范围和文件状态见 [REPOSITORY_PROVENANCE.json](REPOSITORY_PROVENANCE.json) 和 [独立仓库交付说明](experiments/dpr/INDEPENDENT_HANDOFF.md)。当前调整只改变文档和交付清单，不改变模型、参数或已接受结果。

## 配置和运行

已验证环境为 **Python 3.12.14、Windows、CPU**。DPR 流程无需 GPU/GNN。原项目其他 GUI、Cox、SEIR 等模块保留，但不是本论文全部经过验证的功能。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/dpr/requirements-dpr.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py audit
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py fetch-data
.\.venv\Scripts\python.exe -B experiments/dpr/delivery.py check --run-dir data/processed/dpr_runs/quick
```

这套安装命令是新机器使用说明；此前验证是隔离环境离线复制已安装依赖，再构建/安装项目 wheel，不代表已验证从 PyPI 全新安装。完整实验和图表操作见 [运行说明](experiments/dpr/README.md) 与 [运行顺序](experiments/dpr/RUN_ORDER.md)。不要为了查看汇总结果重复训练。

## 数据、结果和许可

原始数据不随公开仓库提供。fetch-data 已实际验证：下载原作者固定提交中的 HomeACF 文件并检查 SHA-256，错误版本不会被覆盖或接收。原作者 MIT 许可及版权通知见 [许可说明](experiments/dpr/THIRD_PARTY_NOTICES.md)。本地离线数据附件只用于交接，不在上传清单内。

[实验台账](experiments/dpr/experiment_registry.json)对应 13 组分析及论文位置；[必要汇总结果](experiments/dpr/evidence/README.md)包含 18 份已接受结果文件和完整合成实验日志，不包含个体预测缓存。[变量字典](experiments/dpr/data_dictionary.json)说明冻结编码及其局限。

## 验证和边界

此前已通过六组关键新拟合检查、最终压缩包种子 0 重训、项目 wheel 安装、数据下载/校验、3,708 条合成记录核对、七张图和十四张表检查。本次没有重复训练，只核查文档调整后的文件/运行依赖闭包、哈希、压缩包和入口。详见 [验证说明](experiments/dpr/VERIFICATION.md) 与 [机器可读状态](experiments/dpr/VALIDATION_STATUS.json)。

不声称全部实验从头重跑、Linux 已验证、真实筛查时序已经恢复或真实临床部署有效。表格导出是已接受版本快照，不是重新拟合后自动成表。

## 独立交付和公开

准备上传的准确范围是 [upload_manifest.json](experiments/dpr/upload_manifest.json)，打包使用 release_manifest.json。私人伦理原件、个人运行路径、原始参与者记录、环境、缓存、临时文件、旧投稿草稿和本地验证日志均不作为公开交付材料。

本地来源 Git 历史保留；源码 ZIP 不包含 .git。已有 origin/upstream 是旧来源记录，不是本独立仓库的上传授权或新地址。公开时只发布经审核的白名单文件；不要直接把本地目录及全部来源历史一键推送。源历史包含旧诊断日志等文件，如拟公开完整历史，需另行审核其公开范围，不在本次修改/删除历史。

团队仓库：[tb-risk-research/tb-risk-dpr-reproducibility](https://github.com/tb-risk-research/tb-risk-dpr-reproducibility)。用户已确认团队归属、仓库名称及使用洁净交付树建立独立提交历史。当前仓库为私有，公开时机仍待确认；公开归档和 DOI 由主负责人处理。日后确有通用改进时，再选择性向源项目贡献。

本仓库第一方代码及随仓库提供的第一方支持材料采用 [MIT 许可证](LICENSE)，版权通知为 Copyright (c) 2026 Zhenyao Guo（郭臻尧）。本次按用户授权补齐源项目已声明 MIT 的根级许可正文。范围与排除项见 [LICENSING.md](LICENSING.md)；第三方代码、依赖和数据保留各自原许可。许可已补齐，公开发布时间与 DOI 归档尚未执行。

首次上传保留上游工作流源码，但仓库 GitHub Actions 已关闭，避免自动运行原项目全量测试、构建或向翻译服务发送文档。只有明确审核后再启用。

## Current paper delivery: v0.17 (local preparation, not uploaded)

This candidate retains the upstream repository and base commit recorded above. It adds current aggregate reviewer sensitivities, original sensitivity execution sources, portable path adapters, and accepted table/figure snapshots. No model fitting was repeated. The existing .git and source repository have not been altered; this staging folder is a file-level delivery candidate, not a replacement Git history.

Use `python experiments/dpr/export_publication.py --output data/processed/publication_v017` to export exactly 5 main tables, 15 supplementary table blocks (S1–S13 with S4a/b and S9a/b), and 11 PNG/PDF figure pairs. This copies audited accepted displays, not freshly fitted results. `delivery.py tables/figures` now select these snapshots; the older v0.8 generators are retained explicitly in experiments/dpr/legacy. Existing model-run entry points and dependencies are unchanged. See REVIEWER_SENSITIVITY_RUN_ORDER.md for regeneration dependencies and limits.

No participant-level raw data, prediction caches, private ethics originals, author correspondence, manuscript drafts, local environments, or newly explored external-validation experiments are included. Original data are recovered from the pinned author source and checksum; source licence notices are retained. The first-party repository licence is MIT; see root LICENSE and LICENSING.md. The complete permission notice has been added under the user's instruction. Access, public release timing and archive DOI remain separate steps; no release has been published.

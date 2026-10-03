#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开数据集下载器：WHO GHO / OWID / Prem 接触矩阵 / 世界银行 / TB Portals(鉴权探测)

设计原则（与 tb_risk 工程约定一致）：
- 原始数据只追加不覆盖：data/raw/ 按源分目录，manifest 记录 URL/许可/日期/SHA256
- 全部走无鉴权官方通道；TB Portals 个体数据需注册（401 实测），仅做可用性探测并记录
- raw.githubusercontent.com 在本机代理下超时，GitHub 文件改走 api.github.com base64 通道

用法：
    python data/download_public_datasets.py            # 全部源
    python data/download_public_datasets.py --only gho,worldbank
"""
import argparse
import base64
import csv
import datetime
import hashlib
import io
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
MANIFEST = os.path.join(RAW, "_manifest.json")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) tb-risk-research/1.0"}
TIMEOUT = 60

# WHO GHO 指标（全球健康观察站 OData API，官方无鉴权）
# https://www.who.int/data/gho
GHO_BASE = "https://ghoapi.azureedge.net/api"
GHO_INDICATORS = {
    "TB_e_inc_num": "估算新发结核病例数",
    "TB_e_prev_num": "估算结核患病数",
    "TB_e_mort_exc_tbhiv_num": "结核死亡数（不含HIV）",
    "TB_e_mdr_num": "估算MDR-TB病例数（肺结核通报中）",
    "TB_e_inc_rr_num": "估算MDR/RR-TB新发病例数",
    "TB_e_inc_tbhiv_100k": "HIV阳性结核发病率(/10万)",
    "TB_c_notified": "结核通报病例总数",
    "TB_notif_num": "新发结核通报数",
    "TB_c_new_tsr": "新病例治疗成功率(%)",
    "TB_notif_num_agesex": "按年龄性别通报数",
}
GHO_SPATIAL = ["CHN", "GLOBAL"]  # 中国 + 全球汇总

# OWID grapher（WHO 衍生，交叉核验用）
# 注意：tuberculosis-deaths（绝对数）grapher 已 403/下线（2026-09-05 实测，
# 无 csvType 参数同样 403），改用 tuberculosis-death-rate（每10万死亡率）
OWID_CSVS = {
    "tuberculosis-incidence-per-100000-people.csv":
        "https://ourworldindata.org/grapher/tuberculosis-incidence-per-100000-people.csv?csvType=full",
    "tuberculosis-death-rate.csv":
        "https://ourworldindata.org/grapher/tuberculosis-death-rate.csv?csvType=full",
    # TB 协变量时序（2026-09-05 全球搜寻新增，SEIR 外部驱动/协变量分析）
    "share-of-the-population-infected-with-hiv.csv":
        "https://ourworldindata.org/grapher/share-of-the-population-infected-with-hiv.csv",
    "diabetes-prevalence.csv":
        "https://ourworldindata.org/grapher/diabetes-prevalence.csv",
    "share-of-adults-who-smoke.csv":
        "https://ourworldindata.org/grapher/share-of-adults-who-smoke.csv",
    "prevalence-of-undernourishment.csv":
        "https://ourworldindata.org/grapher/prevalence-of-undernourishment.csv",
}

# Prem et al. (2017, PLOS Comput Biol) 合成接触矩阵：mobs-lab/mixing-patterns
# 中国国家层面：家庭/学校/工作/社区 × 85 年龄组（另有 18 组版本）
CONTACT_REPO = "mobs-lab/mixing-patterns"
CONTACT_BRANCH = "main"
CHINA_MATRIX_FILES = [
    f"China_country_level_{tag}_{g}.csv"
    for tag in ["M_overall_contact_matrix", "F_household_setting", "M_household_setting",
                "F_school_setting", "M_school_setting", "F_work_setting", "M_work_setting",
                "F_community_setting", "M_community_setting"]
    for g in ([18, 85] if "overall" in tag else [85])
]

# 世界银行 API（人口学/环境背景，JSON 无鉴权）
WB_BASE = "https://api.worldbank.org/v2/country/CHN/indicator"
WB_INDICATORS = {
    "SP.POP.TOTL": "总人口",
    "SP.POP.TOTL.FE.ZS": "女性人口占比(%)",
    "SP.POP.65UP.TO.ZS": "65岁以上人口占比(%)",
    "SP.POP.1564.TO.ZS": "15-64岁人口占比(%)",
    "SP.POP.0014.TO.ZS": "0-14岁人口占比(%)",
    "EN.ATM.PM25.PC.MC.ZS": "PM2.5年均暴露(μg/m³)",
    "EN.POP.DNST": "人口密度(人/km²)",
    "SP.DYN.LE00.IN": "出生预期寿命(年)",
}

TBPORTALS_BASE = "https://analytic.tbportals.niaid.nih.gov"

# NHANES 2011-2012（美国 CDC 公开微观据，无需注册）
# TB_G = QuantiFERON Gold (IGRA) 真实检测结果（LTBI 标签源）；
# 注意 TST_G 实为睾酮检测（TeSTosterone），非结核菌素试验
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles"
NHANES_FILES = {
    "TST_G.xpt": "血清睾酮(注意：TeSTosterone，非结核菌素试验；保留供交叉核对)",
    "TB_G.xpt": "QuantiFERON Gold in-Tube (IGRA，LTBI 标签源)",
    "DEMO_G.xpt": "人口学(年龄/性别/出生地/种族/收入)",
    "BMX_G.xpt": "体测(BMI)",
    "DIQ_G.xpt": "糖尿病问卷",
    "SMQ_G.xpt": "吸烟问卷",
    "MCQ_G.xpt": "医疗史(含癌症)",
}

# PM2.5：世行 API 已下线 EN.ATM.* 环境指标（data.worldbank.org 显示
# "no longer available"，实测连指标元数据端点均 404/Invalid value）。
# 改用 OWID grapher 通道（源：State of Global Air / HEI），与世行数值同源口径。
PM25_CSVS = {
    "pm25-air-pollution.csv":
        ("https://ourworldindata.org/grapher/pm25-air-pollution.csv?csvType=full",
         "OWID/State of Global Air, CC BY 4.0（世行下线后的替代源）"),
}

# LSHTM Data Compass：TREATS 队列（赞比亚+南非 14 社区 15-24 岁，QFT-Plus LTBI）
# DOI 10.17037/DATA.00003627（baseline, 4529 人）/ 10.17037/DATA.00003879（recentremote）
# 注意：URL 必须带文件名后缀（裸 /eprint/{id}/{n}/ 返回 404，2026-09 实测）；
# HIV 状态文件 (3627/3) 受访问控制 (401)，开放 CSV 不含 hiv 列
LSHTM_FILES = [
    ("lshtm/TREATS-ic-baseline-dataset_open.csv",
     "https://datacompass.lshtm.ac.uk/id/eprint/3627/4/TREATS-ic-baseline-dataset_open.csv",
     "TREATS baseline：4529 名 15-24 岁青少年 QFT-Plus LTBI + 社区/家庭暴露"),
    ("lshtm/TREATS-ic-baseline_data_codebook.html",
     "https://datacompass.lshtm.ac.uk/id/eprint/3627/1/TREATS-ic-baseline_data_codebook.html",
     "TREATS baseline 变量代码簿"),
    ("lshtm/3627_UserGuide.html",
     "https://datacompass.lshtm.ac.uk/id/eprint/3627/2/3627_UserGuide.html",
     "TREATS baseline 用户指南"),
    ("lshtm/TREATS-ic-recentremote_data.csv",
     "https://datacompass.lshtm.ac.uk/id/eprint/3879/2/TREATS-ic-recentremote_data.csv",
     "TREATS recentremote：1343 人 QFT-Plus 连续读数(nil/mitogen/TB1nil/TB2nil)"),
    ("lshtm/3879-UserGuide.html",
     "https://datacompass.lshtm.ac.uk/id/eprint/3879/4/3879-UserGuide.html",
     "TREATS recentremote 用户指南"),
    ("lshtm/TREATS-ic-recentremote_data_codebook.html",
     "https://datacompass.lshtm.ac.uk/id/eprint/3879/7/TREATS-ic-recentremote_data_codebook.html",
     "TREATS recentremote 变量代码簿"),
    # CRP 社区筛查（赞比亚+南非，开放版个体数据；对照 CRP_dataset_controlled.zip 受控）
    ("lshtm/CRP_dataset_open.csv",
     "https://datacompass.lshtm.ac.uk/id/eprint/5150/2/CRP_dataset_open.csv",
     "POC CRP 社区 TB 筛查开放版：9588 筛查，1777 验证（Xpert/培养），117 确诊"),
    ("lshtm/CRP_dataset_codebook.html",
     "https://datacompass.lshtm.ac.uk/id/eprint/5150/4/CRP_dataset_codebook.html",
     "CRP 数据集代码簿"),
    # Cape Town 社会接触调查（开放，经验性非洲接触模式，GNN 校准参照）
    ("lshtm/Cape_Town_social_contact_data.zip",
     "https://datacompass.lshtm.ac.uk/id/eprint/2756/10/Cape_Town_social_contact_data.zip",
     "Cape Town 社会接触调查：1115 人 × 家庭/密切/建筑/交通接触"),
    ("lshtm/2756_Codebook.html",
     "https://datacompass.lshtm.ac.uk/id/eprint/2756/14/2756_Codebook.html",
     "Cape Town 接触调查代码簿"),
    # ERASE-TB：仅 codebook（元数据）。个体数据 401 受控，
    # 且受预注册协议管辖（docs/erase_tb_validation_protocol_prereg.md），禁止用于训练
    ("lshtm/ERASE_predictionmodel_data_codebook.html",
     "https://datacompass.lshtm.ac.uk/id/eprint/4571/1/ERASE_predictionmodel_data_codebook.html",
     "ERASE-TB 预测模型数据集代码簿（个体数据 401 受控+预注册管辖，仅文档）"),
]

# LSHTM 受控数据登记（401，需向数据保管人申请；URL 模式 /id/eprint/{eid}/{n}/{filename}）
LSHTM_RESTRICTED = {
    "lshtm/ERASE_predictionmodel_data.csv": (
        "401 受控：ERASE-TB 家庭接触者 IGRA 队列（eprint/4571/2）——本项目预注册"
        "外验目标数据集（docs/erase_tb_validation_protocol_prereg.md），"
        "严禁用于训练，结局分析前须先跑 G1-G3 门检"),
    "lshtm/AHCoS_prediction_model_data.csv": (
        "401 受控：南非农村家庭传播风险评分数据（eprint/4932/3，"
        "AHRI 家庭接触研究，家庭级结局=儿童 QFT-Plus 阳性）"),
    "lshtm/hetu-dataset.txt": (
        "401 受控：QuantiFERON Gold Plus vs G 个体对比（eprint/1891/2）"),
    "lshtm/TBdataset.txt": (
        "401 受控：亚临床肺 TB 与 HIV（eprint/3373/1）"),
    "lshtm/tbmigrant_dataset.txt": (
        "401 受控：旅行与移民 TB（eprint/3534/1）"),
    "lshtm/castle_cohort_data.txt": (
        "401 受控：CASTLE TB 筛查试验队列（eprint/3552/2）"),
    "lshtm/castle_trial_data.txt": (
        "401 受控：CASTLE TB 筛查试验数据（eprint/3552/3）"),
    "lshtm/TREATS_IDP_analysis_data.csv": (
        "401 受控：TREATS 异常 CXR 无 TB 者结局（eprint/3878/4）"),
    "lshtm/Dataset.csv": (
        "401 受控：诊断算法与疾病严重度（eprint/5244/1）"),
    "lshtm/Minimal_dataset.zip": (
        "401 受控：ART 普治下 PLHIV TB 患病率（eprint/5372/1）"),
    "lshtm/Karachi_prevalence_data": (
        "Karachi 成人 TB 患病率调查（eprint/3788）：仅 codebook+分析代码，"
        "数据文件未公开挂载"),
    "lshtm/remoteinfection_KPS_data": (
        "Karonga 30 年远期感染研究（eprint/2829）：仅 codebook+生存分析代码，"
        "数据文件未公开挂载"),
}


# ---------------------------------------------------------------------------
# 2026-09-05 全球搜寻新增源（三路 agent 搜寻 + 本机通道实测筛选）
# 实测可达：Mendeley public-files API / Harvard Dataverse 直接下载 API /
#           WHO TME generateCSV / OWID grapher
# 实测封锁（本机出口 IP 403 或 DNS 失败，浏览器或代理可能可下）：见 BLOCKED_MANUAL
# ---------------------------------------------------------------------------

# Mendeley Data（Elsevier，public-files API 通道，实测 200）
MENDELEY_FILES = [
    # 台湾 NTUH QFT-Plus/活动性 TB 双结局（Wang Jann-Yuan 2021, CC BY 4.0）
    # 336 人 × 34 列：QFT-Plus 定量(Nil/TB_Ag/T1/T2/CD8) + cxr_score +
    # 糖尿病/ESRD/癌症/肝硬化/COPD/激素 + smear/culture/TB 结局
    ("mendeley/taiwan_qftplus.csv",
     "https://data.mendeley.com/public-files/datasets/457bkx9p6n/files/"
     "c529542b-16b3-44f5-9649-454769d1b9fa/file_downloaded",
     "台湾 QFT-Plus 双结局队列：LTBI(QFT) + 活动性 TB(culture/smear) + CXR 评分 + 合并症",
     "CC BY 4.0 (Mendeley Data, doi:10.17632/457bkx9p6n.2)"),
    # 巴西东北部 IGRA/TST/IFN-γ/IFNG+874 基因型（Carneiro et al. 2018）
    ("mendeley/brazil_igra_ifng.xlsx",
     "https://data.mendeley.com/public-files/datasets/mfcwvxwrhc/files/"
     "517c91d2-bf73-4e31-997f-8ca5a9772393/file_downloaded",
     "巴西 IGRA/TST 双结局 + IFN-γ 定量 + IFNG+874 基因型（TB/LTBI 分组）",
     "CC BY 4.0 (Mendeley Data, doi:10.17632/mfcwvxwrhc.1)"),
]

# Harvard Dataverse（直接下载 API /api/access/datafile/{id}，实测 200 无需登录）
DATAVERSE_FILES = [
    # Yuen et al. 2019 秘鲁 Carabayllo 接触者管理级联（CC0, doi:10.7910/DVN/EMNPYJ）
    ("dataverse/carabayllo_contact_evaluation.tab",
     "https://dataverse.harvard.edu/api/access/datafile/3425482",
     "秘鲁 Carabayllo 家庭接触者评估级联（Partners In Health）",
     "CC0 1.0 (Harvard Dataverse, doi:10.7910/DVN/EMNPYJ)"),
    # Park 2018 韩国 HCW TST/IGRA 串行检测（CC0, doi:10.7910/DVN/1NYBLW）
    ("dataverse/korea_hcw_tst_igra.xlsx",
     "https://dataverse.harvard.edu/api/access/datafile/3210719",
     "韩国 458 名医护人员年度 TST + 阳转者 IGRA + LTBI 治疗后发病（2009-2013）",
     "CC0 1.0 (Harvard Dataverse, doi:10.7910/DVN/1NYBLW)"),
    ("dataverse/korea_hcw_codingbook.xlsx",
     "https://dataverse.harvard.edu/api/access/datafile/3210720",
     "韩国 HCW 数据集编码簿",
     "CC0 1.0 (Harvard Dataverse, doi:10.7910/DVN/1NYBLW)"),
]

# WHO 全球 TB 报告 TME CSV 通道（与 GHO 指标 API 不同源：报告附件原始表，
# 变量更细、含不确定性区间；generateCSV.asp 实测 200 无需注册）
WHO_TME_CSVS = {
    "estimates": "TB 负担估算（发病率/死亡率+95%UI，215+国 × 2000-2024）",
    "estimates_age_sex": "按年龄/性别/风险因素分层发病率",
    "mdr_rr_estimates": "MDR/RR-TB 负担估算",
    "ltbi_estimates": "密切接触者 TB 感染估算（家庭接触者数/TPT 覆盖率）",
    "notifications": "病例报告数（各型/年龄/性别/HIV/耐药明细）",
    "contact_tpt": "接触者筛查与 TPT（启动/完成）",
    "outcomes": "治疗结局（治愈/死亡/失访/失败）",
    "outcomes_age_sex": "治疗结局（分年龄性别）",
    "provisional_notifications": "月度/季度临时报告数（2020-）",
    "dictionary": "数据字典（全变量定义）",
}

# 本机出口被封的高价值数据集（figshare/AWS 系 CDN 403、Zenodo DNS 失败、
# Dryad JS 挑战、data.cdc.gov 403——2026-09-05 五轮实测）。
# 普通浏览器（走系统代理）大概率可下：手动下载后放入 raw/ 对应子目录，
# 重跑下载器即可被 manifest 收录（record 不校验来源）。
BLOCKED_MANUAL = {
    "figshare/uganda_household_igra.xlsx": (
        "本机 403（figshare CDN 封锁出口 IP）。乌干达家庭接触者 IGRA 队列"
        "（Muchuro 2022, n=355, 27 列：IGRA+HIV+BCG scar+接触强度梯度+户ID）。"
        "手动: https://figshare.com/ndownloader/files/36177084"),
    "figshare/peru_aibana_contacts.dta": (
        "本机 403（PLOS S1 重定向 googleapis 亦不可达）。秘鲁利马 14,044 家庭"
        "接触者营养-进展队列（Aibana 2016：TST 转阳 1,787 + 继发 TB 406 双终点，"
        "含 HIV/糖尿病/BCG/营养）。手动: https://doi.org/10.1371/"
        "journal.pone.0166333.s001（浏览器打开）"),
    "figshare/german_students_dual_igra.sav": (
        "本机 403。德国留学生 QFT-GIT vs QFT-Plus 双平台配对（n=134）。"
        "手动: https://figshare.com/ndownloader/files/6989846"),
    "dryad/peru_mdr_household_incident.xlsx": (
        "本机 JS 挑战（脚本 401）。秘鲁 MDR vs 敏感 TB 家庭接触者 3 年前瞻队列"
        "（Grandjean 2016, n=3,417, incident TB 149 例, 户级聚类）——最高价值。"
        "手动: https://datadryad.org/stash/dataset/doi:10.5061/dryad.br760"
        " 页面 Download 按钮"),
    "dryad/uganda_qftplus_contacts.xlsx": (
        "Dryad API 可达（元数据正常，文件 doi:10.5061/dryad.k3j9kd5bg, "
        "file_id 1883732, sha-256 da66ca…856c, 57,198 B）但 /downloads/* "
        "全部 403（2026-09-05 晚二次复测：curl/页内 XHR/顶层导航三通道均"
        "403，疑 AWS WAF 速率型 IP 封锁——当日 20:07 br760 浏览器通道曾"
        "成功，冷却后可重试；Anubis JS 层浏览器可过）。乌干达坎帕拉成人"
        "接触者 QFT-Plus（n=202, HIV 33%, Chunda/Mayito 2022）。"
        "重试: https://datadryad.org/dataset/doi:10.5061/dryad.k3j9kd5bg"
        " 页面 Download 按钮"),
    "zenodo/spain_qftgit_contacts_5y.sav": (
        "本机 DNS 污染（zenodo.org→假 IP 199.59.149.235/Facebook IPv6）"
        "且 DoH 端点（alidns/1.1.1.1）本机 SSL 阻断，--resolve 绕行不通"
        "——需用户 VPN（2026-09-05 复测确认）。西班牙 BCG 接种成人接触者 "
        "QFT-GIT 筛查 + 5 年随访发病（n=661, CC0）——SOP 潜在第四国"
        "前瞻队列。手动: https://zenodo.org/records/4931404"),
    "zenodo/lima_treatment_default.xls": (
        "本机 DNS 失败+IP 403。秘鲁利马涂阳患者治疗中断队列（n=1,233, 127 "
        "中断, CC0）。手动: https://zenodo.org/records/4992464"),
    "cdc/tbesc2_partA_tst_igra.zip": (
        "本机 403（data.cdc.gov Socrata 封锁）。CDC TBESC-II Part A：TST vs "
        "IGRA 头对头 + 进展预测（n≈2,121 接触者, Public Domain）。"
        "手动: https://data.cdc.gov/download/5hpj-p74g/application/x-zip-compressed"),
    "cdc/tbesc2_partB_ltbi_cascade.zip": (
        "本机 403。CDC TBESC-II Part B：LTBI 预防级联监测。"
        "手动: https://data.cdc.gov/download/epap-ayij/application/x-zip-compressed"),
    "dataverse/carabayllo_contact_evaluation.tab": (
        "本机 S3 重定向域名 dvn-cloud-iqss.s3.amazonaws.com DNS 污染"
        "（间歇可达，2026-09-05 仅一次成功后未再现）。秘鲁 Carabayllo "
        "家庭接触者评估级联（Yuen 2019, CC0）。重跑 --only dataverse "
        "碰运气或手动: https://dataverse.harvard.edu/api/access/"
        "datafile/3425482（浏览器）"),
    "dataverse/korea_hcw_tst_igra.xlsx": (
        "本机 S3 DNS 污染（同上）。韩国 458 名医护人员 TST/IGRA 串行"
        "检测（Park 2018, CC0, 2009-2013）。重跑 --only dataverse 或"
        "手动: https://dataverse.harvard.edu/api/access/datafile/3210719"),
}


def dl_lshtm(m, session):
    """LSHTM Data Compass → raw/lshtm/*（开放个体级数据 + codebook + 受控登记）

    开放个体数据：TREATS baseline/recentremote、CRP 开放版、Cape Town 接触调查
    受控(401)：ERASE-TB(预注册管辖)/AHCoS/CASTLE 等，见 LSHTM_RESTRICTED
    """
    outdir = os.path.join(RAW, "lshtm")
    os.makedirs(outdir, exist_ok=True)
    m["unavailable"].update(LSHTM_RESTRICTED)
    m["unavailable"]["lshtm/TREATS-ic-baseline-HIVstatus.txt"] = (
        "401 受访问控制：HIV 状态需向 LSHTM 数据保管人申请"
        "（DOI 10.17037/DATA.00003627 文件 3）")
    for relpath, url, note in LSHTM_FILES:
        fname = os.path.basename(relpath)
        try:
            content = fetch(session, url).content
            fpath = os.path.join(outdir, fname)
            with open(fpath, "wb") as f:
                f.write(content)
            record(m, relpath, url,
                   "CC BY 4.0（LSHTM Data Compass, DOI 10.17037/DATA.00003627/.00003879）",
                   note)
            print(f"  [OK] {fname} ({len(content)/1024:.0f}KB)")
        except Exception as e:
            print(f"  [失败] {fname}: {str(e)[:120]}")
            m["unavailable"][relpath] = str(e)[:200]


def dl_nhanes(m, session):
    """NHANES 2011-2012 XPT → raw/nhanes/*.xpt（公开个体级微观据）"""
    outdir = os.path.join(RAW, "nhanes")
    os.makedirs(outdir, exist_ok=True)
    for fname, cn_name in NHANES_FILES.items():
        url = f"{NHANES_BASE}/{fname}"
        try:
            content = fetch(session, url).content
            fpath = os.path.join(outdir, fname)
            with open(fpath, "wb") as f:
                f.write(content)
            record(m, f"nhanes/{fname}", url,
                   "Public domain（CDC NCHS NHANES，美国联邦政府公开数据）",
                   f"{cn_name} 2011-2012 周期")
            print(f"  [OK] {fname} ({len(content)/1024:.0f}KB)")
        except Exception as e:
            print(f"  [失败] {fname}: {str(e)[:120]}")
            m["unavailable"][f"nhanes/{fname}"] = str(e)[:200]


def dl_pm25(m, session):
    """PM2.5（OWID grapher；世行 EN.ATM.PM25.PC.MC.ZS 已下线，见 manifest 说明）"""
    outdir = os.path.join(RAW, "owid")
    os.makedirs(outdir, exist_ok=True)
    m["unavailable"]["worldbank/EN.ATM.PM25.PC.MC.ZS"] = (
        "世行已下线该指标：data.worldbank.org/indicator/EN.ATM.PM25.PC.MC.ZS "
        "显示 'no longer available'；API 元数据端点亦不存在（2026-08 实测）。"
        "PM2.5 改用 OWID(State of Global Air) 同口径数据")
    for fname, (url, license_) in PM25_CSVS.items():
        try:
            content = fetch(session, url).content
            fpath = os.path.join(outdir, fname)
            with open(fpath, "wb") as f:
                f.write(content)
            record(m, f"owid/{fname}", url, license_,
                   "PM2.5 年均暴露(μg/m³，人口加权)；世行下线后的替代源")
            print(f"  [OK] {fname} ({len(content)/1024:.0f}KB)")
        except Exception as e:
            print(f"  [失败] {fname}: {str(e)[:120]}")
            m["unavailable"][f"owid/{fname}"] = str(e)[:200]


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"downloaded": {}, "unavailable": {}, "generated_at": None}


def save_manifest(m):
    m["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def record(m, relpath, url, license_, note=""):
    m["downloaded"][relpath] = {
        "url": url, "license": license_, "note": note,
        "sha256": sha256_bytes(open(os.path.join(RAW, relpath), "rb").read()),
        "size_bytes": os.path.getsize(os.path.join(RAW, relpath)),
        "downloaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    # manifest 卫生（2026-09-05 审计）：成功后清除同路径的陈旧 unavailable
    # 条目——旧 bug/旧 URL 的失败记录在修复后会永久残留，导致 unavailable
    # 虚高（实测 40 条中 11 条为已修复的陈旧噪音，如 pm2.5→pm25 URL 修正）
    m["unavailable"].pop(relpath, None)


def fetch(session, url, headers=None, **kw):
    r = session.get(url, headers=headers or UA, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


def dl_gho(m, session):
    """WHO GHO：中国+全球时序指标 → raw/who_gho/*.csv"""
    outdir = os.path.join(RAW, "who_gho")
    os.makedirs(outdir, exist_ok=True)
    for code, cn_name in GHO_INDICATORS.items():
        for spatial in GHO_SPATIAL:
            url = (f"{GHO_BASE}/{code}?$filter=SpatialDim%20eq%20%27{spatial}%27"
                   f"&$select=SpatialDim,TimeDim,NumericValue,Dim1,Dim2")
            try:
                data = fetch(session, url).json()["value"]
                if not data:
                    print(f"  [空] {code}/{spatial}")
                    continue
                fname = f"{code}_{spatial}.csv"
                fpath = os.path.join(outdir, fname)
                cols = ["SpatialDim", "TimeDim", "Dim1", "Dim2", "NumericValue"]
                with open(fpath, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    for row in sorted(data, key=lambda x: x.get("TimeDim") or 0):
                        w.writerow(row)
                record(m, f"who_gho/{fname}", url,
                       "WHO GHO 开放数据（引用注明 WHO Global Health Observatory）",
                       f"{cn_name} | {len(data)} 行")
                print(f"  [OK] {fname} ({len(data)} 行)")
            except Exception as e:
                print(f"  [失败] {code}/{spatial}: {e}")
                m["unavailable"][f"who_gho/{code}_{spatial}"] = str(e)[:200]


def dl_owid(m, session):
    """OWID grapher CSV（WHO 衍生）→ raw/owid/*.csv"""
    outdir = os.path.join(RAW, "owid")
    os.makedirs(outdir, exist_ok=True)
    for fname, url in OWID_CSVS.items():
        try:
            content = fetch(session, url).content
            fpath = os.path.join(outdir, fname)
            with open(fpath, "wb") as f:
                f.write(content)
            record(m, f"owid/{fname}", url,
                   "CC BY 4.0 (OWID，原始来源 WHO)", "")
            print(f"  [OK] {fname} ({len(content)} bytes)")
        except Exception as e:
            print(f"  [失败] {fname}: {e}")
            m["unavailable"][f"owid/{fname}"] = str(e)[:200]


def dl_contact_matrices(m, session):
    """Prem et al. 中国接触矩阵 → raw/contact_matrices/*.csv（GitHub API base64 通道）"""
    outdir = os.path.join(RAW, "contact_matrices")
    os.makedirs(outdir, exist_ok=True)
    for fname in CHINA_MATRIX_FILES:
        path = f"data/contact_matrices/{fname}"
        url = (f"https://api.github.com/repos/{CONTACT_REPO}/contents/{path}"
               f"?ref={CONTACT_BRANCH}")
        try:
            j = fetch(session, url,
                      headers={"Accept": "application/vnd.github+json", **UA}).json()
            if j.get("encoding") != "base64":
                raise RuntimeError(f"意外编码: {j.get('encoding')}")
            content = base64.b64decode(j["content"])
            fpath = os.path.join(outdir, fname)
            with open(fpath, "wb") as f:
                f.write(content)
            record(m, f"contact_matrices/{fname}", j.get("html_url", url),
                   "CC BY 4.0 (Prem et al. 2017, PLOS Comput Biol)", "")
            print(f"  [OK] {fname} ({len(content)} bytes)")
        except Exception as e:
            print(f"  [失败] {fname}: {str(e)[:120]}")
            m["unavailable"][f"contact_matrices/{fname}"] = str(e)[:200]


def dl_worldbank(m, session):
    """世界银行：中国人口/环境指标 → raw/worldbank/*.csv"""
    outdir = os.path.join(RAW, "worldbank")
    os.makedirs(outdir, exist_ok=True)
    for code, cn_name in WB_INDICATORS.items():
        url = f"{WB_BASE}/{code}?format=json&date=1990:2024&per_page=100"
        try:
            payload = fetch(session, url).json()
            rows = payload[1] if len(payload) > 1 else []
            if not rows:
                print(f"  [空] {code}")
                m["unavailable"][f"worldbank/{code}"] = "API 无数据（PM2.5 等环境指标建议走 CNEMC/ERA5）"
                continue
            fname = f"CHN_{code.replace('.', '_')}.csv"
            fpath = os.path.join(outdir, fname)
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["year", "value", "indicator"])
                for row in sorted(rows, key=lambda x: int(x["date"])):
                    w.writerow([row["date"], row["value"], cn_name])
            record(m, f"worldbank/{fname}", url,
                   "CC BY 4.0 (世界银行开放数据)", cn_name)
            print(f"  [OK] {fname} ({len(rows)} 行)")
        except Exception as e:
            print(f"  [失败] {code}: {e}")
            m["unavailable"][f"worldbank/{code}"] = str(e)[:200]


def dl_tbportals(m, session):
    """TB Portals Analytic API：仅探测可用性。

    实测 /api/Patient-Case 返回 401（需在 tbportals.niaid.nih.gov 注册并
    在 Analytic API 设置 API Key）。注册后：
      - 在下方 TBPORTALS_TOKEN 填入 token，或设环境变量 TBPORTALS_TOKEN
      - 重跑本函数即可拉取个体级临床数据（Patient-Case / Bacteriology 等）
    """
    token = os.environ.get("TBPORTALS_TOKEN", "")
    outdir = os.path.join(RAW, "tb_portals")
    os.makedirs(outdir, exist_ok=True)
    url = f"{TBPORTALS_BASE}/api/Patient-Case?returnCsv=true&cohortId="
    try:
        headers = dict(UA)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = session.get(url, headers=headers, timeout=TIMEOUT)
        if r.status_code == 401 and not token:
            m["unavailable"]["tb_portals/Patient-Case.csv"] = (
                "401 Unauthorized：个体级数据需在 tbportals.niaid.nih.gov 注册"
                "后于 Analytic API 创建 token（设环境变量 TBPORTALS_TOKEN 后重跑）")
            print("  [需注册] TB Portals 个体数据 401（已在 manifest 记录接入方式）")
            return
        r.raise_for_status()
        fpath = os.path.join(outdir, "Patient-Case.csv")
        with open(fpath, "wb") as f:
            f.write(r.content)
        record(m, "tb_portals/Patient-Case.csv", url,
               "TB Portals 数据使用协议（引用 NIAID TB Portals）")
        print(f"  [OK] Patient-Case.csv ({len(r.content)} bytes)")
    except Exception as e:
        m["unavailable"]["tb_portals/Patient-Case.csv"] = str(e)[:200]
        print(f"  [失败] TB Portals: {e}")


def dl_mendeley(m, session):
    """Mendeley Data public-files API → raw/mendeley/*（个体级队列）"""
    outdir = os.path.join(RAW, "mendeley")
    os.makedirs(outdir, exist_ok=True)
    for relpath, url, note, license_ in MENDELEY_FILES:
        fname = os.path.basename(relpath)
        try:
            content = fetch(session, url).content
            fpath = os.path.join(outdir, fname)
            with open(fpath, "wb") as f:
                f.write(content)
            record(m, relpath, url, license_, note)
            print(f"  [OK] {fname} ({len(content)/1024:.0f}KB)")
        except Exception as e:
            print(f"  [失败] {fname}: {str(e)[:120]}")
            m["unavailable"][relpath] = str(e)[:200]


def dl_dataverse(m, session):
    """Harvard Dataverse 直接下载 API → raw/dataverse/*（CC0 队列）

    已知问题（2026-09-05）：文件重定向到 dvn-cloud-iqss.s3.amazonaws.com
    （presigned URL），该 S3 域名本机 DNS 污染、仅间歇可达——失败属预期，
    重跑几次碰窗口，或见 BLOCKED_MANUAL 手动通道。
    """
    outdir = os.path.join(RAW, "dataverse")
    os.makedirs(outdir, exist_ok=True)
    for relpath, url, note, license_ in DATAVERSE_FILES:
        fname = os.path.basename(relpath)
        try:
            content = fetch(session, url).content
            fpath = os.path.join(outdir, fname)
            with open(fpath, "wb") as f:
                f.write(content)
            record(m, relpath, url, license_, note)
            print(f"  [OK] {fname} ({len(content)/1024:.0f}KB)")
        except Exception as e:
            print(f"  [失败] {fname}: {str(e)[:120]}")
            m["unavailable"][relpath] = str(e)[:200]


def dl_whotme(m, session):
    """WHO 全球 TB 报告 TME CSV → raw/who_tme/*.csv（与 GHO 不同源，更细）"""
    outdir = os.path.join(RAW, "who_tme")
    os.makedirs(outdir, exist_ok=True)
    base = "https://extranet.who.int/tme/generateCSV.asp?ds="
    for ds, cn_name in WHO_TME_CSVS.items():
        url = base + ds
        fname = f"who_tme_{ds}.csv"
        try:
            content = fetch(session, url).content
            fpath = os.path.join(outdir, fname)
            with open(fpath, "wb") as f:
                f.write(content)
            record(m, f"who_tme/{fname}", url,
                   "WHO 数据政策（免费使用，署名）", cn_name)
            print(f"  [OK] {fname} ({len(content)/1024:.0f}KB)")
        except Exception as e:
            print(f"  [失败] {fname}: {str(e)[:120]}")
            m["unavailable"][f"who_tme/{fname}"] = str(e)[:200]


def dl_blocked_manual(m, session):
    """登记本机出口被封的高价值数据集（不尝试下载，避免无效请求）

    figshare/Dryad/Zenodo/data.cdc.gov 对本机出口 IP 403 或 DNS 失败
    （2026-09-05 五轮实测：完整浏览器头、DoH 解析 + curl --resolve、
    SODA API 均无效）。浏览器走系统代理大概率可下：手动下载后放入
    raw/ 对应子目录（figshare/ dryad/ zenodo/ cdc/），重跑本下载器，
    record() 会收录已有文件（下载失败但文件存在时改为登记本地副本）。
    """
    for relpath, note in BLOCKED_MANUAL.items():
        m["unavailable"][relpath] = note
        # 若用户已手动放置文件，则收录进 downloaded
        local = os.path.join(RAW, relpath.replace("/", os.sep))
        if os.path.exists(local):
            record(m, relpath, "manual:" + relpath,
                   "见 BLOCKED_MANUAL（手动浏览器下载）",
                   "本机脚本通道被封，用户手动放置的副本")
            m["unavailable"].pop(relpath, None)
            print(f"  [手动副本已收录] {relpath}")
    print(f"  [登记] {len(BLOCKED_MANUAL)} 个被封通道数据集（附手动下载 URL）")


SOURCES = {
    "gho": dl_gho,
    "owid": dl_owid,
    "contact": dl_contact_matrices,
    "worldbank": dl_worldbank,
    "tbportals": dl_tbportals,
    "nhanes": dl_nhanes,
    "pm25": dl_pm25,
    "lshtm": dl_lshtm,
    "mendeley": dl_mendeley,
    "dataverse": dl_dataverse,
    "whotme": dl_whotme,
    "blocked": dl_blocked_manual,
}


def main():
    parser = argparse.ArgumentParser(description="公开数据集下载器")
    parser.add_argument("--only", default=None,
                        help="逗号分隔的源名: gho,owid,contact,worldbank,"
                             "tbportals,nhanes,pm25,lshtm,mendeley,dataverse,"
                             "whotme,blocked")
    args = parser.parse_args()

    selected = SOURCES if not args.only else {
        k: v for k, v in SOURCES.items() if k in args.only.split(",")}

    session = requests.Session()
    manifest = load_manifest()
    for name, fn in selected.items():
        print(f"==> {name}")
        try:
            fn(manifest, session)
        except Exception as e:
            print(f"  [源级失败] {name}: {e}")
            manifest["unavailable"][f"_source_{name}"] = str(e)[:200]

    # 对账（2026-09-05 审计）：unavailable 与 downloaded 互斥——
    # 已在盘且已登记 downloaded 的路径不可能是"不可用"；防止陈旧失败
    # 记录（旧 bug/旧 URL）永久残留虚增 unavailable 计数
    reconciled = [k for k in manifest["unavailable"]
                  if k in manifest["downloaded"]]
    for k in reconciled:
        del manifest["unavailable"][k]
    if reconciled:
        print(f"[对账] 清除 {len(reconciled)} 条与 downloaded 重叠的陈旧 "
              f"unavailable 记录")

    save_manifest(manifest)
    print(f"\nmanifest 已更新: {MANIFEST}")
    print(f"成功 {len(manifest['downloaded'])} 个文件, 不可用 {len(manifest['unavailable'])} 项")


if __name__ == "__main__":
    sys.exit(main())

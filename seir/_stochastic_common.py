"""随机SEIR模型共享常量与默认参数。

由原 seir/stochastic.py 拆分而来，集中存放各版本（7-房室 v2 / 9-房室 v3 /
270维 v4）共用的房室索引与文献默认参数，避免子模块间循环导入。
"""

import numpy as np

# 房室索引常量
IDX_S, IDX_LF, IDX_LS = 0, 1, 2
IDX_ISUB, IDX_ICLIN, IDX_R, IDX_C = 3, 4, 5, 6
N_COMPARTMENTS = 7

# 文献默认参数
# 注：RHO_PROG/ETA_SUB/GAMMA/BETA_REINF 等速率参数均为"合理默认值"，量级
# 对照 Horton et al. (2023) 与 Emery et al. (2023) 对亚临床结核自然史的参数化。
DEFAULT_RHO_FAST = 0.1 / 365.0        # ~0.1/年 → ~0.000274/day (Andrews 2012)
DEFAULT_RHO_CONV = 1.0 / 365.0        # ~1.0/年 → ~0.00274/day (2年窗口)
DEFAULT_RHO_REACT = 0.004 / 365.0     # ~0.4%/年 → ~1.1e-5/day (Vynnycky 1997)
DEFAULT_SIGMA_CLEAR = 2.0 / 365.0     # ~2.0/年 → ~0.00548/day (Horton 2023)
DEFAULT_RHO_PROG = 0.05               # I_sub→I_clin 进展率 /day
                                      # 文献：Horton et al. PNAS 2023；Emery et al. eLife 2023
                                      #（亚临床进展速率量级）
DEFAULT_ETA_SUB = 1.0                 # 亚临床相对传染性（保守默认）
                                      # 文献：Emery et al. eLife 2023（亚临床传染性
                                      # 中位数 ~1.93，代码取保守下限 1.0）
DEFAULT_P_CLIN = 0.332                # 直接临床进展比例 (Horton 2023)
DEFAULT_GAMMA = 0.05                  # 恢复率 /day (I_clin→R)
                                      # 文献：Vynnycky & Fine 1997（未经治疗TB自然史）
DEFAULT_BETA_REINF = 0.4              # 清除者再感染易感性降低系数
                                      # 文献：Emery et al. eLife 2023（再感染易感性量级）

# ==================== v3.0 9-房室模型参数 ====================
# 房室索引（9维）
IDX_S_V3, IDX_LF_V3, IDX_LS_V3 = 0, 1, 2
IDX_M_V3, IDX_ISUB_V3 = 3, 4
IDX_ISP_V3, IDX_ISN_V3 = 5, 6
IDX_R_V3, IDX_C_V3 = 7, 8
N_COMPARTMENTS_V3 = 9

# 改进4：疾病状态波动参数
DEFAULT_RHO_MIN = 0.5 / 365.0         # M→I_sub 进展率 ~0.5/年
DEFAULT_P_M = 0.5                     # 经 Minimal 的比例
                                      # 文献：Emery et al. eLife 2023（中间/亚临床
                                      # 房室占比量级，默认取 0.5）
DEFAULT_OMEGA_REG_M = 1.0 / 365.0     # M→L_slow 回退率 ~1.0/年 (Horton 2023)
DEFAULT_OMEGA_REG_SUB = 1.0 / 365.0   # I_sub→M 回退率 ~1.0/年

# 改进5：涂阳/涂阴分层参数 (Ragonnet et al. 2021, CID)
DEFAULT_P_SP = 0.55                   # 涂阳概率
DEFAULT_MU_SP = 0.389 / 365.0         # 涂阳死亡率 0.389/年
DEFAULT_R_SP = 0.231 / 365.0          # 涂阳自愈率 0.231/年
DEFAULT_MU_SN = 0.025 / 365.0         # 涂阴死亡率 0.025/年
DEFAULT_R_SN = 0.130 / 365.0          # 涂阴自愈率 0.130/年
DEFAULT_P_SN2SP = 0.1 / 365.0         # 涂阴→涂阳转化率 ~0.1/年
DEFAULT_ETA_SN = 0.35                 # 涂阴相对传染性

# 改进6：外源性再感染参数
DEFAULT_BETA_EXO = 0.33               # L_slow 外源性再感染易感性 (Beta(2,4))

# ==================== v4.0 年龄×HIV×耐药分层参数 ====================
# 改进7：年龄分层
N_AGE_V4 = 5                          # 年龄组数: 0-4, 5-14, 15-49, 50-64, 65+
N_HIV_V4 = 3                          # HIV 分层: HIV-, HIV+未治疗, HIV+ART
N_DR_V4 = 2                           # 耐药分层: DS, DR
STATE_SIZE_V4 = N_AGE_V4 * N_HIV_V4 * N_DR_V4 * N_COMPARTMENTS_V3  # = 270

# 年龄进展因子 (与 constants.py AGE_PROGRESSION 一致, Marais 2011, Davies 2006)
DEFAULT_AGE_PROGRESSION = np.array([4.0, 1.8, 1.2, 1.0, 2.0])

# WAIFW 混合矩阵 (POLYMOD-like, assortative mixing, Mossong 2008)
# 行=易感者年龄, 列=感染者年龄; 对角线=同年龄组内接触最强
DEFAULT_WAIFW = np.array([
    [4.0,  1.0,  1.5,  0.5,  0.3],   # 0-4
    [1.0,  8.0,  2.0,  0.5,  0.3],   # 5-14
    [1.5,  2.0, 12.0,  2.0,  1.0],   # 15-49
    [0.5,  0.5,  2.0,  6.0,  1.5],   # 50-64
    [0.3,  0.3,  1.0,  1.5,  4.0],   # 65+
])

# 改进8：HIV 共感染参数 (Pawlowski 2012, Houben 2016)
DEFAULT_RR_HIV = 20.0                 # HIV+ 再激活相对风险 (范围 10-50)
DEFAULT_ART_REDUCTION = 0.65          # ART 降低 TB 风险比例
DEFAULT_HIV_INFECTION_RATE = 0.001    # 外生 HIV 感染率 /day
DEFAULT_ART_INITIATION_RATE = 0.1 / 365.0  # ART 启动率 ~0.1/年

# 改进9：耐药 TB 参数 (Karmakar 2022, Houben & Dodd 2016)
DEFAULT_DR_FITNESS_COST = 0.15        # DR 株适合度代价 (传染性降低 15%)
DEFAULT_P_ACQ = 0.03                  # 获得性耐药概率 (治疗中 DS→DR)

__all__ = [
    # 7-房室索引与参数
    "IDX_S", "IDX_LF", "IDX_LS", "IDX_ISUB", "IDX_ICLIN", "IDX_R", "IDX_C",
    "N_COMPARTMENTS",
    "DEFAULT_RHO_FAST", "DEFAULT_RHO_CONV", "DEFAULT_RHO_REACT",
    "DEFAULT_SIGMA_CLEAR", "DEFAULT_RHO_PROG", "DEFAULT_ETA_SUB",
    "DEFAULT_P_CLIN", "DEFAULT_GAMMA", "DEFAULT_BETA_REINF",
    # v3.0 9-房室索引与参数
    "IDX_S_V3", "IDX_LF_V3", "IDX_LS_V3", "IDX_M_V3", "IDX_ISUB_V3",
    "IDX_ISP_V3", "IDX_ISN_V3", "IDX_R_V3", "IDX_C_V3", "N_COMPARTMENTS_V3",
    "DEFAULT_RHO_MIN", "DEFAULT_P_M", "DEFAULT_OMEGA_REG_M",
    "DEFAULT_OMEGA_REG_SUB", "DEFAULT_P_SP", "DEFAULT_MU_SP", "DEFAULT_R_SP",
    "DEFAULT_MU_SN", "DEFAULT_R_SN", "DEFAULT_P_SN2SP", "DEFAULT_ETA_SN",
    "DEFAULT_BETA_EXO",
    # v4.0 年龄×HIV×耐药参数
    "N_AGE_V4", "N_HIV_V4", "N_DR_V4", "STATE_SIZE_V4",
    "DEFAULT_AGE_PROGRESSION", "DEFAULT_WAIFW",
    "DEFAULT_RR_HIV", "DEFAULT_ART_REDUCTION",
    "DEFAULT_HIV_INFECTION_RATE", "DEFAULT_ART_INITIATION_RATE",
    "DEFAULT_DR_FITNESS_COST", "DEFAULT_P_ACQ",
]

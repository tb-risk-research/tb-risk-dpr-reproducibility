#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双头预测器（core 层产品能力，round-10 P2 正式化 / round-12 v2 升级）。

架构依据（sinan_bacteriology_head_20260826.json）：两个头的 top-decile
重叠仅 0.7% —— 死亡分层与「找患者」近乎正交，双头独立部署有明确
互补价值。0.7% 的定量依据让双头不再是猜想，而是测量过的事实。

任务映射（与证据面板任务分解对齐）：
  任务 A' = 细菌学确诊（找传染源）—— 排序涂片/培养阳性概率，
           对准病例发现 / 传染源控制
  任务 B  = 全因死亡预后 —— 对准治疗结局分层 / 高危随访资源分配

模型工件由 ``data/run_sinan_bacteriology_head.py`` 落盘（v1），
round-12 起由 ``data/run_sinan_bact_head_v2.py`` 落盘 v2（+HIV 检测
结果 + 职业编码 top-30；far test 死亡 AUROC 0.8116→0.8187）：
  tb_risk/models/sinan_dual_head/{encoder,death_head,bact_head}_v2.joblib

职业列 OCC_TRUNC 为派生列（ID_OCUPA_N 按训练期 top-30 词表截断 +
OTHER），词表由 v2 归档 feature_schema.occupation_top 提供并在加载时
核对（项目惯例：模型版本与特征 schema 必须双向核对，避免模型与
数据错配）。
"""
import json
import logging
import os

import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:  # pragma: no cover - 环境无 joblib 时降级
    joblib = None
    HAS_JOBLIB = False

LOGGER = logging.getLogger("tb_risk.core.dual_head_predictor")

# 特征 schema（v2：与 data/run_sinan_bact_head_v2.py 单一口径；
# 加载时与归档 JSON 的 feature_schema 对照）
NUMERIC_FEATS = ['AGE_YEARS', 'NU_CONTATO']
BINARY_FEATS = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
                'AGRAVDROGA', 'AGRAVTABAC', 'POP_LIBER', 'POP_RUA',
                'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV']
# encoder 拟合类别列（含 v2 新增 HIV 与派生列 OCC_TRUNC）
CATEGORICAL_FEATS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'FORMA',
                     'EXTRAPU1_N', 'RAIOX_TORA', 'TESTE_TUBE', 'CS_GESTANT',
                     'HIV', 'OCC_TRUNC']
# 派生列：由 ID_OCUPA_N 按训练期 top-30 词表截断 + OTHER（build_features）
DERIVED_FEATS = ['OCC_TRUNC']
OCC_RAW_COL = 'ID_OCUPA_N'
# 病原学确诊方式列（标签构成，特征侧整体剔除——泄漏控制）
EXCLUDED_LEAKAGE = ['BACILOSC_E', 'CULTURA_ES', 'TEST_MOLEC', 'HISTOPATOL']
REQUIRED_COLUMNS = (NUMERIC_FEATS + BINARY_FEATS
                    + [c for c in CATEGORICAL_FEATS if c not in DERIVED_FEATS]
                    + [OCC_RAW_COL])

# 任务映射（GUI 证据链与评估结果页的展示口径）
TASK_MAPPING = {
    'A_prime': {
        'head': 'bacteriology_pos',
        'label': "任务 A'：细菌学确诊（找传染源）",
        'deployment': '病例发现 / 传染源控制',
    },
    'B': {
        'head': 'death_allcause',
        'label': '任务 B：全因死亡预后',
        'deployment': '治疗结局分层 / 高危随访资源分配',
    },
}

MODEL_VERSION = 'v2'
SCHEMA_ARCHIVE = 'sinan_bact_head_v2_20260826.json'


class DualHeadPredictor:
    """SINAN 双头预测器——单次加载、双头输出。

    用法：
        pred = DualHeadPredictor()      # 懒加载
        pred.load()                       # 或首次 predict 自动加载
        out = pred.predict(df)           # df 含 REQUIRED_COLUMNS 列
        out['bacteriology_risk']          # 任务 A'（找传染源）
        out['death_risk']                 # 任务 B（死亡预后）

    输出键（两套语义名等价）：
        death_risk / task_b_risk          — 全因死亡头
        bacteriology_risk / task_a_prime_risk — 细菌学确诊头
        available                        — 模型工件是否可用
    """

    def __init__(self, base_dir=None):
        self._base = base_dir or self._default_base()
        self._model_dir = os.path.join(self._base, 'tb_risk', 'models',
                                       'sinan_dual_head')
        self._encoder = None
        self._death_head = None
        self._bact_head = None
        self._occ_top = None
        self._loaded = False
        self._load_error = None

    @staticmethod
    def _default_base():
        # core/dual_head_predictor.py → 项目根（tb_risk 的上级）
        return os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

    # ---------------- 加载 ----------------

    def is_available(self):
        """模型工件是否齐备（不实际加载）。"""
        if not HAS_JOBLIB:
            return False
        return all(os.path.exists(os.path.join(
            self._model_dir, f'{name}_{MODEL_VERSION}.joblib'))
            for name in ('encoder', 'death_head', 'bact_head'))

    def load(self):
        """加载 encoder + 双头模型（带 schema 版本核对）。"""
        if self._loaded:
            return True
        if not self.is_available():
            self._load_error = (
                f'模型工件缺失：{self._model_dir}'
                f'（先运行 data/run_sinan_bact_head_v2.py 落盘 v2）')
            LOGGER.warning("双头模型不可用: %s", self._load_error)
            return False
        try:
            self._encoder = joblib.load(os.path.join(
                self._model_dir, f'encoder_{MODEL_VERSION}.joblib'))
            self._death_head = joblib.load(os.path.join(
                self._model_dir, f'death_head_{MODEL_VERSION}.joblib'))
            self._bact_head = joblib.load(os.path.join(
                self._model_dir, f'bact_head_{MODEL_VERSION}.joblib'))
            self._check_schema()
            self._loaded = True
            self._load_error = None
            return True
        except Exception as exc:  # noqa: BLE001 - 产品侧防御
            self._load_error = f'双头模型加载失败：{exc}'
            LOGGER.error("双头模型加载失败: %s", exc, exc_info=True)
            self._encoder = self._death_head = self._bact_head = None
            self._occ_top = None
            return False

    def _check_schema(self):
        """与 v2 归档 JSON 的 feature_schema 对照（版本一致性惯例）。

        v2 的职业截断词表（occupation_top）只存在于归档——缺失时
        加载失败（无词表无法构造 OCC_TRUNC，静默降级会产出错误
        特征）。
        """
        archive = os.path.join(self._base, 'tb_risk', 'data', 'processed',
                               SCHEMA_ARCHIVE)
        if not os.path.exists(archive):
            raise ValueError(
                f'v2 归档缺失：{SCHEMA_ARCHIVE}'
                f'（先运行 data/run_sinan_bact_head_v2.py）')
        with open(archive, encoding='utf-8') as fh:
            schema = (json.load(fh).get('model_artifacts_v2', {})
                      .get('feature_schema', {}))
        if not schema:
            raise ValueError('v2 归档无 model_artifacts_v2.feature_schema')
        expected = {
            'numeric': NUMERIC_FEATS, 'binary': BINARY_FEATS,
            'categorical': CATEGORICAL_FEATS,
            'excluded_leakage': EXCLUDED_LEAKAGE}
        for key, ours in expected.items():
            theirs = schema.get(key)
            if theirs and list(theirs) != list(ours):
                raise ValueError(
                    f'feature_schema 不一致（{key}）：'
                    f'预测器 {ours} vs 归档 {theirs}')
        occ_top = schema.get('occupation_top')
        if not occ_top:
            raise ValueError('v2 归档缺 occupation_top（职业词表）')
        self._occ_top = list(occ_top)

    # ---------------- 预测 ----------------

    def build_features(self, df):
        """从 SINAN DataFrame 构造模型特征矩阵（v2 训练口径）。

        v2 encoder 拟合类别列含 HIV 与派生列 OCC_TRUNC（由
        ID_OCUPA_N 按训练期 top-30 词表截断 + OTHER）。缺失数值列
        报 ValueError；类别列空串替换 'UNK'（OneHotEncoder
        handle_unknown='ignore'）。
        """
        import pandas as pd
        from scipy import sparse

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f'缺少数值特征列: {missing}')

        age = df[NUMERIC_FEATS[0]]
        age_med = age.median() if len(age) else 0
        if pd.isna(age_med):
            age_med = 0.0  # 单行缺失年龄：回退常数（median 全 NaN 边角）
        num = np.column_stack([
            np.clip(pd_numeric(age, age_med), 0, 100),
            np.clip(pd_numeric(df[NUMERIC_FEATS[1]], 0), 0, 10),
        ]).astype(np.float32)

        # OCC_TRUNC：派生列（词表在加载时已核对存在）
        occ = df[OCC_RAW_COL].astype(str).str.strip()
        occ_col = occ.where(occ.isin(self._occ_top or []), 'OTHER')
        # 列顺序必须与 encoder 拟合顺序一致：BINARY + CATEGORICAL
        data = {c: df[c].astype(str).str.strip().replace('', 'UNK')
                for c in BINARY_FEATS}
        for c in CATEGORICAL_FEATS:
            data[c] = (occ_col if c == 'OCC_TRUNC'
                       else df[c].astype(str).str.strip().replace('', 'UNK'))
        cat_df = pd.DataFrame(data)
        if self._encoder is None and not self.load():
            raise RuntimeError(self._load_error or '模型未加载')
        Z = self._encoder.transform(cat_df).astype(np.float32)
        return sparse.hstack([sparse.csr_matrix(num), Z], format='csr')

    def predict(self, df):
        """双头输出：死亡风险 + 细菌学确诊风险。"""
        if not self._loaded and not self.load():
            return {'available': False, 'reason': self._load_error}
        X = self.build_features(df)
        death = self._death_head.predict_proba(X)[:, 1]
        bact = self._bact_head.predict_proba(X)[:, 1]
        return {
            'available': True,
            'death_risk': death.tolist(),
            'bacteriology_risk': bact.tolist(),
            'task_a_prime_risk': bact.tolist(),
            'task_b_risk': death.tolist(),
        }

    def predict_one(self, row):
        """单行预测（dict 或单行 DataFrame）。"""
        import pandas as pd
        df = row if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
        out = self.predict(df)
        if not out.get('available'):
            return out
        return {
            'available': True,
            'death_risk': float(out['death_risk'][0]),
            'bacteriology_risk': float(out['bacteriology_risk'][0]),
        }

    @property
    def load_error(self):
        return self._load_error

    @property
    def task_mapping(self):
        """任务映射（GUI 展示口径）。"""
        return TASK_MAPPING


def pd_numeric(series, fill_value):
    """to_numeric + fillna 的安全封装。"""
    import pandas as pd
    return (pd.to_numeric(series, errors='coerce')
            .fillna(fill_value).values)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cox PH 生存路径：peru_mdr 转正（P5，2026-09-16）

背景（F4 判决，peru_survival_20260915.json）：peru_mdr（MDR 密切接触者，
3406 行 / 149 事件 / 688 户）是 7 队列中唯一的时间-事件结构——
follow_up_days 1–962d，事件中位 153d vs 删失中位 424d，二分类口径扔掉
275 天信息差。B 版（族感知 22+7 维）OOF Harrell C 0.678 > A 版（22 维）
0.629（配对差 +0.049，户级 cluster bootstrap CI [0.014, 0.091]）→
B 版转正为 peru_mdr 训练路径（P5）。显著特征是两个缺失指示器
（index_smear_missing HR=1.25 / index_cough_missing HR=1.24）——
先证诊断信息缺失即风险信号。

本模块 = F4 实验脚本（data/run_peru_survival.py）数值机器的单一真值源
+ 部署能力：
  - CoxSurvivalModel：可持久化 Cox PH（z 标准化 + 零方差/共线列掩码 +
    BFGS 数值防护——F4 三层数值陷阱：11 零方差列致 Hessian 奇异、
    3 对 r=+1.0 共线致秩亏 β 爆炸、Newton 法在稀有二元列准分离下
    hess NaN）；
  - schoenfeld_ph_test：Grambsch-Therneau 比例风险检验——补齐 F4
    自披露缺陷"PH 假设未做正式检验"；
  - train_survival_model：主管线入口（OOF Harrell C + 固定窗 AUROC +
    全量拟合 HR 表 + PH 检验，点估计口径；bootstrap CI 属实验层，
    见 run_peru_survival.py）。

生存路径的判定门（与分类路径互斥的时间轴列）：
  数据含 SURVIVAL_TIME_COLUMNS（follow_up_days）→ 生存路径接管
  （cohort_features.LEAK_EXCLUSIONS 已将其登记为
  excluded_survival_only——分类特征空间排斥时间轴列）。
"""
import logging

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（项目惯例：与 training.py 的 SKLEARN_AVAILABLE 同款）
try:
    from statsmodels.duration.hazard_regression import PHReg
    from scipy.stats import chi2 as _chi2
    from scipy.stats import rankdata as _rankdata
    STATSMODELS_SURVIVAL_AVAILABLE = True
except ImportError:  # pragma: no cover —— 依赖缺失仅禁用本模块
    PHReg = None
    _chi2 = None
    _rankdata = None
    STATSMODELS_SURVIVAL_AVAILABLE = False

# 时间-事件结构探测列（含此列的数据集走生存路径；peru_mdr 是现役唯一）
SURVIVAL_TIME_COLUMNS = ('follow_up_days',)

# 固定窗（天）：6 个月 / 12 个月（与二分类 AUROC 口径可比的删失感知窗）
SURVIVAL_WINDOWS = (180, 365)


# ============================================================
# F4 数值机器（自 data/run_peru_survival.py 迁入，单一真值源）
# ============================================================

def zfit(X_tr):
    """训练折 z 标准化参数（均值/标准差；零方差列除以 1 保护）。"""
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def _live_mask(X_tr):
    """训练折内活列掩码：零方差列剔除后再进 Cox。

    peru_mdr 上 22 维中 11 列 sd=0（合成场景产物，真实队列无方差）——
    常数列 z 标准化后全 0，在 Cox Hessian 中对应零行/零列 → 奇异
    （np.linalg.inv 抛 LinAlgError，首次全量运行即崩于此）。剔除后
    死列系数置 0，不影响线性预测子。
    """
    return X_tr.std(axis=0) > 1e-12


def fit_mask(X_tr, thresh=0.999):
    """拟合列掩码：零方差剔除 + z 标准化后贪心去高共线。

    peru_mdr 上 past_illness 与 highrisk_comorbid r=+1.000000
    （交叉表相同，设计矩阵秩 10/11）——秩亏使 Newton 步病态、β 爆炸，
    exp(xb) 下溢为 0 → MLE 不收敛 → params 含 NaN。贪心保留先出现者，
    |r|>thresh 的后现者剔除。

    返回 (mask, dropped)；dropped[被剔除列序号] = 与之共线的保留列序号。
    """
    live = _live_mask(X_tr)
    idx = np.where(live)[0]
    if len(idx) == 0:
        return live, {}
    Xl = X_tr[:, idx]
    Xz = (Xl - Xl.mean(axis=0)) / np.where(
        Xl.std(axis=0) < 1e-12, 1.0, Xl.std(axis=0))
    keep, dropped = [], {}
    for j in range(len(idx)):
        dup = None
        for k in keep:
            r = np.corrcoef(Xz[:, j], Xz[:, k])[0, 1]
            if np.isfinite(r) and abs(r) > thresh:
                dup = k
                break
        if dup is None:
            keep.append(j)
        else:
            dropped[int(idx[j])] = int(idx[dup])
    mask = np.zeros(X_tr.shape[1], bool)
    mask[idx[keep]] = True
    return mask, dropped


def cox_fit(t, X, e, strata=None):
    """Cox PH（Efron ties）拟合，返回 results。

    method='bfgs'：statsmodels 默认 Newton 法在稀有二元列（peru 上
    dm_tb_synergy 5 阳/1 事件）的准分离似然面下大步跑飞，exp(xb) 下溢
    → hess NaN → MLE 不收敛 → params 含 NaN；BFGS 只用梯度不逆
    Hessian，病态下仍能收敛到有限解。

    strata 非 None 时分层拟合（PH violation 处置，缺陷2）：层特异
    基线风险，分层变量不进线性预测子。
    """
    kw = {'strata': strata} if strata is not None else {}
    return PHReg(t, X, status=e, ties='efron', **kw).fit(method='bfgs')


def _harrell_c_components(t, e, risk):
    """Harrell C 的 (一致对数, 可比对数) 组件（供聚合口径复用）。"""
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    risk = np.asarray(risk, float)
    n = len(t)
    conc = 0.0
    comp = 0
    for i in range(n):
        if not e[i]:
            continue
        # i 是事件者：与所有观察时间更长者可比
        mask = t > t[i]
        if not mask.any():
            continue
        ri = risk[i]
        rj = risk[mask]
        conc += float(np.sum(rj < ri)) + 0.5 * float(np.sum(rj == ri))
        comp += int(mask.sum())
    return conc, comp


def harrell_c(t, e, risk):
    """Harrell 一致性指数（OOF 口径）。

    可比对：较短时间内是事件者 vs 另一人；一致 = 事件者风险更高；
    风险并列计 0.5。时间并列不计入（保守惯例）。
    """
    conc, comp = _harrell_c_components(t, e, risk)
    return (conc / comp) if comp else None


def harrell_c_stratified(t, e, risk, strata):
    """分层 Harrell C：仅同层可比对进入一致性统计。

    分层 Cox 的层特异基线使跨层风险不可比（同一 LP 值在不同层对应
    不同事件概率）——跨层对既不公平也不可解释，统计口径限定同层。
    静态模型在同一对集上评估时同样受限同层对（违反变量同层内为常数，
    不贡献排序），保证 A/B 对比只反映基线异质校正的差异。
    """
    strata = np.asarray(strata)
    if strata.shape[0] != len(t):
        raise ValueError('strata 与 t/e 行不对齐')
    conc, comp = 0.0, 0
    for s in np.unique(strata):
        m = strata == s
        c_s, k_s = _harrell_c_components(t[m], e[m], risk[m])
        conc += c_s
        comp += k_s
    return (conc / comp) if comp else None


def window_label(t, e, W):
    """固定窗 W 的删失感知二分类：窗内事件=1，窗前截尾无事件=移出。"""
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    keep = (e == 1) | (t >= W)
    y = ((e == 1) & (t <= W)).astype(int)[keep]
    return keep, y


def cluster_bootstrap_ci(stat_fn, groups, n_bootstrap, seed, progress_every=0):
    """户级 cluster bootstrap：重采户 → 计统计量 → 百分位 CI。

    stat_fn(idx) 接收样本行索引数组，返回标量（或 None）。
    （validation.cluster_bootstrap_metric_ci 是 auroc/auprc 专用的
    单一真值源；本函数是任意统计量泛化版，生存指标（Harrell C /
    配对差 / HR）经此走户级重采样。）
    progress_every>0 时每 N 次打印进度行（flush，长跑可观测）。
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    stats = []
    for _b in range(n_bootstrap):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_g[g] for g in pick])
        v = stat_fn(idx)
        if v is not None:
            stats.append(v)
        if progress_every and (_b + 1) % progress_every == 0:
            print(f'[bootstrap] {_b + 1}/{n_bootstrap} valid={len(stats)}',
                  flush=True)
    if not stats:
        return None, None, 0
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)], len(stats), float(np.mean(stats))


# ============================================================
# Schoenfeld PH 检验（Grambsch-Therneau score test）
# ============================================================

def schoenfeld_ph_test(res, t, e, names_fitted, transform='log'):
    """比例风险假设检验： scaled Schoenfeld 残差对变换时间的 score test。

    F4 自披露缺陷的补齐（"PH 假设未做正式检验，以固定窗 AUROC 稳定性
    作实用替代证据"）。实现与 R survival::cox.zph 同族（Grambsch &
    Therneau 1994）：

      - statsmodels PHRegResults.schoenfeld_residuals：行对齐拟合样本
        （事件行 = x_k − x̄(t_k) 风险集加权均值，删失行 NaN；tie 处理
        随拟合的 Efron 口径）；
      - 事件 k 的变换时间 g(t_k) 中心化后：
          U = Σ_k (g_k − ḡ) r_k                      （score 向量）
          全局：T = d·UᵀVU / Σ(g−ḡ)² ~ χ²_p          （V = cov_params）
          逐变量：T_j = d·(VU)_j² / (Σ(g−ḡ)²·V_jj) ~ χ²_1
                （= 缩放残差 s_k = d·V·r_k 对 g_k 斜率的 Wald 检验）
      - 拒绝（p<0.05）= 该协变量效应随时间漂移，PH 假设存疑 →
        HR 按时间平均效应解读，固定窗 AUROC 分窗披露。

    参数：
        res: cox_fit 返回的 PHRegResults（其 exog = 掩码后 z 标准化矩阵）
        t, e: 拟合用时间/事件（与 res 行对齐）
        names_fitted: 掩码后拟合列名（res.params 逐分量对齐）
        transform: 'log'（默认，cox.zph 传统）| 'rank' | 'identity'

    返回 dict：{'transform', 'n_events', 'per_variable': [...],
                'global': {'chisq', 'df', 'p_value'}, 'n_violated_0.05'}
    """
    if not STATSMODELS_SURVIVAL_AVAILABLE:
        raise RuntimeError('statsmodels 不可用，PH 检验无法执行')
    sr = np.asarray(res.schoenfeld_residuals, dtype=float)
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    if sr.shape[0] != len(t) or sr.shape[0] != len(e):
        raise ValueError('schoenfeld 残差与 t/e 行不对齐')
    ev = e == 1
    R = sr[ev]
    tt = t[ev]
    # 事件行残差恒有定义（删失行才是 NaN）；防御性再剔一次
    ok = ~np.isnan(R).any(axis=1)
    R, tt = R[ok], tt[ok]
    d, p = R.shape
    if d < 5 or p == 0:
        return {'transform': transform, 'n_events': int(d),
                'per_variable': [], 'global': None,
                'note': '事件数不足（<5），PH 检验无意义'}
    if np.any(tt <= 0) and transform == 'log':
        raise ValueError("transform='log' 要求事件时间 > 0")
    if transform == 'log':
        g = np.log(tt)
    elif transform == 'rank':
        g = _rankdata(tt)
    elif transform == 'identity':
        g = tt.copy()
    else:
        raise ValueError(f'未知 transform: {transform!r}')
    g = g - g.mean()
    denom = float((g ** 2).sum())
    if denom <= 0:
        return {'transform': transform, 'n_events': int(d),
                'per_variable': [], 'global': None,
                'note': '变换时间无离散度，PH 检验退化'}
    U = g @ R                                   # (p,)
    V = np.asarray(res.cov_params(), dtype=float)
    VU = V @ U
    T_global = float(d * (U @ VU) / denom)
    p_global = float(_chi2.sf(T_global, p))
    per_var = []
    n_viol = 0
    for j in range(p):
        vjj = float(V[j, j])
        if not np.isfinite(vjj) or vjj <= 0:
            per_var.append({'feature': names_fitted[j], 'chisq': None,
                            'p_value': None,
                            'note': '方差成分非正，该列检验退化'})
            continue
        Tj = float(d * VU[j] ** 2 / (denom * vjj))
        pj = float(_chi2.sf(Tj, 1))
        if pj < 0.05:
            n_viol += 1
        per_var.append({'feature': names_fitted[j], 'chisq': Tj,
                        'p_value': pj})
    return {
        'transform': transform,
        'n_events': int(d),
        'per_variable': per_var,
        'global': {'chisq': T_global, 'df': int(p), 'p_value': p_global},
        'n_violated_0.05': int(n_viol),
        'interpretation': (
            'Grambsch-Therneau score test（R cox.zph 同族）：拒绝'
            '（p<0.05）= 协变量效应随时间漂移，PH 存疑 → HR 按时间'
            '平均效应解读，判别以分窗 AUROC 披露'),
    }


# ============================================================
# 可持久化 Cox 模型（部署工件）
# ============================================================

class CoxSurvivalModel:
    """Cox PH 部署模型：z 标准化 + 列掩码 + 系数（剔除列置 0）。

    预测输出 = 线性预测子（log-HR，相对训练队列参照）：越高风险越高。
    排序分值而非绝对概率（基线累积风险未估计——绝对风险需基线表，
    判别部署用排序即可；校准需求出现时再扩展）。
    """

    def __init__(self):
        self.feature_names = None    # 全列名（含剔除列，预测端列对齐用）
        self.mu = None               # z 标准化均值（全列长）
        self.sd = None               # z 标准化标准差（零方差列 1.0）
        self.mask = None             # 拟合列掩码（bool，全列长）
        self.beta = None             # 系数（全列长，剔除列 0）
        self.dropped = {}            # {被剔除列名: 共线保留列名}
        self.n_samples = None
        self.n_events = None
        self.time_column = None
        self.cohort = None
        self.fit_method = 'bfgs'

    def fit(self, t, e, X, names, time_column=None, cohort=None):
        """全量拟合（数值防护见 fit_mask/cox_fit 的 F4 陷阱记录）。"""
        if not STATSMODELS_SURVIVAL_AVAILABLE:
            raise RuntimeError('statsmodels 不可用，Cox 拟合无法执行')
        t = np.asarray(t, float)
        e = np.asarray(e, int)
        X = np.asarray(X, float)
        if X.shape[0] != len(t) or len(t) != len(e):
            raise ValueError('X/t/e 行不对齐')
        if len(np.unique(e)) < 2:
            raise ValueError('事件指示只有一类，Cox 无法拟合')
        mu, sd = zfit(X)
        Xs = (X - mu) / sd
        mask, dropped_idx = fit_mask(X)
        names = list(names)
        res = cox_fit(t, Xs[:, mask], e)
        beta = np.zeros(X.shape[1])
        beta[mask] = res.params
        self.feature_names = names
        self.mu = mu
        self.sd = sd
        self.mask = mask
        self.beta = beta
        self.dropped = {names[j]: names[k] for j, k in dropped_idx.items()}
        self.n_samples = int(X.shape[0])
        self.n_events = int(e.sum())
        self.time_column = time_column
        self.cohort = cohort
        self._res = res               # 拟合结果（内存态，不落盘）
        return res

    @property
    def results(self):
        """底层 PHRegResults（PH 检验/HR 推断用；落盘后为 None）。"""
        return getattr(self, '_res', None)

    def predict(self, X):
        """线性预测子（log-HR）：((X−mu)/sd) @ beta，剔除列贡献 0。"""
        if self.beta is None:
            raise RuntimeError('模型未拟合')
        X = np.asarray(X, float)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f'特征维度 {X.shape[1]} != 训练 {len(self.feature_names)}')
        return ((X - self.mu) / self.sd) @ self.beta

    def to_dict(self):
        """序列化（_res 不落盘——系数/掩码已完整携带预测所需全部信息）。"""
        return {
            'feature_names': list(self.feature_names),
            'mu': np.asarray(self.mu).tolist(),
            'sd': np.asarray(self.sd).tolist(),
            'mask': np.asarray(self.mask, bool).tolist(),
            'beta': np.asarray(self.beta).tolist(),
            'dropped': dict(self.dropped),
            'n_samples': self.n_samples,
            'n_events': self.n_events,
            'time_column': self.time_column,
            'cohort': self.cohort,
            'fit_method': self.fit_method,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.feature_names = list(d['feature_names'])
        m.mu = np.asarray(d['mu'], float)
        m.sd = np.asarray(d['sd'], float)
        m.mask = np.asarray(d['mask'], bool)
        m.beta = np.asarray(d['beta'], float)
        m.dropped = dict(d.get('dropped') or {})
        m.n_samples = d.get('n_samples')
        m.n_events = d.get('n_events')
        m.time_column = d.get('time_column')
        m.cohort = d.get('cohort')
        m.fit_method = d.get('fit_method', 'bfgs')
        return m


# ============================================================
# 主管线入口（点估计口径；bootstrap CI 属实验层）
# ============================================================

def train_survival_model(t, e, X, names, groups=None, folds=5, seed=42,
                         time_column=None, cohort=None, windows=None):
    """生存路径训练：OOF 判别 + 全量拟合（部署工件）+ HR 表 + PH 检验。

    口径对齐 F4（data/run_peru_survival.py，peru_survival_20260915.json）：
      - CV：groups 非空 → StratifiedGroupKFold（户感知，事件分层），
        否则 StratifiedKFold；
      - 每折：折内 z 标准化 + 折内列掩码 + Cox(BFGS) → OOF 线性预测子；
      - 判别：OOF Harrell C + 固定窗 AUROC（180d/365d，删失感知）点估计；
      - 全量拟合 → CoxSurvivalModel（部署工件，随 checkpoint 落盘）；
      - HR 表：exp(β) 每 1 SD 口径 + 渐近 CI + 零方差/共线剔除登记 + EPV；
      - PH 检验：schoenfeld_ph_test（transform='log'）。

    返回 dict（'model' 键 = CoxSurvivalModel，调用方挂 predictor）。
    """
    if not STATSMODELS_SURVIVAL_AVAILABLE:
        return {'success': False,
                'error_type': 'statsmodels_unavailable',
                'note': 'statsmodels 不可用，生存路径跳过'}
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    X = np.asarray(X, float)
    names = list(names)
    windows = list(windows or SURVIVAL_WINDOWS)
    n_events = int(e.sum())
    if n_events < 10:
        return {'success': False, 'error_type': 'too_few_events',
                'n_events': n_events,
                'note': f'事件 {n_events} < 10，生存路径跳过（EPV 无意义）'}

    # ---- OOF CV ----
    if groups is not None:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                  random_state=seed)
        fold_iter = cv.split(X, e, groups)
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_iter = cv.split(X, e)
    oof = np.full(len(t), np.nan)
    for tr, te in fold_iter:
        mu, sd = zfit(X[tr])
        mask, _ = fit_mask(X[tr])
        Xtr = (X[tr] - mu) / sd
        res = cox_fit(t[tr], Xtr[:, mask], e[tr])
        beta = np.zeros(X.shape[1])
        beta[mask] = res.params
        oof[te] = ((X[te] - mu) / sd) @ beta
    if np.isnan(oof).any():
        return {'success': False, 'error_type': 'oof_incomplete',
                'note': 'OOF 有未覆盖样本（折分配异常）'}

    c_oof = harrell_c(t, e, oof)
    windows_out = {}
    for W in windows:
        keep, y_w = window_label(t, e, W)
        if len(np.unique(y_w)) < 2:
            windows_out[f'auroc_{W}d'] = None
            continue
        from sklearn.metrics import roc_auc_score
        windows_out[f'auroc_{W}d'] = {
            'point': float(roc_auc_score(y_w, oof[keep])),
            'n_at_risk': int(keep.sum()),
            'n_events_in_window': int(y_w.sum()),
            'note': ('窗前截尾无事件者移出风险集（删失感知）；窗后事件'
                     '按非事件计（与二分类终点口径的差异披露）')}

    # ---- 全量拟合（部署工件）+ HR 表 + PH 检验 ----
    model = CoxSurvivalModel()
    res_full = model.fit(t, e, X, names,
                         time_column=time_column, cohort=cohort)
    mask_idx = np.where(model.mask)[0]
    names_fitted = [names[j] for j in mask_idx]
    ci_asym = np.exp(res_full.conf_int())
    pvals = res_full.pvalues
    table = []
    for j, name in enumerate(names):
        if not model.mask[j]:
            if name in model.dropped:
                table.append({
                    'feature': name, 'status': 'collinear_dropped',
                    'note': (f'与 {model.dropped[name]} |r|>0.999'
                             '（全数据）：完全共线，剔除保秩')})
            else:
                table.append({
                    'feature': name, 'status': 'zero_variance_in_full_data',
                    'note': '全数据 sd=0（合成场景产物）：常数列，无信息'})
            continue
        k = int(np.where(mask_idx == j)[0][0])
        table.append({
            'feature': name, 'status': 'live',
            'coef_per_sd': float(model.beta[j]),
            'HR_per_sd': float(np.exp(model.beta[j])),
            'HR_ci95_asymptotic': [float(ci_asym[k, 0]),
                                   float(ci_asym[k, 1])],
            'p_value': float(pvals[k])})
    epv = n_events / len(names_fitted)
    ph = schoenfeld_ph_test(res_full, t, e, names_fitted, transform='log')

    return {
        'success': True,
        'protocol': {
            'model': 'Cox PH（statsmodels PHReg，Efron ties，BFGS）',
            'cv': (f'{"StratifiedGroupKFold" if groups is not None else "StratifiedKFold"}'
                   f'({folds})，事件分层，seed={seed}'),
            'standardization': '连续列训练折内 z 标准化 → HR 为每 1 SD 口径',
            'note': ('点估计口径（主管线）；cluster bootstrap CI 与 A/B 配对'
                     '差属实验层（data/run_peru_survival.py，'
                     'peru_survival_20260915.json）'),
        },
        'n_samples': int(len(t)),
        'n_events': n_events,
        'n_features': len(names),
        'n_features_fitted': len(names_fitted),
        'events_per_variable': round(epv, 2),
        'epv_note': (f'EPV={n_events}/{len(names_fitted)} 拟合参数（'
                     f'{len(names_fitted)}/{len(names)} 列进模型，零方差/'
                     '共线列剔除）：EPV<10 时 HR 表按探索性解读'),
        'harrell_c_oof': c_oof,
        'fixed_window': windows_out,
        'hr_table': table,
        'ph_test': ph,
        'model': model,
    }


# ============================================================
# PH 违反处置（缺陷2，2026-09-17）：违反变量作层的重拟合对比
# ============================================================

def _stratum_id(X, strata_idx):
    """二元分层列 → 单一层 id（位组合；列序即位权）。"""
    sid = np.zeros(X.shape[0], dtype=int)
    for j in strata_idx:
        col = np.asarray(X[:, j], float)
        sid = sid * 2 + (col > 0.5).astype(int)
    return sid


def stratified_cox_refit(t, e, X, names, strata_vars, groups=None,
                          folds=5, seed=42, n_bootstrap=0,
                          bootstrap_seed=20260917, progress_every=0):
    """PH 违反处置实验：静态 Cox vs 分层 Cox 的同对集 OOF Harrell C。

    设计（与 R survival::cox.zph 拒绝后的标准处置对齐）：
      - PH 违反的二元协变量（t7 run：has_tb 0.0114 / is_high_risk
        0.0245 / past_illness 0.0478）从线性预测子移出、作分层因子
        ——时间漂移效应被层特异基线风险吸收，不再时间平均化；
      - 对比对集：分层模型跨层基线不同 → 仅同层可比对有意义；静态
        模型限同一对集（违反变量同层内为常数、不贡献同层排序），
        A/B 差异纯来自非分层系数的基线异质校正；
      - 两组 OOF 用同一折结构（StratifiedGroupKFold by 户，事件分层，
        seed 固定），与 train_survival_model 口径一致；
      - n_bootstrap>0 时户级 cluster bootstrap 给 ΔC 的 95% CI
        （每个重采样内重跑两组 OOF，与 F4 判决同款）。

    判决规则（预注册）：ΔC>0 且 bootstrap CI 下界>0 → 分层版转正；
    否则静态版维持（HR 按时间平均效应解读 + 固定窗分窗披露，现状）。

    Returns:
        dict: strata_vars / per_stratum_events / c_static_within /
              c_stratified_within / delta_c / delta_c_ci95 /
              c_static_full_pairs / ph_retest / protocol
    """
    if not STATSMODELS_SURVIVAL_AVAILABLE:
        raise RuntimeError('statsmodels 不可用，分层重拟合无法执行')
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    X = np.asarray(X, float)
    names = list(names)
    strata_vars = list(strata_vars)
    if not strata_vars:
        raise ValueError('strata_vars 为空')
    unknown = [v for v in strata_vars if v not in names]
    if unknown:
        raise ValueError(f'strata_vars 不在特征名中: {unknown}')
    strata_idx = [names.index(v) for v in strata_vars]
    for v, j in zip(strata_vars, strata_idx):
        vals = np.unique(X[:, j])
        if not np.all(np.isin(vals, [0.0, 1.0])):
            raise ValueError(
                f'分层变量 {v} 非二元（unique={vals[:5]}）——分层处置'
                '仅适用于分类/二元协变量')
    sid = _stratum_id(X, strata_idx)
    keep_idx = [j for j in range(len(names)) if j not in strata_idx]
    # 分层设计的共线陷阱（实测：peru highrisk_comorbid↔past_illness
    # r=+1.0）：静态臂由 fit_mask 剔其一；分层臂把 past_illness 移出作
    # 层后，其共线副本反而入场——该列在层内近常数 → Hessian 奇异
    # （BFGS 收敛但 cov_params 不可逆）。分层臂额外剔除与任一分层
    # 变量 |r|>0.999 的协变量并登记披露。
    strata_collinear = []
    for j in list(keep_idx):
        for s in strata_idx:
            r = np.corrcoef(X[:, j], X[:, s])[0, 1]
            if np.isfinite(r) and abs(r) > 0.999:
                strata_collinear.append(
                    {'feature': names[j], 'stratum_var': names[s],
                     'note': '与分层变量 |r|>0.999：层内近常数，剔除保秩'})
                keep_idx.remove(j)
                break
    names_keep = [names[j] for j in keep_idx]

    per_stratum = []
    for s in np.unique(sid):
        m = sid == s
        per_stratum.append({
            'stratum': {v: int((s >> (len(strata_idx) - 1 - k)) & 1)
                        for k, v in enumerate(strata_vars)},
            'n': int(m.sum()),
            'n_events': int(e[m].sum()),
        })

    def _oof_both(tt, ee, XX, ss, gg):
        """同一折结构下两组 OOF 线性预测子 → 同层对集 C 与 ΔC。"""
        if gg is not None:
            from sklearn.model_selection import StratifiedGroupKFold
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                      random_state=seed)
            split = cv.split(XX, ee, gg)
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=folds, shuffle=True,
                                 random_state=seed)
            split = cv.split(XX, ee)
        oof_static = np.full(len(tt), np.nan)
        oof_strat = np.full(len(tt), np.nan)
        for tr, te in split:
            # 静态臂：全列进协变量（与 train_survival_model 同构）
            mu, sd = zfit(XX[tr])
            mask, _ = fit_mask(XX[tr])
            Xtr = (XX[tr] - mu) / sd
            res = cox_fit(tt[tr], Xtr[:, mask], ee[tr])
            beta = np.zeros(XX.shape[1])
            beta[mask] = res.params
            oof_static[te] = ((XX[te] - mu) / sd) @ beta
            # 分层臂：违反变量移出协变量、作层
            Xk_tr, Xk_te = XX[tr][:, keep_idx], XX[te][:, keep_idx]
            mu_k, sd_k = zfit(Xk_tr)
            mask_k, _ = fit_mask(Xk_tr)
            Xktr = (Xk_tr - mu_k) / sd_k
            res_k = cox_fit(tt[tr], Xktr[:, mask_k], ee[tr],
                            strata=ss[tr])
            beta_k = np.zeros(len(keep_idx))
            beta_k[mask_k] = res_k.params
            oof_strat[te] = ((Xk_te - mu_k) / sd_k) @ beta_k
        if np.isnan(oof_static).any() or np.isnan(oof_strat).any():
            return None
        c_a = harrell_c_stratified(tt, ee, oof_static, ss)
        c_b = harrell_c_stratified(tt, ee, oof_strat, ss)
        if c_a is None or c_b is None:
            return None
        return {'c_static': c_a, 'c_stratified': c_b,
                'delta_c': c_b - c_a}

    main = _oof_both(t, e, X, sid,
                     np.asarray(groups) if groups is not None else None)
    if main is None:
        return {'success': False,
                'error_type': 'oof_incomplete',
                'note': 'OOF 有未覆盖样本或无同层可比对'}

    out = {
        'success': True,
        'strata_vars': strata_vars,
        'n_strata': len(per_stratum),
        'per_stratum_events': per_stratum,
        'strata_collinear_dropped': strata_collinear,
        'c_static_within_strata_pairs': main['c_static'],
        'c_stratified_within_strata_pairs': main['c_stratified'],
        'delta_c': main['delta_c'],
        'n_bootstrap': int(n_bootstrap),
        'protocol': {
            'cv': (f'{"StratifiedGroupKFold" if groups is not None else "StratifiedKFold"}'
                   f'({folds})，事件分层，seed={seed}'),
            'comparison_set': ('同层可比对（分层模型跨层基线不同不可比；'
                               '静态模型限同一对集保证公平）'),
            'decision_rule': ('ΔC>0 且 bootstrap CI 下界>0 → 分层版转正；'
                              '否则静态版维持（HR 时间平均效应解读 + '
                              '固定窗分窗披露）'),
            'full_pairs_anchor_note': ('全对集静态 C（t7 harrell_c_oof='
                                       '0.6777）由主管线档案引用，本函数'
                                       '不重复计算——同层对集 C 与其口径'
                                       '不同，不可直接对比'),
        },
    }

    if n_bootstrap and groups is not None:
        groups_arr = np.asarray(groups)

        def _stat(idx):
            try:
                r = _oof_both(t[idx], e[idx], X[idx], sid[idx],
                              groups_arr[idx])
                return r['delta_c'] if r else None
            except Exception:
                return None

        ci, n_ok, mean = cluster_bootstrap_ci(
            _stat, groups_arr, n_bootstrap, bootstrap_seed,
            progress_every=progress_every)
        out['delta_c_ci95'] = ci
        out['delta_c_bootstrap_mean'] = mean
        out['n_bootstrap_valid'] = n_ok

    # PH 复检：分层拟合后剩余协变量的 PH 检验（违反变量已作层）
    try:
        Xk = X[:, keep_idx]
        mu_k, sd_k = zfit(Xk)
        mask_k, _ = fit_mask(Xk)
        res_k = cox_fit(t, ((Xk - mu_k) / sd_k)[:, mask_k], e, strata=sid)
        out['ph_retest'] = schoenfeld_ph_test(
            res_k, t, e,
            [names_keep[j] for j in np.where(mask_k)[0]],
            transform='log')
    except Exception as exc:  # statsmodels schoenfeld 对 strata 的支持
        out['ph_retest'] = {'error': str(exc),
                            'note': '分层拟合的 PH 复检不可用（残差实现'
                                    '限制），以固定窗 AUROC 稳定性替代'}
    return out

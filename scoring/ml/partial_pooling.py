#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多队列部分池化：cohort 随机截距 + 共享斜率的混合效应逻辑回归。

任务2（2026-09-14，用户批准范围）：7 个队列（nhanes/treats/crp/kenya/
taiwan/brazil/peru_mdr）此前各自独立训练，小队列（Taiwan 129 /
Brazil 202）斜率估计方差大。本模块用 statsmodels
BinomialBayesMixedGLM（变分贝叶斯）做部分池化：

    logit P(y=1) = α + βᵀx + u_{cohort},  u_{cohort} ~ N(0, σ²_α)

- 共享斜率 β：大队列（Kenya 63k）主导估计，小队列借力先验；
- cohort 随机截距 u：吸收各队列基线风险差异（患病率、终点定义、
  人群结构），shrunk BLUP（经验贝叶斯收缩）；
- 预测三种模式：
    1. 已知训练内 cohort → α + u_{cohort}（shrunk 截距）；
    2. cohort=None → 总体截距 α（零样本臂，跨端点部署仅作方向
       对照——项目硬约束：零样本禁止部署到确诊终点人群）；
    3. recalibrate_intercept_locally → 本地截距替换 α、斜率冻结
       （跨端点部署的许可路径：微调/重校准型迁移）。

端点异质性披露（全 7 队列混合池化为用户决策）：
    LTBI 族（nhanes/treats/brazil：IGRA/QFT 标签）vs
    确诊/活动性 TB 族（kenya/crp/taiwan）vs
    incident TB（peru_mdr）——随机截距吸收主差异，但斜率可能被
    终点差异污染；LOCO 评估（留出队列零样本/重校准）是唯一诚实
    的泛化证据，within-cohort CV 不回答跨队列问题。

依赖：statsmodels>=0.13.5（pyproject [ml]/[full]）；缺失时
STATSMODELS_AVAILABLE=False，fit 抛 RuntimeError（调用方决定降级）。
"""
import logging

import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")

# 条件导入（项目惯例：与 training.py 的 SKLEARN_AVAILABLE 等同款）
try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    import statsmodels.api as _sm_api
    import scipy.sparse as _sp
    from sklearn.preprocessing import StandardScaler
    STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover —— 依赖缺失仅禁用本模块
    BinomialBayesMixedGLM = None
    _sm_api = None
    _sp = None
    StandardScaler = None
    STATSMODELS_AVAILABLE = False


def _sigmoid_stable(z):
    """数值稳定 sigmoid（大负参直接 exp 溢出，分段计算避免 RuntimeWarning）。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _moment_match_intercept(offset, y, tol=1e-10, max_iter=200,
                            bound=200.0):
    """截距-only + offset Logit 的得分方程二分解（MLE 等价回退）。

    得分方程：mean(y) = mean(sigmoid(a + offset))。左边固定于 (0,1)、
    右边对 a 严格单调连续（极限 0→1）→ 唯一有限根恒存在。statsmodels
    Newton 拟合在 offset 饱和（全部 p≈0/1）时 Hessian 奇异崩溃
    （族内分池 2026-09-15 实测：treats 族内模型在本地半区触发），
    二分法不受饱和影响，且与 MLE 同解（同一得分方程）。
    """
    ybar = float(np.mean(y))
    offset = np.asarray(offset, dtype=float)

    def score(a):
        return ybar - float(np.mean(_sigmoid_stable(a + offset)))

    lo, hi = -bound, bound
    # score 单调递减：扩 bracket（理论极限 ±∞，bound 内必有根）
    while score(lo) <= 0 and lo > -1e9:
        lo *= 2.0
    while score(hi) >= 0 and hi < 1e9:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


class CohortMixedModel:
    """cohort 随机截距 + 共享斜率的混合效应逻辑回归（VB 拟合）。

    内部 StandardScaler（22 维含 age/暴露天数等异尺度列，未标准化的
    VB 拟合易不收敛）；scaler 随对象持久化，预测/本地重校准复用同一
    变换（口径一致）。
    """

    def __init__(self, feature_names=None):
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.scaler = None
        self.fe_mean = None          # 固定效应后验均值 [截距, 斜率...]
        self.fe_sd = None
        self.vcp_mean = None         # 方差成分参数（对数尺度）
        self.vc_mean = {}            # {str(cohort): BLUP}（str 键规范化）
        self.cohort_names = []
        self.fit_method = None
        self.n_samples = None
        self.local_intercept = None  # 本地重校准截距（None=未重校准）
        self.local_recal_note = None  # 'logit_mle' | 'moment_match_fallback(...)'
        self.convergence_note = None
        self.elbo_selected = None    # 选中起点的 ELBO（多种子选优依据）
        self.elbo_all_starts = []    # 各起点 ELBO（稳定性披露）
        self.starts_agreement = None # 起点间 fe_mean 最大差（稳定性披露）

    # ------------------------------------------------------------------
    # 拟合
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_one_vb(model, seed):
        """单起点 VB：种子隔离 + 警告捕获。

        statsmodels fit_vb 的后验 SD 起点用未设种子的全局
        np.random.normal（bayes_mixed_glm.py L759）→ 同一数据两次拟合
        可落入不同盆地（S7 冒烟实测：brazil LOCO 两次 AUROC 0.40 vs
        0.59）。本方法保存/恢复全局 RNG 状态 + 显式播种，使单起点
        拟合可复现且不污染调用方随机流。

        返回：
            (res, non_converged: bool)；异常向上抛（由调用方回退 MAP）。
        """
        import warnings as _warnings
        rng_state = np.random.get_state()
        try:
            np.random.seed(seed)
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter('always')
                res = model.fit_vb(verbose=False)
            non_conv = any('did not converge' in str(w.message).lower()
                           for w in caught)
            return res, non_conv
        finally:
            np.random.set_state(rng_state)

    def _elbo_of(self, model, res):
        """拟合后补算 ELBO（多种子选优准则）。

        fit_vb 内部目标 = -vb_elbo(mean, sd)，结果对象按 fep/vcp/vc 分块
        暴露后验均值与 SD——重拼全向量即可复算同一口径的 ELBO。
        """
        try:
            mean_vec = np.concatenate(
                [np.asarray(res.fe_mean), np.asarray(res.vcp_mean),
                 np.asarray(res.vc_mean)])
            sd_vec = np.concatenate(
                [np.asarray(res.fe_sd), np.asarray(res.vcp_sd),
                 np.asarray(res.vc_sd)])
            return float(model.vb_elbo(mean_vec, sd_vec))
        except Exception:  # pragma: no cover - 补算失败不影响主流程
            return None

    def fit(self, X, y, cohort, fit_method='vb', random_state=42, n_starts=3):
        """拟合混合效应模型。

        参数：
            X: (n, p) 特征矩阵（原始尺度，内部标准化）
            y: (n,) 0/1 标签
            cohort: (n,) 队列标识（任意 hashable，内部 str 规范化）
            fit_method: 'vb'（变分贝叶斯，主）/ 'map'（Laplace 近似，回退）
            random_state (int): 起点随机种子基准（statsmodels 起点
                未播种的确定性修复；多种子起点 = random_state + k）
            n_starts (int): VB 多起点数（ELBO 选优 + 稳定性披露；
                1 = 单起点旧行为）

        返回：
            self（链式）；失败抛 RuntimeError。
        """
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError('statsmodels 不可用，无法拟合混合效应模型')

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        cohort = np.asarray([str(c) for c in np.asarray(cohort).reshape(-1)])
        n = len(y)
        if len(X) != n or len(cohort) != n:
            raise ValueError('X/y/cohort 长度不一致')
        if len(np.unique(y)) < 2:
            raise ValueError('标签只有一类，无法拟合')
        if n < 20:
            raise ValueError(f'样本量过少（{n} < 20），混合效应估计不稳定')

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        exog = np.column_stack([np.ones(n), Xs])  # 首列截距

        uniq, cohort_idx = np.unique(cohort, return_inverse=True)
        G = _sp.csc_matrix(
            (np.ones(n), (np.arange(n), cohort_idx)),
            shape=(n, len(uniq)))
        model = BinomialBayesMixedGLM(
            y, exog, exog_vc=G, ident=np.zeros(len(uniq), dtype=int),
            vcp_names=['cohort_intercept'],
            vc_names=[str(c) for c in uniq])

        res = None
        if fit_method == 'vb':
            # 多起点 VB：ELBO 选优（变分局部最优的标准补救）；
            # 起点间 fe_mean 最大差作为稳定性披露（大差 = 结果盆地敏感，
            # 须谨慎引用）
            candidates = []
            for k in range(max(1, int(n_starts))):
                try:
                    r, non_conv = self._fit_one_vb(model, int(random_state) + k)
                except Exception as e:  # pragma: no cover - 单起点失败继续
                    LOGGER.warning('fit_vb 起点 %d 失败: %s', k, e)
                    continue
                elbo = self._elbo_of(model, r)
                candidates.append({'res': r, 'non_conv': non_conv, 'elbo': elbo})
            scored = [c for c in candidates if c['elbo'] is not None]
            if scored:
                best = max(scored, key=lambda c: c['elbo'])
                res = best['res']
                self.fit_method = 'vb'
                self.elbo_all_starts = [c['elbo'] for c in candidates
                                        if c['elbo'] is not None]
                self.elbo_selected = best['elbo']
                if len(scored) > 1:
                    fe_all = [np.asarray(c['res'].fe_mean) for c in scored]
                    self.starts_agreement = float(max(
                        np.max(np.abs(fe_all[i] - fe_all[j]))
                        for i in range(len(fe_all))
                        for j in range(i + 1, len(fe_all))))
                notes = []
                if best['non_conv']:
                    notes.append('vb_non_converged')
                else:
                    notes.append('vb_converged')
                if self.starts_agreement is not None and self.starts_agreement > 0.25:
                    notes.append(f'multi_start_unstable(fe_max_diff='
                                 f'{self.starts_agreement:.3f})')
                self.convergence_note = '; '.join(notes)
                if 'vb_non_converged' in self.convergence_note:
                    LOGGER.warning('VB 未收敛（结果须谨慎引用）')
                if 'multi_start_unstable' in self.convergence_note:
                    LOGGER.warning('VB 多起点不稳定（fe_max_diff=%.3f）——'
                                   '结果盆地敏感', self.starts_agreement)
            elif candidates:
                # 有候选但 ELBO 均补算失败：取首个（兼容旧行为）
                res = candidates[0]['res']
                self.fit_method = 'vb'
                self.convergence_note = ('vb_elbo_unavailable'
                                         + ('; vb_non_converged'
                                            if candidates[0]['non_conv'] else ''))
            else:
                LOGGER.warning('fit_vb 全部起点失败，回退 fit_map')
        if res is None:
            rng_state = np.random.get_state()
            try:
                np.random.seed(random_state)
                res = model.fit_map()
            finally:
                np.random.set_state(rng_state)
            self.fit_method = 'map'
            self.convergence_note = 'map_fallback'

        self.fe_mean = np.asarray(res.fe_mean, dtype=float)
        self.fe_sd = np.asarray(res.fe_sd, dtype=float)
        self.vcp_mean = np.asarray(res.vcp_mean, dtype=float)
        self.vc_mean = {str(c): float(u)
                        for c, u in zip(uniq, np.asarray(res.vc_mean))}
        self.cohort_names = [str(c) for c in uniq]
        self.n_samples = int(n)
        self.local_intercept = None

        LOGGER.info('混合效应模型拟合完成: method=%s, n=%d, cohorts=%d, '
                    '随机截距 sd=%.4f, conv=%s',
                    self.fit_method, n, len(uniq),
                    float(np.exp(self.vcp_mean[0])), self.convergence_note)
        return self

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def _linear_predictor(self, X, cohort=None):
        """线性预测子：截距（总体/本地重校准）+ 共享斜率 + 可选 BLUP。"""
        if self.fe_mean is None:
            raise RuntimeError('模型未拟合')
        X = np.asarray(X, dtype=float)
        Xs = self.scaler.transform(X)
        # 截距：本地重校准值优先，否则总体截距 fe_mean[0]
        base = (self.local_intercept if self.local_intercept is not None
                else float(self.fe_mean[0]))
        eta = base + Xs @ self.fe_mean[1:]
        if cohort is not None:
            key = str(cohort)
            if key in self.vc_mean:
                eta = eta + self.vc_mean[key]
            # 未知 cohort（零样本）→ 无 BLUP，总体截距即预测
        return eta

    def predict_proba(self, X, cohort=None):
        """预测 P(y=1)。

        cohort=训练内队列 → shrunk BLUP 截距；cohort=None 或未知队列
        → 总体截距（零样本臂——跨端点仅作方向对照，禁止部署）；
        已 recalibrate_intercept_locally → 本地截距 + 冻结斜率。
        """
        eta = self._linear_predictor(X, cohort=cohort)
        return 1.0 / (1.0 + np.exp(-eta))

    # ------------------------------------------------------------------
    # 本地重校准（跨端点部署的许可路径）
    # ------------------------------------------------------------------

    def recalibrate_intercept_locally(self, X_local, y_local):
        """斜率冻结、只重估截距的本地重校准。

        实现：Logit(y, [1], offset=X_std @ β)——offset 把共享斜率的
        线性贡献固定，唯一自由参数为新截距。满足项目硬约束
        （跨终点部署前必须本地重校准；与 Kenya 预训练→微调同属
        许可路径，零样本臂不是）。

        返回：
            float: 重校准后的截距（并存 self.local_intercept，
            后续 predict_proba 自动使用）。
        """
        if self.fe_mean is None:
            raise RuntimeError('模型未拟合')
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError('statsmodels 不可用')
        X = np.asarray(X_local, dtype=float)
        y = np.asarray(y_local, dtype=int).reshape(-1)
        if len(np.unique(y)) < 2:
            raise ValueError('本地标签只有一类，无法重校准截距')
        Xs = self.scaler.transform(X)
        offset = Xs @ self.fe_mean[1:]  # 斜率贡献冻结为 offset
        try:
            res = _sm_api.Logit(y, np.ones((len(y), 1)),
                                offset=offset).fit(disp=0)
            self.local_intercept = float(res.params[0])
            self.local_recal_note = 'logit_mle'
        except Exception as e:  # Hessian 奇异等数值失败 → 矩匹配二分回退
            # （与 MLE 同解：同一得分方程；唯一有限根恒存在，见函数 docstring）
            LOGGER.warning('Logit 截距重校准数值失败（%s），回退矩匹配二分解', e)
            self.local_intercept = _moment_match_intercept(offset, y)
            self.local_recal_note = f'moment_match_fallback({type(e).__name__})'
        LOGGER.info('本地截距重校准完成: %.4f → %.4f（斜率冻结，%s）',
                    float(self.fe_mean[0]), self.local_intercept,
                    self.local_recal_note)
        return self.local_intercept

    # ------------------------------------------------------------------
    # 摘要 / 披露
    # ------------------------------------------------------------------

    def summary_dict(self):
        """结构化摘要（截距方差、各队列 BLUP、端点披露、部署模式）。"""
        if self.fe_mean is None:
            return {'fitted': False}
        return {
            'fitted': True,
            'fit_method': self.fit_method,
            'convergence_note': self.convergence_note,
            'elbo_selected': self.elbo_selected,
            'elbo_all_starts': self.elbo_all_starts,
            'starts_agreement_fe_max_diff': self.starts_agreement,
            'n_samples': self.n_samples,
            'n_features': int(len(self.fe_mean) - 1),
            'feature_names': self.feature_names,
            'population_intercept': float(self.fe_mean[0]),
            'random_intercept_sd': float(np.exp(self.vcp_mean[0])),
            'vcp_log_scale': float(self.vcp_mean[0]),
            'cohort_blups': dict(sorted(self.vc_mean.items())),
            'fixed_slopes': ({
                name: float(b) for name, b in
                zip((self.feature_names or
                     [f'x{i}' for i in range(len(self.fe_mean) - 1)]),
                    self.fe_mean[1:])}
                if self.feature_names or len(self.fe_mean) > 1 else {}),
            'local_intercept': self.local_intercept,
            'local_recal_note': self.local_recal_note,
            'deployment_modes': {
                'known_cohort': 'population_intercept + cohort BLUP（shrunk）',
                'zero_shot': ('population_intercept only——跨终点仅作方向'
                              '对照，禁止部署（项目硬约束）'),
                'local_recalibrated': ('local_intercept + 冻结斜率——'
                                       '跨端点部署许可路径'),
            },
            'endpoint_heterogeneity_note': (
                '全 7 队列混合池化（用户决策）：LTBI 族（nhanes/treats/'
                'brazil）与确诊/活动性 TB 族（kenya/crp/taiwan）及 '
                'incident TB（peru_mdr）共用斜率；随机截距吸收基线差异'
                '但斜率可能被终点差异污染——LOCO 评估是唯一诚实证据'),
        }


def fit_cohort_mixed_logistic(X, y, cohort, feature_names=None,
                              fit_method='vb', random_state=42, n_starts=3):
    """模块级入口：拟合 cohort 随机截距 + 共享斜率混合模型。

    返回：
        CohortMixedModel（已拟合）
    """
    model = CohortMixedModel(feature_names=feature_names)
    return model.fit(X, y, cohort, fit_method=fit_method,
                     random_state=random_state, n_starts=n_starts)

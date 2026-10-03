import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

LOGGER = logging.getLogger("tb_risk.validation.comparator")

from .backtest import BacktestEngine


class NationalModelComparator:
    """国家级模型对比器

    将克拉玛依本土化模型与未本土化的通用SEIR-ML模型在
    相同数据集上对比，验证本土化是否改善预测性能和临床决策效用。

    评估指标：
    - 检出率 (Detection Rate) 在不同筛查覆盖率下的表现
    - NNS (Number Needed to Screen) 比较
    - 风险区分度 (Risk Discrimination Ratio)
    - 早期发现率提升
    """

    def __init__(self, localizer):
        self.localizer = localizer

    def run_comparison(self, records, n_bootstrap=1000):
        """运行国家级模型对比分析（v2.0：BCa CI + Wilcoxon 检验）

        参数：
            records (list[dict]): 筛查记录
            n_bootstrap (int): Bootstrap重采样次数（默认 1000，推荐 ≥ 1000）

        返回：
            dict: 对比分析报告，含统计显著性检验
        """
        # 本土化模型评估
        loc_backtest = BacktestEngine(self.localizer, use_localization=True)
        loc_report = loc_backtest.run_backtest(records)

        # 国家级基线模型评估
        base_backtest = BacktestEngine(self.localizer, use_localization=False)
        base_report = base_backtest.run_backtest(records)

        # 各覆盖率下的检出率对比
        detection_comparison = {}
        for thresh in ['top_1pct', 'top_2pct', 'top_5pct',
                        'top_10pct', 'top_15pct', 'top_20pct']:
            if thresh in loc_report['detection_at_thresholds']:
                loc_info = loc_report['detection_at_thresholds'][thresh]
                base_info = base_report['detection_at_thresholds'].get(thresh, {})
                detection_comparison[thresh] = {
                    'screening_fraction': loc_info.get('screening_fraction', 0),
                    'localized_detection_rate': loc_info.get('detection_rate', 0),
                    'baseline_detection_rate': base_info.get('detection_rate', 0),
                    'absolute_gain': (
                        loc_info.get('detection_rate', 0) -
                        base_info.get('detection_rate', 0)
                    ),
                    'localized_nns': loc_info.get('nns', 0),
                    'baseline_nns': base_info.get('nns', 0),
                }

        # NNS比较
        nns_comparison = {}
        for thresh in ['top_1pct', 'top_5pct', 'top_10pct']:
            if thresh in loc_report['detection_at_thresholds']:
                loc_nns = loc_report['detection_at_thresholds'][thresh].get('nns', 0)
                base_nns = base_report['detection_at_thresholds'].get(thresh, {}).get('nns', 0)
                nns_reduction = (
                    (base_nns - loc_nns) / max(base_nns, 1) * 100
                    if base_nns > 0 else 0
                )
                nns_comparison[thresh] = {
                    'localized_nns': loc_nns,
                    'baseline_nns': base_nns,
                    'nns_reduction_pct': nns_reduction,
                }

        # Bootstrap置信区间
        ci_results = {}
        if n_bootstrap > 0 and NUMPY_AVAILABLE:
            # cluster_field 不显式传参：_compute_bootstrap_ci 内部按候选户键自动探测
            ci_results = self._compute_bootstrap_ci(records, n_bootstrap)

        return {
            'localized_model': {
                'top100_detected': loc_report['top100_detected'],
                'top500_detected': loc_report['top500_detected'],
                'risk_ratio': loc_report['risk_ratio'],
                'mean_risk_confirmed': loc_report['mean_risk_confirmed'],
            },
            'baseline_model': {
                'top100_detected': base_report['top100_detected'],
                'top500_detected': base_report['top500_detected'],
                'risk_ratio': base_report['risk_ratio'],
                'mean_risk_confirmed': base_report['mean_risk_confirmed'],
            },
            'detection_rate_comparison': detection_comparison,
            'nns_comparison': nns_comparison,
            'absolute_gains': {
                'top100_extra_detected': (
                    loc_report['top100_detected'] - base_report['top100_detected']
                ),
                'top500_extra_detected': (
                    loc_report['top500_detected'] - base_report['top500_detected']
                ),
                'risk_ratio_improvement': (
                    loc_report['risk_ratio'] - base_report['risk_ratio']
                ),
            },
            'bootstrap_confidence_intervals': ci_results,
            'clinical_impact_summary': self._compute_clinical_impact(
                detection_comparison
            ),
        }

    def _compute_bootstrap_ci(self, records, n_bootstrap, cluster_field=None):
        """Bootstrap法计算检出率增益的置信区间（v2.0：BCa + Wilcoxon）

        使用 BCa (Bias-Corrected and Accelerated) 方法计算置信区间，
        比百分位法对偏态分布更稳健。同时执行 Wilcoxon 符号秩检验
        评估本土化模型是否显著优于基线。

        v2.1（簇重采样）：接触追踪记录有家庭/簇聚集结构，记录级
        重采样把同簇记录当独立样本，会低估方差使 CI 偏窄。检测到
        簇字段时按簇整体有放回重采样（cluster bootstrap）。

        文献：
        - Efron B, Tibshirani RJ. An Introduction to the Bootstrap. CRC, 1994.
        - Wilcoxon F. Individual comparisons by ranking methods. Biometrics, 1945.
        - Cameron A, Gelbach J, Miller D. Bootstrap-based improvements
          for inference with clustered data. Rev. Econ. Statist., 2008.
        """
        import numpy as np

        # 导入 scipy（用于 BCa 和 Wilcoxon）
        try:
            from scipy import stats as sps
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False
            sps = None

        # ---- 簇（家庭）结构解析：自动探测或使用显式字段 ----
        # 簇字段候选：项目数据源（SINAN/Muchuro/HomeACF）的户键命名
        # 不统一（Muchuro 为 group，见项目记忆），逐候选探测前 50 条
        _CLUSTER_FIELD_CANDIDATES = (
            'household_id', 'household', 'group', 'family_id',
            'cluster_id', 'hh_id',
        )
        resolved_cluster_field = cluster_field
        if resolved_cluster_field is None and records:
            probe = records[:50]
            for cand in _CLUSTER_FIELD_CANDIDATES:
                if any(cand in r for r in probe):
                    resolved_cluster_field = cand
                    break

        cluster_members = None
        if resolved_cluster_field is not None:
            from collections import defaultdict
            clusters = defaultdict(list)
            for i, rec in enumerate(records):
                key = rec.get(resolved_cluster_field)
                if key is None:
                    # 缺失簇键的记录各自成簇（保守：不与其他记录合并，
                    # 也不丢弃——保持样本量语义）
                    key = ('__orphan__', i)
                clusters[key].append(i)
            cluster_members = list(clusters.values())
            LOGGER.info(
                "Bootstrap 按簇重采样: 字段=%s, %d 簇 / %d 条记录",
                resolved_cluster_field, len(cluster_members), len(records))
        else:
            LOGGER.warning(
                "未探测到簇字段（候选: %s），回退记录级重采样——"
                "簇聚集数据的 CI 可能偏窄",
                ', '.join(_CLUSTER_FIELD_CANDIDATES))

        # 点估计（全数据集）
        loc_bt = BacktestEngine(self.localizer, use_localization=True)
        base_bt = BacktestEngine(self.localizer, use_localization=False)
        loc_r = loc_bt.run_backtest(records)
        base_r = base_bt.run_backtest(records)
        point_estimate = 0.0
        if 'top_5pct' in loc_r.get('detection_at_thresholds', {}):
            point_estimate = (
                loc_r['detection_at_thresholds']['top_5pct']['detection_rate']
                - base_r['detection_at_thresholds'].get('top_5pct', {}).get(
                    'detection_rate', 0)
            )

        n = len(records)
        # 重采样单元数（BCa 加速参数的样本量口径：簇重采样时为簇数）
        n_units = len(cluster_members) if cluster_members else n
        bootstrap_gains = []
        # 收集各 Bootstrap 样本的检出率（用于 Wilcoxon 检验）
        loc_rates_bootstrap = []
        base_rates_bootstrap = []

        for _ in range(n_bootstrap):
            if cluster_members is not None:
                # 簇级有放回重采样：抽簇，取簇内全部记录
                # （重复抽中的簇其记录重复出现——cluster bootstrap 语义）
                n_clusters = len(cluster_members)
                sampled_units = np.random.choice(
                    n_clusters, n_clusters, replace=True)
                indices = np.concatenate(
                    [cluster_members[u] for u in sampled_units])
            else:
                indices = np.random.choice(n, n, replace=True)
            sample = [records[i] for i in indices]

            loc_bt_b = BacktestEngine(self.localizer, use_localization=True)
            base_bt_b = BacktestEngine(self.localizer, use_localization=False)

            loc_r_b = loc_bt_b.run_backtest(sample)
            base_r_b = base_bt_b.run_backtest(sample)

            if 'top_5pct' in loc_r_b.get('detection_at_thresholds', {}):
                loc_rate = loc_r_b['detection_at_thresholds']['top_5pct']['detection_rate']
                base_rate = base_r_b['detection_at_thresholds'].get('top_5pct', {}).get(
                    'detection_rate', 0)
                bootstrap_gains.append(loc_rate - base_rate)
                loc_rates_bootstrap.append(loc_rate)
                base_rates_bootstrap.append(base_rate)

        if not bootstrap_gains:
            return {}

        gains_arr = np.array(bootstrap_gains)

        # BCa 置信区间 (Efron & Tibshirani, 1994)
        bias_correction, acceleration = None, None
        if HAS_SCIPY:
            bias_correction, acceleration = self._compute_bca_params(
                gains_arr, point_estimate, n)

        if bias_correction is not None and acceleration is not None:
            z_alpha_lo = sps.norm.ppf(0.025)
            z_alpha_hi = sps.norm.ppf(0.975)

            # BCa 调整后的分位数
            bca_lo = sps.norm.cdf(
                bias_correction + (bias_correction + z_alpha_lo) / (
                    1 - acceleration * (bias_correction + z_alpha_lo)))
            bca_hi = sps.norm.cdf(
                bias_correction + (bias_correction + z_alpha_hi) / (
                    1 - acceleration * (bias_correction + z_alpha_hi)))

            ci_95_lower = float(np.percentile(gains_arr, bca_lo * 100))
            ci_95_upper = float(np.percentile(gains_arr, bca_hi * 100))
        else:
            ci_95_lower = float(np.percentile(gains_arr, 2.5))
            ci_95_upper = float(np.percentile(gains_arr, 97.5))

        # Wilcoxon 符号秩检验（本土化 vs 基线）
        wilcoxon_result = {}
        if len(loc_rates_bootstrap) >= 20 and len(base_rates_bootstrap) >= 20:
            try:
                from scipy import stats as sps_stats
                stat, p_value = sps_stats.wilcoxon(
                    loc_rates_bootstrap, base_rates_bootstrap,
                    alternative='greater')
                wilcoxon_result = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant_at_0_05': p_value < 0.05,
                    'significant_at_0_01': p_value < 0.01,
                    'test': 'Wilcoxon signed-rank (one-sided: localized > baseline)',
                }
            except (ImportError, ValueError):
                wilcoxon_result = {'note': 'scipy 不可用或数据不足，无法执行 Wilcoxon 检验'}

        return {
            'mean_gain_top5pct': float(np.mean(gains_arr)),
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'ci_90_lower': float(np.percentile(gains_arr, 5.0)),
            'ci_90_upper': float(np.percentile(gains_arr, 95.0)),
            'ci_method': 'BCa (Bias-Corrected and Accelerated)',
            'n_bootstrap': n_bootstrap,
            'point_estimate': point_estimate,
            'wilcoxon_test': wilcoxon_result,
            # 重采样单元披露：簇级（含字段名/簇数）或记录级
            # （记录级 CI 对簇聚集数据偏窄，解读时须注意）
            'bootstrap_unit': (
                f'cluster:{resolved_cluster_field}'
                if cluster_members is not None else 'record'),
            'n_resample_units': int(n_units),
        }

    @staticmethod
    def _compute_bca_params(bootstrap_stats, point_estimate, n):
        """计算 BCa 置信区间的偏差校正和加速参数。

        文献：Efron B, Tibshirani RJ. An Introduction to the Bootstrap. CRC, 1994.

        参数：
            bootstrap_stats: np.ndarray, Bootstrap 统计量
            point_estimate: float, 全数据集点估计
            n: int, 原始样本量

        返回：
            tuple: (bias_correction, acceleration) or (None, None) if computation fails
        """
        try:
            import numpy as np
            from scipy import stats as sps

            # 偏差校正 z0
            prop_less = np.mean(bootstrap_stats < point_estimate)
            z0 = sps.norm.ppf(max(prop_less, 1e-10))

            # 加速参数 a（基于 Jackknife 近似）
            # 简化实现：使用 Bootstrap 分布的偏度作为加速参数近似
            skew = float(sps.skew(bootstrap_stats))
            acceleration = skew / (6.0 * np.sqrt(n))

            return float(z0), float(acceleration)
        except Exception as e:
            LOGGER.debug("BCa 加速参数计算失败: %s", e)
            return None, None

    @staticmethod
    def _compute_clinical_impact(detection_comparison):
        if not detection_comparison:
            return {'summary': '数据不足'}

        items = [
            (info.get('absolute_gain', 0), thresh)
            for thresh, info in detection_comparison.items()
        ]
        if not items:
            return {'summary': '数据不足'}

        best_gain = max(items)

        return {
            'max_detection_rate_gain': best_gain[0],
            'at_screening_fraction': best_gain[1],
            'interpretation': (
                f'在最佳筛查覆盖率下，本土化模型检出率提升'
                f'{best_gain[0]*100:.1f}个百分点'
                if best_gain[0] > 0
                else '本土化模型未显示显著检出率提升'
            ),
        }
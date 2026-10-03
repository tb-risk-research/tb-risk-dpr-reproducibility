import copy
import logging

from ..scoring.engine import ScoringEngine

LOGGER = logging.getLogger("tb_risk.validation")


class AblationAnalyzer:
    """消融实验分析器

    逐一禁用各子模块，通过 monkey-patching 将各模块的核心
    参数临时替换为恒等值（乘数=1.0），在相同数据集上比较
    平均发病概率和风险区分度变化，量化每个维度的本土化增益。
    """

    MODULE_NAMES = [
        ('epi',       '一-流行病学校准'),
        ('oilfield',  '二-油田职业暴露'),
        ('idu',       '三-IDU人群风险'),
        ('climate',   '四-气候气溶胶交互'),
        ('altitude',  '五-海拔适应性'),
        ('sdoh',      '六-社会决定因素(SDOH)'),
        ('policy',    '七-地方政策引擎'),
    ]

    def __init__(self, localizer):
        self.localizer = localizer

    def run_ablation(self, records):
        results = {}

        full_engine = ScoringEngine(self.localizer, use_localization=True)
        full_report = self._evaluate_with_engine(full_engine, records)
        results['full_model'] = {'label': '完整本土化模型', 'report': full_report}

        base_engine = ScoringEngine(self.localizer, use_localization=False)
        base_report = self._evaluate_with_engine(base_engine, records)
        results['baseline'] = {'label': '无本土化基线', 'report': base_report}

        for module_key, module_label in self.MODULE_NAMES:
            disabled_report = self._evaluate_disabled(
                records, module_key
            )
            results[f'disable_{module_key}'] = {
                'label': f'禁用: {module_label}',
                'report': disabled_report,
            }

        ablation_gains = {}
        full_mean = full_report['mean_disease_prob']
        for module_key, module_label in self.MODULE_NAMES:
            key = f'disable_{module_key}'
            if key in results:
                dm = results[key]['report']['mean_disease_prob']
                delta = full_mean - dm
                pct = (delta / max(dm, 0.001)) * 100
                ablation_gains[module_key] = {
                    'label': module_label,
                    'full_mean_risk': full_mean,
                    'disabled_mean_risk': dm,
                    'absolute_delta': delta,
                    'relative_change_pct': pct,
                }

        results['ablation_gains'] = ablation_gains
        return results

    def _evaluate_disabled(self, records, module_key):
        """通过deepcopy配置禁用指定模块的贡献（避免污染单例）"""
        saved_config = copy.deepcopy(self.localizer._config._data)
        patches = self._get_patches(module_key)
        self._apply_patches(patches)
        try:
            engine = ScoringEngine(self.localizer, use_localization=True)
            result = self._evaluate_with_engine(engine, records)
        finally:
            self.localizer._config._data = saved_config
            self._revert_patches(patches)
        return result

    def _get_patches(self, module_key):
        """生成monkey-patch描述：通过ConfigProxy修改属性型参数，
        通过setattr替换方法型函数。"""
        patches = []
        lz = self.localizer

        if module_key == 'epi':
            orig = lz._config.get_dict('epidemiology')
            if 'base_incidence_per_100k' in orig:
                patches.append(('_config', 'epidemiology.base_incidence_per_100k',
                                orig.get('base_incidence_per_100k', 121.0)))
                lz._config._data['epidemiology']['base_incidence_per_100k'] = 0.001
        elif module_key == 'oilfield':
            orig = lz._config.get_dict('occupation')
            patches.append(('_config', 'occupation.oilfield_camp_factor',
                            orig.get('oilfield_camp_factor', 0.85)))
            lz._config._data['occupation']['oilfield_camp_factor'] = 1.0
        elif module_key == 'idu':
            # IDURiskModule 在 __init__ 中缓存了 _idu_params，仅修改 config 不会生效，
            # 必须同时 patch 模块内部的缓存字典。
            orig_data = copy.deepcopy(
                lz._config._data.get('population', {}).get('special_populations', {}).get('idu', {})
            )
            patches.append(('_config_idu', 'idu_immunosuppression_combined',
                            orig_data.get('immunosuppression_combined', 10.0)))
            if hasattr(lz.idu, '_idu_params'):
                patches.append(('_idu_params', 'immunosuppression_combined',
                                lz.idu._idu_params.get('immunosuppression_combined', 10.0)))
                lz.idu._idu_params['immunosuppression_combined'] = 1.0
            if 'population' in lz._config._data:
                pop = lz._config._data['population']
                if 'special_populations' not in pop:
                    pop['special_populations'] = {}
                if 'idu' not in pop['special_populations']:
                    pop['special_populations']['idu'] = {}
                pop['special_populations']['idu']['immunosuppression_combined'] = 1.0
        elif module_key == 'climate':
            orig_fn = lz.climate.calculate_climate_multiplier
            patches.append(('climate', 'calculate_climate_multiplier', orig_fn))
            lz.climate.calculate_climate_multiplier = lambda *a, **kw: 1.0
        elif module_key == 'altitude':
            orig_fn = lz.altitude.get_progression_adjustment
            patches.append(('altitude', 'get_progression_adjustment', orig_fn))
            lz.altitude.get_progression_adjustment = \
                lambda *a, **kw: {'applicable': False, 'factor': 1.0,
                                  'description': 'ablated'}
        elif module_key == 'sdoh':
            # 禁用社会决定因素（SDOH）模块：将 compute_sdoh_multiplier 替换为返回 1.0
            orig_fn = lz.sdoh.compute_sdoh_multiplier
            patches.append(('sdoh', 'compute_sdoh_multiplier', orig_fn))
            lz.sdoh.compute_sdoh_multiplier = lambda *a, **kw: {
                'multiplier': 1.0,
                'components': {},
                'description': 'SDOH消融: 所有社会决定因素设为1.0',
                'fairness_audit': {'method': 'ablated'},
            }
        elif module_key == 'policy':
            # 禁用地方政策干预：将政策效果归零
            # 这包括集中收治率、关爱行动、社区筛查等政策杠杆
            if hasattr(lz, 'policy') and lz.policy is not None:
                if hasattr(lz.policy, 'centralized_treatment_rate'):
                    orig_rate = lz.policy.centralized_treatment_rate
                    patches.append(('policy', 'centralized_treatment_rate', orig_rate))
                    lz.policy.centralized_treatment_rate = 0.0
                if hasattr(lz.policy, 'care_action_effect'):
                    orig_effect = lz.policy.care_action_effect
                    patches.append(('policy', 'care_action_effect', orig_effect))
                    lz.policy.care_action_effect = 0.0
                if hasattr(lz.policy, 'community_screening_coverage'):
                    orig_coverage = lz.policy.community_screening_coverage
                    patches.append(('policy', 'community_screening_coverage', orig_coverage))
                    lz.policy.community_screening_coverage = 0.0

        return patches

    def _apply_patches(self, patches):
        pass  # 修改已在_get_patches中完成

    def _revert_patches(self, patches):
        lz = self.localizer
        for patch in patches:
            obj_key, attr, orig_val = patch
            if obj_key == '_config':
                keys = attr.split('.')
                node = lz._config._data
                for k in keys[:-1]:
                    if k in node and isinstance(node[k], dict):
                        node = node[k]
                    else:
                        node = None
                        break
                if node is not None and isinstance(node, dict):
                    node[keys[-1]] = orig_val
            elif obj_key == '_config_idu':
                if 'population' in lz._config._data:
                    pop = lz._config._data['population']
                    if 'special_populations' in pop:
                        if 'idu' in pop['special_populations']:
                            pop['special_populations']['idu']['immunosuppression_combined'] = orig_val
            elif obj_key == '_idu_params':
                if hasattr(lz.idu, '_idu_params'):
                    lz.idu._idu_params[attr] = orig_val
            else:
                target_obj = getattr(lz, obj_key, None)
                if target_obj is not None:
                    setattr(target_obj, attr, orig_val)

    def _evaluate_with_engine(self, engine, records):
        scored = []
        for rec in records:
            result = engine.compute_risk_score(rec)
            scored.append({**rec, **result})

        confirmed = [r for r in scored if r.get('is_confirmed', 0)]
        non_confirmed = [r for r in scored if not r.get('is_confirmed', 0)]

        mean_conf = (
            sum(r['disease_probability'] for r in confirmed) / max(len(confirmed), 1)
        ) if confirmed else 0.0
        mean_non = (
            sum(r['disease_probability'] for r in non_confirmed) / max(len(non_confirmed), 1)
        ) if non_confirmed else 0.0

        mean_prog = (
            sum(r.get('progression_multiplier', 1.0) for r in scored) / max(len(scored), 1)
        )

        result = {
            'mean_disease_prob': (
                sum(r['disease_probability'] for r in scored) / max(len(scored), 1)
            ),
            'mean_confirmed_risk': mean_conf,
            'mean_non_confirmed_risk': mean_non,
            'risk_discrimination_ratio': mean_conf / max(mean_non, 0.001),
            'mean_progression_multiplier': mean_prog,
            'scored_records': scored,
        }

        # 计算 AUROC（如果数据足够）
        if len(scored) >= 2:
            y_true = [int(r.get('is_confirmed', 0)) for r in scored]
            y_prob = [r.get('disease_probability', 0.0) for r in scored]
            result['auroc'] = self._compute_auroc(y_true, y_prob)
        else:
            result['auroc'] = None

        return result

    @staticmethod
    def _compute_auroc(y_true, y_prob):
        """计算 AUROC（Area Under ROC Curve）。

        使用梯形法则进行数值积分，避免对 scipy 的硬依赖。
        """
        import numpy as np
        y_true = np.array(y_true, dtype=int)
        # 检查是否只有一类
        if len(np.unique(y_true)) < 2:
            return 0.5
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_prob))
        except (ImportError, ValueError):
            pass

        # 按预测概率降序排序
        order = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[order]

        # 累积 TP 和 FP
        n_pos = np.sum(y_true_sorted)
        n_neg = len(y_true_sorted) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float('nan')

        tpr = np.cumsum(y_true_sorted) / n_pos
        fpr = np.cumsum(1 - y_true_sorted) / n_neg

        return float(np.trapz(tpr, fpr))

    def run_ablation_with_metrics(self, records, n_bootstrap=500):
        """消融实验（含 AUROC + 统计检验）

        在标准消融实验基础上，增加：
        1. 各模块的 AUROC 对比
        2. Wilcoxon 符号秩检验（完整模型 vs 禁用模块）
        3. Bootstrap AUROC 置信区间

        参数：
            records (list[dict]): 筛查记录
            n_bootstrap (int): Bootstrap 重采样次数

        返回：
            dict: 包含 AUROC 和统计检验的扩展消融报告
        """
        base_result = self.run_ablation(records)

        # 计算各组 AUROC
        full_auroc = base_result['full_model']['report'].get('auroc')
        baseline_auroc = base_result['baseline']['report'].get('auroc')

        auroc_comparison = {
            'full_model': full_auroc,
            'baseline': baseline_auroc,
            'disabled_modules': {},
        }

        for module_key, _ in self.MODULE_NAMES:
            key = f'disable_{module_key}'
            if key in base_result:
                auroc = base_result[key]['report'].get('auroc')
                auroc_comparison['disabled_modules'][module_key] = auroc

        # Bootstrap AUROC 置信区间（完整模型）
        auroc_ci = None
        if full_auroc is not None and len(records) >= 10:
            auroc_ci = self._bootstrap_auroc_ci(
                records, n_bootstrap=n_bootstrap)

        # Wilcoxon 检验（完整模型 vs 基线）
        wilcoxon_result = None
        if len(records) >= 20:
            wilcoxon_result = self._wilcoxon_test(
                base_result['full_model']['report'].get('scored_records', []),
                base_result['baseline']['report'].get('scored_records', []))

        base_result['auroc_comparison'] = auroc_comparison
        base_result['auroc_bootstrap_ci'] = auroc_ci
        base_result['wilcoxon_test'] = wilcoxon_result

        # 边际 AUROC 贡献量：各模块 AUROC 与完整模型 AUROC 的差值
        # 正值表示该模块禁用后 AUROC 下降，即该模块有正向贡献
        marginal_auroc = {}
        if full_auroc is not None:
            for module_key, module_label in self.MODULE_NAMES:
                disabled_auroc = auroc_comparison['disabled_modules'].get(module_key)
                if disabled_auroc is not None:
                    delta = full_auroc - disabled_auroc
                    marginal_auroc[module_key] = {
                        'label': module_label,
                        'full_auroc': full_auroc,
                        'disabled_auroc': disabled_auroc,
                        'marginal_auroc_delta': delta,
                        'marginal_auroc_pct': (delta / max(abs(full_auroc), 0.001)) * 100,
                        'direction': 'positive' if delta > 0 else ('negative' if delta < 0 else 'neutral'),
                    }
        base_result['marginal_auroc_contribution'] = marginal_auroc

        # 方向级边际贡献汇总（用于集成权重调整）
        # 按消融维度分组到对应的评分方向：epi→SEIR, oilfield/idu/climate/altitude/sdoh/policy→ML
        # 这些边际贡献可以作为权重调整的数据依据
        direction_contributions = {
            'seir': {'marginal_auroc_sum': 0.0, 'significant_modules': []},
            'ml': {'marginal_auroc_sum': 0.0, 'significant_modules': []},
            'gnn': {'marginal_auroc_sum': 0.0, 'significant_modules': [], 'note': 'GNN方向无消融模块，保持默认权重'},
        }
        # 模块到方向的映射
        module_to_direction = {
            'epi': 'seir',
            'oilfield': 'ml',
            'idu': 'ml',
            'climate': 'seir',
            'altitude': 'seir',
            'sdoh': 'ml',
            'policy': 'ml',
        }
        for module_key, contrib in marginal_auroc.items():
            direction = module_to_direction.get(module_key)
            if direction and contrib['marginal_auroc_delta'] > 0.001:
                direction_contributions[direction]['marginal_auroc_sum'] += contrib['marginal_auroc_delta']
                direction_contributions[direction]['significant_modules'].append(
                    f"{contrib['label']}(ΔAUROC={contrib['marginal_auroc_delta']:.4f})"
                )
        base_result['direction_contributions'] = direction_contributions

        return base_result

    def _bootstrap_auroc_ci(self, records, n_bootstrap=500):
        """Bootstrap 计算 AUROC 的 95% 置信区间"""
        import numpy as np
        n = len(records)
        auroc_samples = []

        full_engine = ScoringEngine(self.localizer, use_localization=True)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            sample = [records[i] for i in indices]
            scored = []
            for rec in sample:
                result = full_engine.compute_risk_score(rec)
                scored.append({**rec, **result})

            y_true = [int(r.get('is_confirmed', 0)) for r in scored]
            y_prob = [r.get('disease_probability', 0.0) for r in scored]
            try:
                auroc = self._compute_auroc(y_true, y_prob)
                if not np.isnan(auroc):
                    auroc_samples.append(auroc)
            except (ValueError, TypeError) as e:
                LOGGER.warning("Bootstrap AUROC 计算失败: %s", e)

        if len(auroc_samples) < 10:
            return None

        auroc_arr = np.array(auroc_samples)
        return {
            'mean': float(np.mean(auroc_arr)),
            'ci_95_lower': float(np.percentile(auroc_arr, 2.5)),
            'ci_95_upper': float(np.percentile(auroc_arr, 97.5)),
            'n_bootstrap': n_bootstrap,
        }

    @staticmethod
    def _wilcoxon_test(full_scored, baseline_scored):
        """Wilcoxon 符号秩检验：完整模型 vs 基线"""
        try:
            from scipy import stats
            full_probs = [r.get('disease_probability', 0.0) for r in full_scored]
            base_probs = [r.get('disease_probability', 0.0) for r in baseline_scored]

            min_len = min(len(full_probs), len(base_probs))
            if min_len < 20:
                return {'note': '样本量不足 (n < 20)，无法执行 Wilcoxon 检验'}

            stat, p_value = stats.wilcoxon(
                full_probs[:min_len], base_probs[:min_len],
                alternative='two-sided')

            return {
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant_at_0_05': p_value < 0.05,
                'significant_at_0_01': p_value < 0.01,
                'test': 'Wilcoxon signed-rank (two-sided: full vs baseline)',
            }
        except (ImportError, ValueError) as e:
            return {'note': f'无法执行 Wilcoxon 检验: {e}'}
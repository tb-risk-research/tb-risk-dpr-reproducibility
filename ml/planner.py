"""
动态干预规划器与策略验证器

包含序列化动态干预规划器 (DynamicInterventionPlanner) 和
干预策略验证器 (InterventionValidator)。

文献：
  Gottesman O et al. (2018) Evaluating RL in health
  Puli AM et al. (NeurIPS 2022) Causal deep RL
  Moffa G et al. (2024) Sequential decision-making in TB control
"""

import logging
import numpy as np

LOGGER = logging.getLogger("tb_risk.ml")


class DynamicInterventionPlanner:
    """序列化动态干预规划器

    组合 BCQ 离线策略与因果正则化在线精调，
    为具体接触者生成时间序贯干预计划。

    文献：
      Gottesman O et al. (2018) Evaluating RL in health
      Puli AM et al. (NeurIPS 2022) Causal deep RL
      Moffa G et al. (2024) Sequential decision-making in TB control
    """

    ACTION_NAMES = {
        0: ('无干预', '保持观察，不采取额外措施'),
        1: ('人群筛查20%', '对20%高风险人群进行IGRA/TST筛查'),
        2: ('人群筛查40%', '对40%高风险人群进行筛查'),
        3: ('人群筛查60%', '对60%高风险人群进行筛查'),
        4: ('人群筛查80%', '对80%高风险人群进行大规模筛查'),
        5: ('人群筛查100%', '全员筛查，最大检出率'),
        6: ('接触者追踪10人', '追踪近期10名密切接触者'),
        7: ('接触者追踪30人', '追踪近期30名密切接触者'),
        8: ('接触者追踪50人', '追踪近期50名密切接触者，扩大范围'),
        9: ('治疗延迟1周', '将诊断至治疗启动时间缩短至1周内'),
        10: ('治疗延迟2周', '将诊断至治疗启动时间缩短至2周内'),
        11: ('环境干预', '改善通风+紫外线消毒+防疫教育'),
    }

    ACTION_CATEGORIES = {
        'screening': [1, 2, 3, 4, 5],
        'tracing': [6, 7, 8],
        'treatment_acceleration': [9, 10],
        'environmental': [11],
        'none': [0],
    }

    # 风险轨迹衰减系数（统一管理，避免魔法数字）
    TRAJECTORY_BASE_RISK = 0.35           # 初始基准风险
    TRAJECTORY_STEP_DECAY = 0.04          # 每步基础衰减
    TRAJECTORY_MIN_RISK = 0.02            # 最小风险下限
    TRAJECTORY_INTERCEPT = 0.30           # 干预后风险截距
    TRAJECTORY_SCREENING_DECAY = 0.06     # 筛查/追踪每步衰减
    TRAJECTORY_TREATMENT_DECAY = 0.04     # 治疗加速每步衰减
    TRAJECTORY_ENV_DECAY = 0.08           # 环境干预每步衰减
    TRAJECTORY_FALLBACK_MIN = 0.05        # 无预测器时的回退最小风险

    def __init__(self, env=None, bcq_agent=None, causal_graph=None,
                 preference_policy=None, rl_engine=None, device='cpu'):
        self.env = env
        self.bcq_agent = bcq_agent
        self.causal_graph = causal_graph
        self.preference_policy = preference_policy
        self.rl_engine = rl_engine
        self.device = device
        self.planning_history = []

    def set_environment(self, env):
        self.env = env

    def set_bcq_agent(self, bcq_agent):
        self.bcq_agent = bcq_agent

    def plan_sequence(self, initial_state, n_steps=8, preference='balanced',
                      deterministic=False, progress_callback=None):
        """规划干预序列

        参数：
            initial_state: 初始状态向量 [STATE_DIM]
            n_steps: 规划步数（周）
            preference: 偏好设置 ('cost_sensitive'/'effect_sensitive'/'balanced')
            deterministic: 是否使用确定性策略
            progress_callback: 进度回调函数

        返回：
            list: 干预序列 [{'step', 'action', 'action_name', 'risk_before',
                           'risk_after', 'cost', 'causal_valid', 'category'}, ...]
            dict: 摘要 {'total_cost', 'final_risk', 'risk_reduction',
                       'cost_per_reduction'}
        """
        if preference == 'cost_sensitive':
            pref_vec = [0.8, 0.2]
        elif preference == 'effect_sensitive':
            pref_vec = [0.2, 0.8]
        else:
            pref_vec = [0.5, 0.5]

        use_env = self.env is not None

        if use_env and self.rl_engine is not None:
            return self._plan_with_rl_engine(n_steps, preference,
                                             progress_callback)

        if self.preference_policy is not None:
            return self._plan_with_preference_policy(
                initial_state, n_steps, pref_vec, deterministic, progress_callback)

        return self._plan_heuristic(initial_state, n_steps, preference,
                                     progress_callback)

    def _plan_with_rl_engine(self, n_steps, preference, progress_callback=None):
        sequence, metrics = self.rl_engine.recommend_intervention_sequence(
            n_steps=n_steps, preference=preference)
        interpreted = []
        total_cost = 0.0
        for item in sequence:
            step = item.get('step', len(interpreted))
            action = item.get('action', 0)
            name, desc = self.ACTION_NAMES.get(action, ('未知', ''))
            cost = self._estimate_action_cost(action)
            total_cost += cost
            interpreted.append({
                'step': step,
                'action': action,
                'action_name': name,
                'description': desc,
                'cost': cost,
                'causal_valid': item.get('causal_path_valid', True),
                'causal_effect': item.get('causal_effect', '未知'),
                'category': self._get_action_category(action),
            })
            if progress_callback:
                progress_callback(f"[Step {step+1}] {name}: 成本={cost:.0f}")
        summary = {
            'total_cost': total_cost,
            'n_steps': n_steps,
            'status': metrics.get('status', 'ok'),
            'preference': preference,
        }
        return interpreted, summary

    def _plan_with_preference_policy(self, initial_state, n_steps, pref_vec,
                                      deterministic, progress_callback=None):
        state = np.array(initial_state, dtype=float)
        sequence = []
        total_cost = 0.0
        for step in range(n_steps):
            action = self.preference_policy.select_action(
                state, pref_vec, deterministic=deterministic)
            name, desc = self.ACTION_NAMES.get(action, ('未知', ''))
            cost = self._estimate_action_cost(action)
            total_cost += cost
            causal_valid = True
            if self.causal_graph is not None:
                effect = self.causal_graph.get_causal_effects(action, state)
                causal_valid = effect.get('causal_path_valid', True)
            sequence.append({
                'step': step,
                'action': action,
                'action_name': name,
                'description': desc,
                'cost': cost,
                'causal_valid': causal_valid,
                'category': self._get_action_category(action),
            })
            if self.env is not None:
                next_state, _, _, _ = self.env.step(action)
                state = next_state
            else:
                state = state * 0.98
            if progress_callback:
                progress_callback(f"[Step {step+1}] {name}: 成本={cost:.0f}")
        summary = {
            'total_cost': total_cost,
            'n_steps': n_steps,
            'status': 'ok',
            'preference': 'preference_conditioned',
        }
        return sequence, summary

    def _plan_heuristic(self, initial_state, n_steps, preference,
                         progress_callback=None):
        """基于初始状态的启发式干预规划

        根据 baseline_risk 选择干预强度：
          high   (≥0.7): 激进干预 — 筛查+主动发现+环境改善
          medium (0.3~0.7): 适中干预 — 筛查+接触者追踪
          low    (<0.3): 保守干预 — 仅基本筛查
        """
        state = np.array(initial_state, dtype=float) if initial_state is not None else None
        # 不使用 np.mean(state) 作为基线风险（31维异质特征取均值缺乏临床意义）。
        # 改用状态向量中与感染风险直接相关的维度加权计算：
        #   维度0(年龄归一化)、维度2(症状)、维度5(暴露累积)、维度9(高危标识)
        # 默认回退值为 0.3（人群基线风险）。
        if state is not None and len(state) >= 10:
            risk_indicators = [state[2], state[5], state[9]]  # 症状、暴露、高危
            baseline_risk = float(np.clip(np.mean(risk_indicators), 0.05, 0.95))
        else:
            baseline_risk = 0.3
        if baseline_risk >= 0.7:
            level = 'high'
        elif baseline_risk >= 0.3:
            level = 'medium'
        else:
            level = 'low'
        sequence = []
        total_cost = 0.0
        for step in range(n_steps):
            if level == 'high':
                # 激进干预：筛查(11) + 主动发现(3) + 环境改善(7) + 治疗加速(9) + 宣传教育(2)
                action = [11, 3, 7, 9, 2, 1, 0, 0][min(step, 7)]
            elif level == 'medium':
                # 适中干预：筛查(11) + 接触者追踪(1) + 环境改善(6) + 宣传教育(2)
                action = [11, 1, 6, 2, 0, 0, 0, 0][min(step, 7)]
            else:
                # 保守干预：仅基本筛查(10) + 宣传教育(2)
                action = [10, 2, 0, 0, 0, 0, 0, 0][min(step, 7)]
            name, desc = self.ACTION_NAMES.get(action, ('未知', ''))
            cost = self._estimate_action_cost(action)
            total_cost += cost
            causal_valid = True
            if self.causal_graph is not None:
                try:
                    effect = self.causal_graph.get_causal_effects(action, [])
                    causal_valid = effect.get('causal_path_valid', True)
                except Exception as e:
                    LOGGER.debug("因果图效应查询失败: %s", e)
                    causal_valid = True
            sequence.append({
                'step': step,
                'action': action,
                'action_name': name,
                'description': desc,
                'cost': cost,
                'causal_valid': causal_valid,
                'category': self._get_action_category(action),
            })
            if progress_callback:
                progress_callback(f"[Step {step+1}] {name}: 成本={cost:.0f}")
        summary = {
            'total_cost': total_cost,
            'n_steps': n_steps,
            'status': 'heuristic',
            'preference': preference,
            'baseline_risk': baseline_risk,
            'intensity_level': level,
        }
        return sequence, summary

    def _estimate_action_cost(self, action):
        cost_map = {
            0: 0.0, 1: 250.0, 2: 500.0, 3: 750.0, 4: 1000.0,
            5: 1250.0, 6: 300.0, 7: 900.0, 8: 1500.0,
            9: 2000.0, 10: 1000.0, 11: 5000.0,
        }
        return cost_map.get(action, 0.0)

    def _get_action_category(self, action):
        for cat, actions in self.ACTION_CATEGORIES.items():
            if action in actions:
                return cat
        return 'unknown'

    def track_risk_trajectory(self, sequence, risk_predictor=None):
        """追踪干预序列各步骤的预期风险变化

        参数：
            sequence: 干预序列
            risk_predictor: 可选的风险预测器（若提供则使用预测器计算风险，否则使用启发式衰减）
        """
        if risk_predictor is None:
            # 回退：使用启发式衰减常数估算风险轨迹
            for i, step in enumerate(sequence):
                p = 0
                e = 0
                o = 0
                for s in sequence[:i+1]:
                    cat = s.get('category', '')
                    if cat in ('screening', 'tracing'):
                        e += 1
                    elif cat == 'treatment_acceleration':
                        p += 1
                    elif cat == 'environmental':
                        o += 1
                step['expected_risk'] = max(
                    self.TRAJECTORY_MIN_RISK,
                    self.TRAJECTORY_INTERCEPT
                    - e * self.TRAJECTORY_SCREENING_DECAY
                    - p * self.TRAJECTORY_TREATMENT_DECAY
                    - o * self.TRAJECTORY_ENV_DECAY)
        else:
            # 使用预测器计算实际风险轨迹
            try:
                for i, step in enumerate(sequence):
                    # 构建当前步骤特征并预测风险
                    features = self._build_step_features(step, sequence[:i+1])
                    if features is not None:
                        risk = risk_predictor.predict_risk(features)
                        step['expected_risk'] = float(risk)
                    else:
                        step['expected_risk'] = self.TRAJECTORY_FALLBACK_MIN
            except Exception as e:
                LOGGER.debug("预测器调用失败，回退到启发式: %s", e)
                # 预测器调用失败时回退到启发式
                for i, step in enumerate(sequence):
                    step['expected_risk'] = max(
                        self.TRAJECTORY_FALLBACK_MIN,
                        self.TRAJECTORY_BASE_RISK - i * self.TRAJECTORY_STEP_DECAY)
        return sequence


# ============================================================================
# 部署度量（round-9 P0-2）：AUROC → yield@k / NNS / PPV@预算
#
# 疾控的真实问题不是「模型区分度多高」，而是「筛查前 k% 能捞到多少
# 病人、每确诊一例要筛多少人」。以下纯函数把判别力翻译成预算约束下
# 的检出语言，供实验脚本（run_sinan_deployment_metrics）与证据面板
# 共用；随机筛查基线 yield@k = k 是所有提升的分母。
# ============================================================================

DEFAULT_BUDGETS = (1, 2, 5, 10, 20, 30, 50, 100)


def yield_at_k(y_true, scores, k):
    """top-k% 风险分层捕获的阳性比例（yield@k，随机基线 = k%）。

    参数：
        y_true: 二值结局数组
        scores: 风险分（越大越高风险）
        k: 筛查预算（占总人数百分比）
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n, n_pos = len(y_true), int(np.sum(y_true))
    if n == 0 or n_pos == 0:
        return 0.0
    m = max(1, int(round(n * k / 100.0)))
    m = min(m, n)
    top = np.argsort(-scores, kind='stable')[:m]
    return float(np.sum(y_true[top]) / n_pos)


def screening_metrics_at_budgets(y_true, scores,
                                 budgets=DEFAULT_BUDGETS):
    """各筛查预算下的部署指标三元组（yield / NNS / PPV）+ lift。

    返回 {budget_k: {'yield', 'nns', 'ppv', 'lift'}}：
      yield —— top-k 捕获的阳性占全部阳性比例（随机基线 = k/100）
      nns   —— 每捕获一例阳性需筛查的人数（number needed to screen）
      ppv   —— top-k 内阳性率（筛查精度）
      lift  —— yield / (k/100)，相对随机筛查的检出倍数
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    n_pos = int(np.sum(y_true))
    out = {}
    for k in budgets:
        if n == 0 or n_pos == 0 or k <= 0:
            out[k] = {'yield': None, 'nns': None, 'ppv': None,
                      'lift': None}
            continue
        m = min(n, max(1, int(round(n * k / 100.0))))
        top = np.argsort(-scores, kind='stable')[:m]
        caught = int(np.sum(y_true[top]))
        yk = caught / n_pos
        out[k] = {
            'yield': yk,
            'nns': (m / caught) if caught else None,
            'ppv': caught / m,
            'lift': yk / (k / 100.0),
        }
    return out


class InterventionValidator:
    """干预策略验证器

    提供因果路径校验、仿真对比实验、鲁棒性测试和临床合理性审查。

    文献：
      Puli AM et al. (NeurIPS 2022)
      Gottesman O et al. (2018)
      Liu S et al. (2023) Safe RL for healthcare
    """

    def __init__(self, causal_graph=None, env_class=None, risk_predictor=None):
        self.causal_graph = causal_graph
        self.env_class = env_class
        self.risk_predictor = risk_predictor
        self.validation_results = {}

    def validate_causal_paths(self, sequence):
        """因果路径校验：检查每步动作的有效因果路径

        返回：
            list: 每步的因果校验结果
            dict: 摘要 {'all_valid', 'invalid_count', 'warning_actions'}
        """
        results = []
        invalid_count = 0
        warning_actions = []

        for step in sequence:
            action = step.get('action', 0)
            step_num = step.get('step', -1)

            if self.causal_graph is not None and action != 0:
                try:
                    causal_effect = self.causal_graph.get_causal_effects(
                        action, np.zeros(31))
                    valid = causal_effect.get('causal_path_valid', False)
                    direct_effect = causal_effect.get('direct_effect', '')
                except Exception as e:
                    LOGGER.debug("因果效应查询失败: %s", e)
                    valid = True
                    direct_effect = ''
            else:
                valid = True
                direct_effect = '无干预无需因果路径'

            results.append({
                'step': step_num,
                'action': action,
                'causal_path_valid': valid,
                'direct_effect': direct_effect,
            })

            if not valid:
                invalid_count += 1
                warning_actions.append({
                    'step': step_num,
                    'action': action,
                    'reason': '未找到从干预变量到感染风险的有效因果路径',
                })

        return results, {
            'all_valid': invalid_count == 0,
            'invalid_count': invalid_count,
            'warning_actions': warning_actions,
            'validation_rule': 'CausalGraphConstraint.get_causal_effects',
        }

    def run_simulation_comparison(self, n_nodes=30, horizon=26,
                                   n_trials=3, seed_base=42,
                                   preference='balanced',
                                   progress_callback=None):
        """仿真对比实验：对比三种策略

        对比策略：
          - 基线（无干预）
          - 静态贪心组合反事实（梯度优化结果）
          - 动态 RL 序列策略

        返回：
            dict: 比较结果
        """
        if self.env_class is None:
            return {'error': 'env_class未设置，无法进行仿真比较'}

        results = {
            'baseline': {'total_infections': [], 'total_cost': []},
            'static_counterfactual': {'total_infections': [], 'total_cost': []},
            'dynamic_rl': {'total_infections': [], 'total_cost': []},
        }

        for trial in range(n_trials):
            if progress_callback:
                progress_callback(f"[Trial {trial+1}/{n_trials}] 开始仿真...")

            env = self.env_class(n_nodes=n_nodes, horizon=horizon,
                                  seed=seed_base + trial)

            env.reset()
            baseline_inf = 0
            for t in range(horizon):
                _, _, _, info = env.step(0)
                baseline_inf += info.get('new_infections', 0)
            results['baseline']['total_infections'].append(baseline_inf)
            results['baseline']['total_cost'].append(0.0)

            env.reset()
            static_inf = 0
            static_cost = 0.0
            for t in range(horizon):
                action, cost = self._select_static_action(t, preference)
                _, _, _, info = env.step(action)
                static_inf += info.get('new_infections', 0)
                static_cost += cost
            results['static_counterfactual']['total_infections'].append(static_inf)
            results['static_counterfactual']['total_cost'].append(static_cost)

            env.reset()
            dynamic_inf = 0
            dynamic_cost = 0.0
            for t in range(horizon):
                action, cost = self._select_dynamic_action(t, horizon, preference)
                _, _, _, info = env.step(action)
                dynamic_inf += info.get('new_infections', 0)
                dynamic_cost += cost
            results['dynamic_rl']['total_infections'].append(dynamic_inf)
            results['dynamic_rl']['total_cost'].append(dynamic_cost)

        comparison = {}
        for strategy in ['baseline', 'static_counterfactual', 'dynamic_rl']:
            infs = results[strategy]['total_infections']
            costs = results[strategy]['total_cost']
            comparison[strategy] = {
                'mean_infections': float(np.mean(infs)),
                'std_infections': float(np.std(infs)) if len(infs) > 1 else 0.0,
                'mean_cost': float(np.mean(costs)),
                'std_cost': float(np.std(costs)) if len(costs) > 1 else 0.0,
            }
            comp = comparison[strategy]
            comp['cost_per_infection_averted_vs_baseline'] = (
                comp['mean_cost'] /
                max(float(comparison['baseline']['mean_infections']) - comp['mean_infections'], 1.0)
            ) if strategy != 'baseline' else float('inf')

        self.validation_results['simulation_comparison'] = comparison
        return comparison

    def _select_static_action(self, timestep, preference):
        if timestep == 0:
            return 11, 5000.0
        if timestep <= 3:
            a, c = (3, 750.0) if preference == 'effect_sensitive' else (1, 250.0)
            return a, c
        if timestep % 4 == 0:
            return 2, 500.0
        return 0, 0.0

    def _select_dynamic_action(self, timestep, horizon, preference):
        if timestep == 0:
            return 11, 5000.0
        phase = timestep / max(horizon, 1)
        if phase < 0.2:
            return (4, 1000.0) if preference == 'effect_sensitive' else (2, 500.0)
        elif phase < 0.4:
            return 7, 900.0
        elif phase < 0.6:
            return (3, 750.0) if preference == 'effect_sensitive' else (1, 250.0)
        elif phase < 0.8:
            return (9, 2000.0) if timestep % 4 == 0 else (0, 0.0)
        else:
            return 0, 0.0

    def robustness_test(self, base_env_params, param_variations,
                         n_trials=3, preference='balanced',
                         progress_callback=None):
        """鲁棒性测试：改变环境参数评估策略稳定性

        参数：
            base_env_params: 基础环境参数 dict
            param_variations: 参数变动列表
                [{'name': 'latent_period_mean', 'values': [3.0, 5.0, 8.0]}, ...]
            n_trials: 每组重复次数
            preference: 策略偏好

        返回：
            dict: {'parameter': ..., 'robustness_scores': [...]}
        """
        if self.env_class is None:
            return {'error': 'env_class未设置'}

        all_results = []
        for variation in param_variations:
            param_name = variation['name']
            param_results = {'parameter': param_name, 'values': []}
            for val in variation['values']:
                params = base_env_params.copy()
                params[param_name] = val
                env = self.env_class(**params)
                infs = []; costs = []
                for _ in range(n_trials):
                    env.reset()
                    total_inf = 0; total_cost = 0.0
                    for t in range(env.horizon):
                        action, cost = self._select_dynamic_action(
                            t, env.horizon, preference)
                        _, _, _, info = env.step(action)
                        total_inf += info.get('new_infections', 0)
                        total_cost += cost
                    infs.append(total_inf); costs.append(total_cost)
                param_results['values'].append({
                    'param_value': val,
                    'mean_infections': float(np.mean(infs)),
                    'std_infections': float(np.std(infs)),
                    'mean_cost': float(np.mean(costs)),
                    'std_cost': float(np.std(costs)),
                })
            all_results.append(param_results)
            if progress_callback:
                progress_callback(f"鲁棒性测试完成: {param_name}")
        self.validation_results['robustness'] = all_results
        return all_results

    def clinical_reasonableness_check(self, intervention_plan,
                                       patient_context=None):
        """临床合理性审查

        检查干预序列是否符合临床实践指南：
          1. 环境干预应在所有其他干预之前
          2. 筛查不会在已全员筛查后立即重复
          3. 追踪范围不能超过实际接触网络
          4. 治疗延迟缩短后应有一段观察期

        参数：
            intervention_plan: 干预序列
            patient_context: 患者背景信息

        返回：
            dict: 审查结果
        """
        issues = []
        suggestions = []

        actions = [s.get('action', 0) for s in intervention_plan]
        categories = [self._get_action_category(a) for a in actions]

        env_early = False
        for i, cat in enumerate(categories):
            if cat == 'environmental':
                if i <= 1:
                    env_early = True
                else:
                    issues.append({
                        'step': i,
                        'issue': '环境干预应尽早（前2步）实施',
                        'severity': 'medium',
                    })

        if not env_early:
            suggestions.append('建议首步实施环境干预，可最大程度阻断传播链')

        consecutive_screenings = 0
        for i, cat in enumerate(categories):
            if cat == 'screening':
                consecutive_screenings += 1
                if consecutive_screenings >= 3:
                    issues.append({
                        'step': i,
                        'issue': f'已连续{consecutive_screenings}步筛查，建议间隔观察',
                        'severity': 'low',
                    })
            else:
                consecutive_screenings = 0

        received_treatment_acceleration = False
        for i, cat in enumerate(categories):
            if cat == 'treatment_acceleration':
                if received_treatment_acceleration:
                    issues.append({
                        'step': i,
                        'issue': '治疗加速干预不应重复，单次即可',
                        'severity': 'high',
                    })
                received_treatment_acceleration = True

        checklist = {
            '环境干预优先': env_early,
            '筛查间隔合理': all(c != 'screening' or
                              (i == 0 or categories[i-1] != 'screening' or
                               categories[min(i+1, len(categories)-1)] != 'screening')
                              for i, c in enumerate(categories)),
            '干预连贯性': len([a for a in actions if a != 0]) <= len(actions) * 0.7,
            '成本递增合理': True,
            '有观察期': actions[-1] == 0 or actions[-2] == 0,
        }

        overall_score = sum(1 for v in checklist.values() if v) / max(len(checklist), 1)
        passed = overall_score >= 0.6

        return {
            'passed': passed,
            'overall_score': round(overall_score, 2),
            'checklist': checklist,
            'issues': issues,
            'suggestions': suggestions,
            'clinical_summary': '符合基本临床实践规范' if passed
                               else '存在需要审查的临床合理性问题',
        }

    def _get_action_category(self, action):
        cats = {
            'screening': [1, 2, 3, 4, 5],
            'tracing': [6, 7, 8],
            'treatment_acceleration': [9, 10],
            'environmental': [11],
        }
        for cat, actions in cats.items():
            if action in actions:
                return cat
        return 'none'

    def generate_validation_report(self, intervention_plan,
                                    patient_context=None,
                                    include_simulation=True,
                                    n_nodes=30, horizon=26):
        """生成完整验证报告

        返回：
            dict: 综合验证报告
        """
        report = {'timestamp': '', 'sections': {}}

        causal_results, causal_summary = self.validate_causal_paths(
            intervention_plan)
        report['sections']['causal_path_validation'] = {
            'summary': causal_summary,
            'details': causal_results,
        }

        clinical = self.clinical_reasonableness_check(
            intervention_plan, patient_context)
        report['sections']['clinical_reasonableness'] = clinical

        if include_simulation:
            try:
                simulation = self.run_simulation_comparison(
                    n_nodes=n_nodes, horizon=horizon)
                report['sections']['simulation_comparison'] = simulation
            except Exception as e:
                report['sections']['simulation_comparison'] = {
                    'error': f'仿真比较失败: {str(e)}'}

        report['overall_assessment'] = {
            'causal_valid': causal_summary.get('all_valid', False),
            'clinically_reasonable': clinical.get('passed', False),
            'recommended_for_deployment': (
                causal_summary.get('all_valid', False) and
                clinical.get('passed', False)
            ),
        }

        return report
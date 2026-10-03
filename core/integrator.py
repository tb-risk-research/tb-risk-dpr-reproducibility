#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三方向有机联动集成器（已从顶层 integrator.py 下沉至 core/ 子包）

顶层 integrator.py 保留为 re-export shim 以兼容既有导入。
由 RiskAssessmentService 持有实例，GUI 通过 service.get_integrator() 获取只读引用，
确保 GUI 与 CLI 入口的 ML 集成行为一致、结果结构统一。
"""

# 条件导入
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    from ..ml.framework import HeterogeneousTBNetwork
except ImportError:
    HeterogeneousTBNetwork = None

try:
    from ..ml.policy import RLInterventionEngine
    CAUSAL_RL_AVAILABLE = True
except ImportError:
    RLInterventionEngine = None
    CAUSAL_RL_AVAILABLE = False

# 问题六-步2：schema 契约用于 _build_seir_lookup 字段访问
from ..schemas import ContactRiskResult
from ..constants import DEFAULT_ENSEMBLE_WEIGHTS as _DEFAULT_ENSEMBLE_WEIGHTS

# 三层递进架构（个体基础层 → 网络增强层 → 社区干预层）：
# 替代"概率层面线性平均"的堆叠+门控整合入口。
try:
    from ..scoring.architecture import (
        ThreeLayerArchitecture,
        compute_individual_base,
        compute_network_increment,
        simulate_community_intervention,
        gated_integration,
        select_high_risk_set,
    )
    THREE_LAYER_AVAILABLE = True
except ImportError:
    ThreeLayerArchitecture = None
    compute_individual_base = None
    compute_network_increment = None
    simulate_community_intervention = None
    gated_integration = None
    select_high_risk_set = None
    THREE_LAYER_AVAILABLE = False


class ThreeDirectionIntegrator:
    """
    三个方向有机联动集成器（文献支撑：Breiman, 1996; Wolpert, 1992）

    方向一：异质性SEIR传播动力学
    方向二：GNN增强的社会接触网络分析
    方向三：SHAP 特征重要性分析（可解释性，非因果推断）

    联动机制：
    SHAP 特征重要性（方向三）→ SEIR参数约束（方向一）→ GNN归纳偏置（方向二）

    集成权重配置：
        权重存储在 self.ensemble_weights（dict）中，可通过以下方式配置：
        1. 构造时传入 custom_weights
        2. 构造后调用 set_ensemble_weights()
        3. 调用 optimize_weights_from_validation() 基于验证集 AUROC 性能加权
           （softmax 归一化的 log-AUROC，文献：Perrone & Cooper 1993; Jacobs et al. 1991）
        未配置时使用 DEFAULT_ENSEMBLE_WEIGHTS 中的经验值。
    """

    # 经验默认权重（单一真值源：constants.DEFAULT_ENSEMBLE_WEIGHTS）
    # 文献：Breiman 1996, Wolpert 1992, Perrone & Cooper 1993
    # 数据依据与警示：见 constants.py DEFAULT_ENSEMBLE_WEIGHTS 注释
    # （2026-08-22 双基准多 seed 评估；凸组合融合未显著超过最优单方向，
    #  部署须配合 set_fallback_policy() 护栏）
    DEFAULT_ENSEMBLE_WEIGHTS = _DEFAULT_ENSEMBLE_WEIGHTS

    # 向后兼容：保留原类常量别名（指向默认值，运行时实际使用 self.ensemble_weights）
    ENSEMBLE_WEIGHT_ML = _DEFAULT_ENSEMBLE_WEIGHTS['three']['ml']
    ENSEMBLE_WEIGHT_SEIR = _DEFAULT_ENSEMBLE_WEIGHTS['three']['seir']
    ENSEMBLE_WEIGHT_GNN = _DEFAULT_ENSEMBLE_WEIGHTS['three']['gnn']
    ENSEMBLE_WEIGHT_ML_TWO = _DEFAULT_ENSEMBLE_WEIGHTS['two_ml_gnn']['ml']
    ENSEMBLE_WEIGHT_OTHER_TWO = _DEFAULT_ENSEMBLE_WEIGHTS['two_ml_gnn']['gnn']

    def __init__(self, custom_weights=None, ml_clin_predictor=None):
        self.gnn_network_builder = HeterogeneousTBNetwork() if (PYTORCH_AVAILABLE and PYG_AVAILABLE and HeterogeneousTBNetwork is not None) else None
        self.is_gnn_trained = False

        # 临床标签 ML 成员（可选）：症状富集门控启用时替代 mech 模型打分。
        # None 或未训练时行为与旧版完全一致（单 ML 成员）。
        # 依据：跨基准汇总（run 20260823_085303）——学权重融合跨域均值
        # 0.797 超过任何单一模型（最高 0.773）；部署近似为"症状富集接触者
        # 用临床标签模型、其余用机制标签模型"的手工门控。
        self.ml_clin_predictor = ml_clin_predictor

        # 因果强化学习干预引擎
        self.rl_engine = None
        self._rl_ready = False
        if CAUSAL_RL_AVAILABLE and RLInterventionEngine is not None:
            try:
                self.rl_engine = RLInterventionEngine(
                    n_nodes=50, horizon=52, pareto_weights=5)
                self._rl_ready = True
            except (ImportError, RuntimeError, OSError):
                self._rl_ready = False

        # 可配置集成权重（深拷贝默认值，避免共享引用）
        import copy
        self.ensemble_weights = copy.deepcopy(self.DEFAULT_ENSEMBLE_WEIGHTS)
        self._weights_source = 'default'  # 'default' | 'custom' | 'validation'
        if custom_weights:
            self.set_ensemble_weights(custom_weights)

        # 部署护栏（集成自动回退）：验证集判定集成弱于最优单方向时，
        # integrate_predictions 的 ensemble 输出降级为该单方向
        self._fallback_to = None
        self._fallback_metrics = None

        # 问题六：SEIR 查找缓存（消除 _get_seir_results 的 O(n²) name 匹配）。
        # 以 id(tb_assessment.results) 为 cache key：结果对象重新赋值时 id 改变，
        # 缓存自动失效。同一轮评估内多次调用 integrate_predictions 时复用查找表。
        # 问题六-附带：schema 落地后 record_id 为强制字段，移除 name 回退，
        # by_name 不再构建（重名错配风险消除）。
        self._seir_lookup = {
            'by_record_id': {}, 'by_id': {},
            'base_infection_prob': 30,
        }
        self._seir_lookup_cache_key = None

    # ==================== 公开只读属性 ====================
    # 暴露内部状态的只读视图，避免外部调用者直接访问 _ 前缀私有成员。

    @property
    def weights_source(self) -> str:
        """集成权重来源标识：'default' | 'custom' | 'validation'"""
        return self._weights_source

    @property
    def rl_ready(self) -> bool:
        """因果强化学习引擎是否就绪"""
        return self._rl_ready

    @property
    def fallback_to(self):
        """部署护栏当前回退方向（None = 集成激活）。

        由 set_fallback_policy() 依据验证集指标设定：集成 AUROC 低于
        最优单方向时自动降级为该单方向，使"集成优于单方向"成为
        被持续监控的运行时属性而非一次性声明。
        """
        return self._fallback_to

    def set_fallback_policy(self, validation_metrics):
        """部署护栏：按验证集指标设定集成回退策略。

        参数：
            validation_metrics: dict，'ml'/'seir'/'gnn' 键为各方向验证集
                AUROC，'ensemble' 为集成 AUROC。方向缺失视为不可用。

        返回：
            str | None: 回退方向名（None = 集成保持激活）。

        示例：
            integrator.set_fallback_policy({
                'ml': 0.84, 'seir': 0.66, 'gnn': 0.76, 'ensemble': 0.75})
            # -> 'ml'（集成 0.75 < ML 0.84，降级为 ML 单方向）
        """
        singles = {k: float(v) for k, v in (validation_metrics or {}).items()
                   if k in ('ml', 'seir', 'gnn') and v is not None}
        ens = (validation_metrics or {}).get('ensemble')
        if not singles:
            self._fallback_to = None
            self._fallback_metrics = None
            return None
        best = max(singles, key=singles.get)
        if ens is not None and float(ens) < singles[best]:
            self._fallback_to = best
        else:
            self._fallback_to = None
        self._fallback_metrics = {'singles': singles,
                                  'ensemble': ens,
                                  'fallback_to': self._fallback_to}
        return self._fallback_to

    def _apply_fallback(self, results, ensemble_result):
        """执行 set_fallback_policy 设定的回退（方向不可用时保持集成）。"""
        if not self._fallback_to:
            return ensemble_result
        prob_map = {
            'ml': lambda: (results.get('ml') or {}).get('risk_probability'),
            'seir': lambda: (results.get('seir') or {}).get(
                'contact_infection_prob'),
            'gnn': lambda: (results.get('gnn') or {}).get('risk_probability'),
        }
        getter = prob_map.get(self._fallback_to)
        if getter is None:
            return ensemble_result
        prob = getter()
        if prob is None:
            return ensemble_result
        out = dict(ensemble_result)
        out['risk_probability'] = float(prob)
        out['risk_class'] = 1 if float(prob) > 50 else 0
        out['integration_note'] = (
            f'部署护栏回退：验证集集成 AUROC 低于最优单方向 '
            f'{self._fallback_to}，已降级为该单方向输出')
        out['fallback_to'] = self._fallback_to
        return out

    def set_ensemble_weights(self, weights):
        """显式配置集成权重。

        参数：
            weights: dict，键为场景（'three'/'two_ml_gnn'/'two_ml_seir'/'two_gnn_seir'），
                    值为 {model_name: weight} dict。可只覆盖部分场景，其余保持默认。
                    权重无需归一化，内部会自动归一化。

        示例：
            integrator.set_ensemble_weights({
                'three': {'ml': 0.5, 'seir': 0.25, 'gnn': 0.25}
            })
        """
        if not isinstance(weights, dict):
            raise TypeError("weights 必须为 dict")
        for scenario, w_dict in weights.items():
            if scenario not in self.ensemble_weights:
                raise ValueError(f"未知场景 '{scenario}'，可选: {list(self.ensemble_weights.keys())}")
            if not isinstance(w_dict, dict):
                raise TypeError(f"场景 '{scenario}' 的权重必须为 dict")
            # 归一化
            total = sum(w_dict.values())
            if total <= 0:
                raise ValueError(f"场景 '{scenario}' 权重总和必须 > 0")
            self.ensemble_weights[scenario] = {k: float(v) / total for k, v in w_dict.items()}
        self._weights_source = 'custom'

    def optimize_weights_from_validation(self, validation_auroc, temperature=1.0):
        """基于验证集 AUROC 性能加权集成权重。

        使用 softmax(log(AUROC) / temperature) 归一化，
        AUROC 越高权重越大。temperature → ∞ 时趋于均匀分布，
        temperature → 0 时趋于 argmax（winner-take-all）。

        文献：
            Perrone & Cooper (1993) "When Networks Disagree: Ensemble Methods
            for Hybrid Neural Networks" — 基于性能的集成权重
            Jacobs et al. (1991) "Adaptive Mixtures of Local Experts" — softmax 加权

        参数：
            validation_auroc: dict，键为模型名（'ml'/'seir'/'gnn'），
                             值为该模型在验证集上的 AUROC（0.0~1.0）
            temperature: softmax 温度，控制权重分布的尖锐程度

        返回：
            dict: 更新后的 self.ensemble_weights
        """
        import math
        if not isinstance(validation_auroc, dict):
            raise TypeError("validation_auroc 必须为 dict")
        if temperature <= 0:
            raise ValueError("temperature 必须 > 0")

        # 校验 AUROC 值并计算 log-AUROC
        log_auroc = {}
        for name, auroc in validation_auroc.items():
            if not isinstance(name, str) or name not in ('ml', 'seir', 'gnn'):
                raise ValueError(f"未知模型名 '{name}'，必须为 'ml'/'seir'/'gnn'")
            try:
                auroc_f = float(auroc)
            except (TypeError, ValueError):
                raise TypeError(f"模型 '{name}' 的 AUROC 必须为数值")
            if not (0.0 < auroc_f <= 1.0):
                raise ValueError(f"模型 '{name}' 的 AUROC 必须在 (0, 1] 范围内")
            log_auroc[name] = math.log(auroc_f)

        # softmax(log-AUROC / temperature)
        max_log = max(log_auroc.values())
        exp_vals = {k: math.exp((v - max_log) / temperature) for k, v in log_auroc.items()}
        total_exp = sum(exp_vals.values())
        softmax_weights = {k: v / total_exp for k, v in exp_vals.items()}

        # 应用到各场景
        available = set(softmax_weights.keys())
        # 三方向
        if available == {'ml', 'seir', 'gnn'}:
            self.ensemble_weights['three'] = {k: softmax_weights[k] for k in ('ml', 'seir', 'gnn')}
        # 双方向组合
        two_combos = [
            ('two_ml_gnn', ('ml', 'gnn')),
            ('two_ml_seir', ('ml', 'seir')),
            ('two_gnn_seir', ('gnn', 'seir')),
        ]
        for scenario, (a, b) in two_combos:
            if {a, b}.issubset(available):
                pair = {a: softmax_weights[a], b: softmax_weights[b]}
                total = sum(pair.values())
                self.ensemble_weights[scenario] = {k: v / total for k, v in pair.items()}

        self._weights_source = 'validation'
        return self.ensemble_weights

    def _get_scenario_weights(self, scenario):
        """获取指定场景的权重 dict（内部统一入口）。"""
        return dict(self.ensemble_weights.get(scenario, {}))

    def set_ml_clin_predictor(self, predictor):
        """挂载/卸载临床标签 ML 成员（症状富集门控用）。

        predictor 为 None 或未训练（is_trained=False）时门控不生效，
        ML 方向退回机制标签模型——部署形态无该模型时行为与旧版一致。
        """
        self.ml_clin_predictor = predictor

    def _ml_predictor_for_contact(self, ml_predictor, contact_data):
        """手工门控：按接触者症状状态选择 ML 成员。

        症状富集（has_symptoms 为真）时用临床标签模型——其训练标签机制
        （症状组合驱动的临床规则）与该人群的发病机制更贴合；无症状接触者
        保持机制标签模型（暴露强度驱动）。判据复用特征管线的 _is_yes，
        保证与 22 维特征抽取的口径一致。
        """
        clin = self.ml_clin_predictor
        if (clin is not None
                and getattr(clin, 'is_trained', False)
                and isinstance(contact_data, dict)):
            from ..utils import _is_yes
            if _is_yes(contact_data.get('has_symptoms', 0)):
                return clin, 'clin'
        return ml_predictor, 'mech'
    
    def integrate_predictions(self, ml_predictor, tb_assessment, contact_data, contact_type='family'):
        """
        集成三个方向的预测结果（文献支撑：Breiman, 1996; Wolpert, 1992）
        
        参数：
            ml_predictor: ML预测器
            tb_assessment: 结核病风险评估实例
            contact_data: 单个接触者的数据
            contact_type: 接触者类型 ('family' 或 'social')
            
        返回：
            dict: 包含各方向贡献的集成预测
        """
        results = {}
        
        # 方向一：传统SEIR评分
        seir_results = self._get_seir_results(tb_assessment, contact_data)
        results['seir'] = seir_results
        
        # 方向二：ML模型预测（手工门控：症状富集接触者切换为临床标签模型）
        mlp, ml_member = self._ml_predictor_for_contact(ml_predictor, contact_data)
        ml_results = self._get_ml_results(mlp, contact_data, contact_type)
        ml_results['member'] = ml_member   # 'mech' | 'clin'（可追溯）
        results['ml'] = ml_results
        
        # 方向二：GNN预测（问题三：完整 GNN 或 NumPy 轻量图卷积回退，保证分支始终有输出）
        gnn_results = None
        if getattr(ml_predictor, 'predict_gnn_risk', None) is not None:
            try:
                gnn_results = ml_predictor.predict_gnn_risk(
                    tb_assessment, contact_data, contact_type)
            except Exception as e:
                import logging
                logging.getLogger("tb_risk.integrator").debug(
                    "GNN 预测异常，跳过该方向: %s", e)
                gnn_results = None
            if gnn_results is not None:
                results['gnn'] = gnn_results
        
        # 方向三：SHAP 特征重要性（可解释性分析，非因果推断）
        shap_results = self._get_shap_results(ml_predictor)
        results['shap'] = shap_results
        
        # 集成预测（加权平均，向后兼容）+ 部署护栏回退
        results['ensemble'] = self._weighted_ensemble(results)
        results['ensemble'] = self._apply_fallback(results,
                                                   results['ensemble'])

        # 三层递进架构整合（堆叠+门控，替代加权平均）。
        # 默认不运行第 3 层 SEIR 干预（较重），仅个体+网络层+决策层；
        # 需干预收益时显式调用 integrate_three_layer(run_intervention=True)。
        if self.three_layer_available:
            try:
                results['three_layer'] = self.integrate_three_layer(
                    ml_predictor, tb_assessment, contact_data, contact_type,
                    run_intervention=False)
            except Exception as e:
                import logging
                logging.getLogger("tb_risk.integrator").debug(
                    "三层整合失败，保留加权平均结果: %s", e)
                results['three_layer'] = {'architecture': 'unavailable'}

        return results
    
    def _build_seir_lookup(self, potential_patients, base_infection_prob):
        """构建 SEIR 风险查找表（O(n) 预处理，record_id/_id 两级匹配）。

        问题六：消除 _get_seir_results 的 O(n²) name 匹配。对 family + social
        列表做一次遍历，按 record_id / _id 两种键分别建立 dict。

        问题六-步2：优先将 member dict 转为 ContactRiskResult 再取 record_id 和
        seir_infection_probability，使字段访问经过 schema 契约而非裸 dict.get()。
        record_id 缺失时由 schema 的 fallback_idx 生成稳定合成 id（如 "family_0"），
        字段类型异常在 float() 转换时被捕获。

        问题六-附带：schema 已在 contact_risk_calculator 出口（步1）与
        assessment_service 入口（步3）实际约束数据形状，record_id 为强制字段
        （缺失时由 fallback_idx 合成），不再需要 name 字符串回退匹配。
        重名错配风险随之消除。
        """
        by_record_id = {}
        by_id = {}
        for group in ('family', 'social'):
            for idx, member in enumerate(potential_patients.get(group, [])):
                # 问题六-步2：经 schema 契约访问字段
                result_obj = ContactRiskResult.from_dict(
                    member, contact_type=group, fallback_idx=idx)
                # 保留问题二 fallback 语义：seir 分为 0 时回退到融合值
                # （兼容旧数据无 seir_infection_probability 字段的场景）
                seir_risk = result_obj.seir_infection_probability
                if seir_risk == 0.0:
                    seir_risk = result_obj.infection_probability
                rid = result_obj.record_id
                if rid:
                    by_record_id[rid] = seir_risk
                mid = result_obj.extra.get('_id')
                if mid is not None:
                    by_id[mid] = seir_risk
        return {
            'by_record_id': by_record_id,
            'by_id': by_id,
            'base_infection_prob': base_infection_prob,
        }

    def _get_seir_results(self, tb_assessment, contact_data):
        """获取SEIR模型结果（O(n) 查找，支持 record_id/_id 两级匹配）。

        问题六：消除 O(n²) name 匹配。优先按 record_id 匹配（唯一稳定标识），
        回退到 _id（数据库主键）。缓存以 id(tb_assessment.results) 为 key，
        结果对象重新赋值时自动失效。

        问题六-附带：schema 落地后 record_id 为强制字段（缺失时由 fallback_idx
        合成），name 字符串回退匹配已移除，重名错配风险消除。

        修复双重计入（问题二）：读取纯 SEIR 分 (seir_infection_probability) 而非已融合的
        infection_probability，避免 SEIR 贡献在 fuse_probabilities 与 _weighted_ensemble
        中被加权两次。旧数据无 seir_infection_probability 字段时回退到 infection_probability。
        """
        results_obj = tb_assessment.results if tb_assessment is not None else None
        is_valid_results = isinstance(results_obj, dict)
        if is_valid_results:
            base_infection_prob = results_obj.get('base_infection_probability', 30)
            potential_patients = results_obj.get('potential_patients', {})
            if not isinstance(potential_patients, dict):
                potential_patients = {}
        else:
            base_infection_prob = 30
            potential_patients = {}

        # 查找缓存（按 results 对象身份失效）。
        # 注意：results 为 None/非 dict 时不能使用 id(临时空 dict) — CPython 会回收
        # 短命对象并复用内存地址，可能让后续真实 results 的 id 与之碰撞，导致缓存
        # 命中陈旧数据。改用 -1 哨兵键避开 id 空间。
        cache_key = id(results_obj) if is_valid_results else -1
        if cache_key != self._seir_lookup_cache_key:
            self._seir_lookup = self._build_seir_lookup(
                potential_patients, base_infection_prob)
            self._seir_lookup_cache_key = cache_key
        lookup = self._seir_lookup

        # 两级匹配：record_id → _id
        seir_risk = 0.0
        rid = contact_data.get('record_id') if isinstance(contact_data, dict) else None
        if rid is not None and rid in lookup['by_record_id']:
            seir_risk = lookup['by_record_id'][rid]
        else:
            mid = contact_data.get('_id') if isinstance(contact_data, dict) else None
            if mid is not None and mid in lookup['by_id']:
                seir_risk = lookup['by_id'][mid]

        return {
            'base_infection_prob': base_infection_prob,
            'contact_infection_prob': seir_risk,
            'risk_class': 1 if seir_risk > 50 else 0
        }
    
    def _get_ml_results(self, ml_predictor, contact_data, contact_type):
        """获取ML模型结果"""
        ml_pred = ml_predictor.predict_risk(contact_data, contact_type)
        
        if ml_pred and 'ensemble' in ml_pred:
            return {
                'risk_probability': ml_pred['ensemble']['risk_probability'],
                'risk_class': ml_pred['ensemble']['risk_class'],
                'is_trained': ml_predictor.is_trained,
                'use_real_data': ml_predictor.use_real_data,
                'model_performance': ml_predictor.model_performance
            }
        
        return {
            'risk_probability': 30.0,
            'risk_class': 0,
            'is_trained': ml_predictor.is_trained,
            'use_real_data': ml_predictor.use_real_data,
            'model_performance': ml_predictor.model_performance
        }
    
    def _get_shap_results(self, ml_predictor):
        """获取 SHAP 特征重要性结果（可解释性分析，非因果推断）。"""
        return {
            'has_shap': ml_predictor.shap_explainer is not None,
            'shap_note': '通过 SHAP 值分析特征重要性，提供模型可解释性（非因果推断）'
        }
    
    def _weighted_ensemble(self, results):
        """加权集成各方向结果（文献支撑：Breiman, 1996; Wolpert, 1992）"""
        # 初始化权重字典
        weights = {}
        
        # 收集各方向的风险概率
        risk_probs = []
        risk_weights = []
        
        # ML结果
        if 'ml' in results and 'risk_probability' in results['ml']:
            ml_prob = results['ml']['risk_probability']
            risk_probs.append(ml_prob)
            weights['ml'] = 0.0  # 先占位，后面统一设置
            risk_weights.append(0.0)
        
        # SEIR结果
        if 'seir' in results and 'contact_infection_prob' in results['seir']:
            seir_prob = results['seir']['contact_infection_prob']
            if seir_prob <= 0:
                # SEIR 无有效结果时跳过该方向，不回退为 ML 值
                # 回退会导致 ML 值被双重计数（ML 权重 0.4 + SEIR=ML 权重 0.3 = 0.7）
                # Log 警告并跳过 SEIR 方向
                import logging
                logging.getLogger("tb_risk.integrator").debug(
                    "SEIR 概率为零，跳过 SEIR 方向以避免 ML 双重计数")
            else:
                risk_probs.append(seir_prob)
                weights['seir'] = 0.0
                risk_weights.append(0.0)
        
        # GNN结果
        has_gnn = 'gnn' in results and 'risk_probability' in results['gnn']
        gnn_rescaled_to_100 = False
        if has_gnn:
            gnn_prob = results['gnn']['risk_probability']
            # 尺度断言：GNN 输出应为 0-100 尺度（与 ML/SEIR 一致）
            # 若 GNN 返回 0-1 尺度而未乘 100，加权平均会严重低估 GNN 贡献
            if gnn_prob < 1.0 and gnn_prob > 0:
                import logging
                logging.getLogger("tb_risk.integrator").warning(
                    "GNN 风险概率 %.4f 疑似在 0-1 尺度（预期 0-100），将自动缩放", gnn_prob)
                gnn_prob = gnn_prob * 100.0
                # 缩放事件写入结果（调用方/下游阈值解释必须可知——
                # 仅日志不留档会使 0-100 尺度假设静默失效）
                gnn_rescaled_to_100 = True
            risk_probs.append(gnn_prob)
            weights['gnn'] = 0.0
            risk_weights.append(0.0)
        
        # 设置权重（使用可配置的 self.ensemble_weights，支持消融实验优化）
        # _weights_source 标识权重来源：default / custom / validation
        if has_gnn and len(risk_probs) >= 3:
            # 三方向场景
            w = self._get_scenario_weights('three')
            weights['ml'] = w.get('ml', self.ENSEMBLE_WEIGHT_ML)
            weights['seir'] = w.get('seir', self.ENSEMBLE_WEIGHT_SEIR)
            weights['gnn'] = w.get('gnn', self.ENSEMBLE_WEIGHT_GNN)
            risk_weights[0] = weights['ml']
            risk_weights[1] = weights['seir']
            risk_weights[2] = weights['gnn']
        elif len(risk_probs) >= 2:
            if 'gnn' in weights and 'ml' in weights:
                w = self._get_scenario_weights('two_ml_gnn')
                weights['ml'] = w.get('ml', self.ENSEMBLE_WEIGHT_ML_TWO)
                weights['gnn'] = w.get('gnn', self.ENSEMBLE_WEIGHT_OTHER_TWO)
                risk_weights[0] = weights['ml']
                risk_weights[1] = weights['gnn']
            elif 'ml' in weights and 'seir' in weights:
                w = self._get_scenario_weights('two_ml_seir')
                weights['ml'] = w.get('ml', self.ENSEMBLE_WEIGHT_ML_TWO)
                weights['seir'] = w.get('seir', self.ENSEMBLE_WEIGHT_OTHER_TWO)
                risk_weights[0] = weights['ml']
                risk_weights[1] = weights['seir']
            elif 'gnn' in weights and 'seir' in weights:
                w = self._get_scenario_weights('two_gnn_seir')
                weights['gnn'] = w.get('gnn', 0.5)
                weights['seir'] = w.get('seir', 0.5)
                # risk_probs 顺序为 [seir, gnn]（ml 缺失）
                risk_weights[0] = weights['seir']
                risk_weights[1] = weights['gnn']
        elif len(risk_probs) == 1:
            # 只有一个结果时，权重为1.0
            if 'ml' in weights:
                weights['ml'] = 1.0
                risk_weights[0] = 1.0
            elif 'gnn' in weights:
                weights['gnn'] = 1.0
                risk_weights[0] = 1.0
            else:
                weights['seir'] = 1.0
                risk_weights[0] = 1.0
        
        # 归一化权重（确保总和为1.0）
        if risk_weights:
            total_w = sum(risk_weights)
            if total_w > 0:
                risk_weights = [w / total_w for w in risk_weights]
            else:
                risk_weights = [1.0 / len(risk_weights)] * len(risk_weights)
            
            # 计算加权平均
            if len(risk_probs) > 0 and len(risk_probs) == len(risk_weights):
                ensemble_prob = sum(p * w for p, w in zip(risk_probs, risk_weights))
            else:
                ensemble_prob = 30.0
        else:
            ensemble_prob = 30.0

        if not risk_probs:
            return {
                'weights': {}, 'risk_probability': None, 'risk_class': None,
                'integration_note': '所有方向均不可用',
                'has_gnn_contribution': False
            }

        n_models = len(risk_probs)
        source_map = {
            'default': '经验默认权重',
            'custom': '自定义权重',
            'validation': '验证集AUROC性能加权',
        }
        source_desc = source_map.get(self._weights_source, self._weights_source)

        # 零权重方向披露：权重为 0 的方向分数仅参考收集、不进入加权
        # （如 'three' 场景 SEIR 零权重为冻结实验决策：SEIR 定位人群层
        # 而非个体排序，见 constants.DEFAULT_ENSEMBLE_WEIGHTS 注释）。
        # 不披露会使"三方向集成"名不副实——调用方看到 seir: 0.0
        # 却不知其语义，易误读为配置错误或遗漏。
        zero_weight_dirs = [k for k, v in weights.items()
                            if isinstance(v, (int, float)) and v <= 0.0]
        effective_dirs = n_models - len(zero_weight_dirs)
        integration_note = f'基于{source_desc}的{effective_dirs}方向加权集成'
        if zero_weight_dirs:
            integration_note += (
                f'（{"/".join(sorted(zero_weight_dirs))} 方向零权重：'
                f'分数仅参考收集，不进入加权，见 DEFAULT_ENSEMBLE_WEIGHTS）')

        # 确保概率在合理范围内
        ensemble_prob = max(0.0, min(100.0, ensemble_prob))
        ensemble_class = 1 if ensemble_prob > 50 else 0

        return {
            'weights': weights,
            'risk_probability': float(ensemble_prob),
            'risk_class': ensemble_class,
            'integration_note': integration_note,
            'has_gnn_contribution': has_gnn,
            'weights_source': self._weights_source,
            # 缩放事件留档：GNN 分数从 0-1 尺度被自动 ×100 时为 True，
            # 下游按 0-100 尺度解释阈值时必须检查此标记
            'gnn_rescaled_to_100': gnn_rescaled_to_100,
            # 零权重方向名单（分数被收集但未参与加权）
            'zero_weight_directions': zero_weight_dirs,
        }

    # ==================== 三层递进架构（堆叠 + 门控，替代加权平均） ====================

    @property
    def three_layer_available(self) -> bool:
        """三层递进架构是否可用。"""
        return THREE_LAYER_AVAILABLE and ThreeLayerArchitecture is not None

    def integrate_three_layer(self, ml_predictor, tb_assessment, contact_data,
                              contact_type='family', run_intervention=False,
                              population=10000, t_horizon_days=730,
                              random_state=42):
        """三层递进架构整合（个体基础层 → 网络增强层 → 社区干预层）。

        与 ``integrate_predictions``（加权平均）的区别：
          - 第 1 层输出个体基线概率 P_base；
          - 第 2 层以 P_base 为特征、学习**残差**（网络增量），不再并列打分；
          - 第 3 层 SEIR 只做人群层干预反事实模拟（不输出个体概率）；
          - 决策层用"堆叠 + 门控"联合（个体概率 + 网络增量 + 干预收益），
            而非概率层面线性平均。

        参数：
            ml_predictor: MLRiskPredictor
            tb_assessment: 风险评估实例
            contact_data (dict): 目标接触者数据
            contact_type (str): 'family' / 'social'
            run_intervention (bool): 是否运行第 3 层 SEIR 干预模拟（较重）
            population / t_horizon_days / random_state: 第 3 层参数

        返回：
            dict: 三层流水线报告（layers / decision / summary），
                架构不可用时返回 {'architecture': 'unavailable'}
        """
        if not self.three_layer_available:
            return {'architecture': 'unavailable'}

        arch = ThreeLayerArchitecture()
        try:
            report = arch.run_pipeline(
                ml_predictor, tb_assessment, contact_data,
                contact_type=contact_type,
                run_intervention=run_intervention,
                population=population,
                t_horizon_days=t_horizon_days,
                random_state=random_state,
            )
            report['provider'] = 'ThreeDirectionIntegrator'
            return report
        except Exception as e:
            import logging
            logging.getLogger("tb_risk.integrator").exception(
                "三层递进架构整合失败: %s", e)
            return {
                'architecture': 'three_layer',
                'status': 'error',
                'error': str(e),
            }


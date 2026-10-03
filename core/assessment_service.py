#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风险评估服务（编排层）

零依赖 tkinter 或任何 GUI 库。可在 CLI/Docker/Web 环境中直接使用。

职责：
- 编排风险评估流程：预处理 → 患者评分 → 接触者风险 → 汇总 → SEIR/ML（可选）
- 管理评估状态、结果缓存与回调
- 作为对外统一 API（门面），将具体计算委托给协作类：
    * PatientScorer        — 患者评分与个体风险
    * ContactRiskCalculator — 接触者风险与潜在患者生成
    * SEIRIntegration       — SEIR 概率/置信度/模拟
"""

import copy
import logging
import threading
from typing import Any, Dict, List, Optional

from ..scoring.engine import ScoringEngine
from ..constants import (
    LATENT_BASELINE,
    SEIR_DEFAULT_BETA, SEIR_DEFAULT_SIGMA, SEIR_DEFAULT_GAMMA,
    SEIR_INITIAL_S, SEIR_INITIAL_E, SEIR_INITIAL_I, SEIR_INITIAL_R,
    SEIR_SIMULATION_DAYS, SEIR_N_TRAJECTORIES, SEIR_RANDOM_SEED,
)  # 问题八-3 + 常量迁移：单一真值源
from ..schemas import PatientRecord, AssessmentResult  # 问题六-步3/优先级2：schema 契约
from .probability import sigmoid_total_score, safe_float
from .patient_scorer import PatientScorer
from .contact_risk_calculator import ContactRiskCalculator
from .seir_integration import SEIRIntegration

LOGGER = logging.getLogger("tb_risk.core")


class _MLPredictionContext:
    """轻量 ML 预测只读上下文。

    供 RiskAssessmentService 在无 GUI 控制器时运行 run_ml_prediction_loop。
    仅提供 integrator._get_seir_results 所需的 ``.results`` 属性；
    GNN 方向在预测器未训练 GNN 时由 predict_gnn_risk 前置返回 None 自动跳过。
    """

    __slots__ = ('results',)

    def __init__(self, results):
        self.results = results


class RiskAssessmentService:
    """结核病风险评估服务（编排层 / 门面）

    用法：
        service = RiskAssessmentService()
        result = service.assess(patient_info, family_members, social_contacts)

    可用于：
    - CLI 模式（无需 tkinter）
    - GUI 控制器层（通过 TB_Risk_Assessment 调用）
    - REST API 后端
    - Docker 容器部署
    """

    # 潜伏→发病基线（问题八-3：单一真值源，引用 constants.LATENT_BASELINE）
    # 门面保留 LATENT_TO_ACTIVE_BASELINE 类属性供 core.__init__ 兼容引用
    LATENT_TO_ACTIVE_BASELINE = LATENT_BASELINE

    def __init__(self, scoring_engine=None, localizer=None,
                 seir_param_uncertainty=None, contact_labels=None,
                 ml_random_state: int = 42,
                 enable_karamay: bool = False):
        """初始化评估服务

        参数：
            scoring_engine: ScoringEngine | None — 评分引擎实例
            localizer: KaramayLocalizer | None — 本土化适配器
            seir_param_uncertainty: SEIRParameterUncertainty | None — SEIR参数不确定性
            contact_labels: dict | None — 接触者标签 {_id: 0/1}
            ml_random_state: int — ML 预测器随机种子（问题十-2：GUI 通过此参数
                将自身 random_state 传入 service，使共享 ml_predictor 实例
                的可复现性与 GUI 旧行为一致）
            enable_karamay: bool — 是否启用克拉玛依本土化（问题十-3：当 localizer
                为 None 且此标志为 True 时，service 通过 create_default_localizer()
                统一创建单例实例，消除 GUI 用 get_instance() / CLI 用 KaramayLocalizer()
                的行为差异）
        """
        if scoring_engine is None:
            scoring_engine = ScoringEngine()
        self._engine = scoring_engine
        # 问题十-3：localizer 创建统一固化在 service 层
        # 若调用方未显式注入 localizer 但启用 karamay，则使用单例工厂创建
        if localizer is None and enable_karamay:
            localizer = self.create_default_localizer()
        self._localizer = localizer
        self._ml_random_state = ml_random_state

        # 协作器（依赖注入 + 组合）
        self._seir_integration = SEIRIntegration(
            seir_param_uncertainty=seir_param_uncertainty)
        self._patient_scorer = PatientScorer(engine=scoring_engine)
        self._contact_risk_calc = ContactRiskCalculator(
            localizer=localizer,
            contact_labels=contact_labels,
            seir_integration=self._seir_integration,
            progress_callback=self._report_progress,
        )

        # 评估结果缓存（线程安全）
        self._results_lock = threading.Lock()
        self._results: Dict[str, Any] = {}
        self._ml_results: Dict[str, Any] = {}
        self._seir_results: Dict[str, Any] = {}
        self._assessment_summary: Dict[str, Any] = {}

        # 三方向集成器（service 持有，GUI 通过 get_integrator() 取只读引用）
        # 统一 GUI 与 CLI 的 ML 集成行为与结果结构
        self._integrator: Optional[Any] = None

        # ML 预测器（惰性创建 + 缓存）
        # 单一实例供 GUI/CLI 共享，避免重复实例化与训练
        # （问题十-2：消除 GUI 自建 ml_predictor 与 service 内部新建的双路径）
        self._ml_predictor: Optional[Any] = None

        # 临床标签 ML 成员（症状富集门控，惰性加载冻结 checkpoint）
        self._ml_clin_predictor: Optional[Any] = None

        # 回调接口（用于 GUI 通知，可选）
        self._on_progress: Optional[callable] = None
        self._on_complete: Optional[callable] = None

    # ==================== 回调接口 ====================

    def set_progress_callback(self, callback: Optional[callable]) -> None:
        """设置进度回调"""
        self._on_progress = callback

    def set_complete_callback(self, callback: Optional[callable]) -> None:
        """设置完成回调"""
        self._on_complete = callback

    def _report_progress(self, progress: int, message: str = "") -> None:
        """报告进度（线程安全）"""
        if self._on_progress:
            try:
                self._on_progress(progress, message)
            except Exception as e:
                LOGGER.debug("进度回调执行失败: %s", e)

    # ==================== 核心评估 API ====================

    def assess(
        self,
        patient_info: Optional[Dict[str, Any]] = None,
        family_members: Optional[List[Dict[str, Any]]] = None,
        social_contacts: Optional[List[Dict[str, Any]]] = None,
        use_ml: bool = False,
        use_seir: bool = False,
    ) -> Dict[str, Any]:
        """执行完整风险评估

        参数：
            patient_info: dict | None
            family_members: list[dict] | None
            social_contacts: list[dict] | None
            use_ml: bool
            use_seir: bool

        返回：
            dict: {
                'potential_patients': {'family': [...], 'social': [...]},
                'summary': {...},
                'patient_score': float,
                'ml_results': dict | None,
                'seir_results': dict | None,
            }
        """
        family_members = family_members or []
        social_contacts = social_contacts or []

        self._report_progress(0, "开始评估...")

        patient_info = self._patient_scorer.preprocess_patient(patient_info)
        family_members = self._patient_scorer.preprocess_contacts(family_members, 'family')
        social_contacts = self._patient_scorer.preprocess_contacts(social_contacts, 'social')

        # 问题六-步3：入口 schema 约束。
        # 用 PatientRecord.from_dict() 包装传入的 patient_info，在入口处校验契约。
        # to_dict() 保留裸 dict 形状以兼容既有下游消费者（patient_scorer /
        # contact_risk_calculator / seir_integration 等均按 dict 访问）。
        # from_dict 宽松构造：basic_info 缺失补 {}，FCI/SNC 缺失补 0.0，
        # 其余字段透传到 meta；类型异常在 float() 转换时被捕获。
        #
        # 注意：仅对 truthy（非空）patient_info 包装。preprocess_patient(None)
        # 返回 {}，若包装为 {'basic_info':{},'FCI':0.0,'SNC':0.0}（truthy），
        # 会使 compute_patient_score 的 `if not patient_info: return 0.0`
        # 早返回失效，从默认值计算出非零分。空输入保持原样透传。
        if patient_info:
            patient_info = PatientRecord.from_dict(patient_info).to_dict()

        self._report_progress(10, "计算患者风险评分...")
        patient_score_result = self._patient_scorer.compute_patient_score(patient_info)
        patient_score = patient_score_result['total_score']
        base_infection_probability = patient_score_result['base_infection_probability']

        # 提取评分因子用于个体风险判定
        basic_info = patient_info.get('basic_info', {}) if patient_info else {}
        fci_score = safe_float(patient_info.get('FCI', 0.0), 0.0) if patient_info else 0.0
        snc_score = safe_float(patient_info.get('SNC', 0.0), 0.0) if patient_info else 0.0
        flp_percentage = safe_float(basic_info.get('flp_percentage', 0.0), 0.0)
        hrsp_percentage = safe_float(basic_info.get('hrsp_percentage', 0.0), 0.0)
        ftd_days = safe_float(basic_info.get('delay_days', 0), 0.0)

        # 计算个体风险等级
        individual_risks_result = self._patient_scorer.compute_individual_risks(
            patient_info, fci_score, snc_score, flp_percentage,
            hrsp_percentage, ftd_days, base_infection_probability)

        self._report_progress(20, "计算接触者风险...")
        patient_treatment_days = self._patient_scorer.get_treatment_days(patient_info)
        potential_patients = self._contact_risk_calc.generate_potential_patients(
            patient_info, family_members, social_contacts,
            patient_treatment_days, individual_risks_result
        )

        self._report_progress(70, "汇总结果...")
        summary = self._build_summary(
            potential_patients, patient_score,
            individual_risks_result, patient_score_result)

        seir_results = None
        if use_seir:
            self._report_progress(80, "SEIR 模拟中...")
            seir_results = self._seir_integration.run_seir_simulation(
                patient_info, family_members, social_contacts
            )

        # 问题一：先构建不含 ML 结果的 provisional result 并写入 self._results，
        # 再调用 _run_ml_predictions。此前 _run_ml_predictions 在 self._results 赋值
        # 之前被调用，导致 CLI 首次 ``assess --ml`` 时 ML 预测读到空字典，
        # run_ml_prediction_loop 因无 potential_patients 而返回空结果。
        # GUI 不受影响（先 assess(use_ml=False) 再单独调 _run_ml_predictions）。
        # 现两种入口行为一致：ML 预测始终能读到已填充的 potential_patients。
        result = {
            'potential_patients': potential_patients,
            'summary': summary,
            'patient_score': patient_score,
            'base_infection_probability': base_infection_probability,
            'individual_risks': individual_risks_result['individual_risks'],
            'overall_risk': individual_risks_result['overall_risk'],
            'overall_suggestion': individual_risks_result['overall_suggestion'],
            'total_score': patient_score_result['total_score'],
            'weights': patient_score_result['weights'],
            'patient_risk_multiplier': patient_score_result['patient_risk_multiplier'],
            'ml_results': None,
            'seir_results': seir_results,
            'synthetic_data_warning': False,
            'model_data_source': 'unknown',
        }

        # 提前写入 self._results，使 _run_ml_predictions 能读到 potential_patients
        with self._results_lock:
            self._results = result
            if seir_results:
                self._seir_results = seir_results
            self._assessment_summary = summary

        ml_results = None
        synthetic_data_warning = False
        model_data_source = 'unknown'
        if use_ml:
            self._report_progress(85, "ML 预测中...")
            # 优先级3：显式传入 potential_patients 与 base_infection_probability，
            # 消除对 self._results 赋值时序的隐式依赖
            ml_results = self._run_ml_predictions(
                patient_info, family_members, social_contacts,
                potential_patients=potential_patients,
                base_infection_probability=base_infection_probability,
            )
            # 将 ML 结果回填到 result 与 self._ml_results
            result['ml_results'] = ml_results
            if ml_results:
                with self._results_lock:
                    self._ml_results = ml_results
                # 检测是否使用合成数据训练的模型（给 CLI/API 用户警示）
                model_perf = ml_results.get('model_performance', {})
                if model_perf:
                    # 取任意一个模型条目的 data_source 作为整体数据源
                    first_key = next(iter(model_perf))
                    first_entry = model_perf[first_key] if isinstance(model_perf[first_key], dict) else {}
                    model_data_source = first_entry.get('data_source', 'unknown')
                    synthetic_data_warning = (model_data_source == 'synthetic')
                    # 如果任一模型基于合成数据，整体标记为合成数据警告
                    for _k, _v in model_perf.items():
                        if isinstance(_v, dict) and _v.get('data_source') == 'synthetic':
                            synthetic_data_warning = True
                            break

        # 合成数据警示标志（顶级字段，方便 CLI/API 直接读取）
        result['synthetic_data_warning'] = synthetic_data_warning
        result['model_data_source'] = model_data_source

        self._report_progress(100, "评估完成")

        if self._on_complete:
            try:
                self._on_complete(result)
            except Exception as e:
                LOGGER.debug("完成回调执行失败: %s", e)

        # 优先级2：出口 schema 约束。
        # 用 AssessmentResult.from_dict() 包装返回值，使 AssessmentResult
        # dataclass 不再空转，评估结果在出口处符合 schema 约束。
        # to_dict() 保留裸 dict 形状以兼容既有消费者（GUI/CLI/REST 均按 dict
        # 访问）。from_dict 宽松构造：已知字段进 dataclass 槽，其余透传到 extra；
        # to_dict 深拷贝输出，返回值与 self._results 内部状态独立（外部修改
        # 不影响内部状态）。
        #
        # 注意：self._results 保持原始 result 引用（未包装），供 get_last_results
        # 与 _run_ml_predictions 内部使用。包装仅作用于返回值，不影响内部状态。
        return AssessmentResult.from_dict(result).to_dict()

    def get_last_results(self) -> Dict[str, Any]:
        """获取最近一次评估结果"""
        with self._results_lock:
            return copy.deepcopy(self._results)

    def get_last_summary(self) -> Dict[str, Any]:
        """获取最近一次评估摘要"""
        with self._results_lock:
            return copy.deepcopy(self._assessment_summary)

    # ==================== 摘要 ====================

    def _build_summary(
        self, potential_patients: Dict, patient_score: float,
        individual_risks_result: Optional[Dict] = None,
        patient_score_result: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """构建评估摘要（增强版，包含个体风险等级）"""
        family = potential_patients.get('family', [])
        social = potential_patients.get('social', [])
        all_patients = family + social

        high_risk = [p for p in all_patients
                     if p.get('priority') in ('极高', '高', '较高')]
        medium_risk = [p for p in all_patients
                       if p.get('priority') == '中']

        summary = {
            'total_contacts': len(all_patients),
            'family_contacts': len(family),
            'social_contacts': len(social),
            'high_risk_count': len(high_risk),
            'medium_risk_count': len(medium_risk),
            'patient_score': patient_score,
            'patient_infection_probability': sigmoid_total_score(patient_score),
            'top_risk_contacts': sorted(
                high_risk[:5],
                key=lambda x: x.get('disease_probability', 0),
                reverse=True
            ),
        }

        # 包含个体风险等级和综合评估
        if individual_risks_result:
            summary['individual_risks'] = individual_risks_result.get('individual_risks', {})
            summary['overall_risk'] = individual_risks_result.get('overall_risk', '')
            summary['overall_suggestion'] = individual_risks_result.get('overall_suggestion', '')

        if patient_score_result:
            summary['total_score'] = patient_score_result.get('total_score', 0.0)
            summary['base_infection_probability'] = patient_score_result.get(
                'base_infection_probability', 0.0)

        return summary

    # ==================== ML（可选） ====================

    def get_integrator(self):
        """获取三方向集成器（只读引用）。

        service 持有 ThreeDirectionIntegrator 实例，GUI 通过此方法取引用，
        使 GUI 与 CLI 共享同一集成器与权重，保证 ML 集成行为一致。
        首次调用时惰性创建，并挂载临床标签 ML 成员（症状富集门控；
        冻结 checkpoint 缺失时门控自动退化为单 ML 成员，行为同旧版）。
        """
        if self._integrator is None:
            from .integrator import ThreeDirectionIntegrator
            self._integrator = ThreeDirectionIntegrator()
            self._integrator.set_ml_clin_predictor(self.ml_clin_predictor)
        return self._integrator

    @property
    def ml_clin_predictor(self):
        """临床标签 ML 成员（惰性加载冻结 checkpoint，缺失时 None）。

        部署症状富集门控的第二 ML 成员：has_symptoms 接触者的 ML 票位
        由该模型（临床规则标签训练，2026-08-24 主模型冻结，见
        data/processed/primary_model_freeze_20260824.json）打分。
        依据：跨基准汇总 run 20260823_085303——学权重融合跨域均值
        0.797 > 任何单一模型（最高 0.773）；门控为该结论的部署近似。
        checkpoint 文件不存在或加载失败时返回 None（门控不生效）。
        """
        if self._ml_clin_predictor is None:
            import os
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'training_archive', 'models',
                'best_ml_20260824_235828_ns12509_cv5.joblib')
            try:
                if os.path.exists(model_path):
                    from ..scoring.predictor import MLRiskPredictor
                    p = MLRiskPredictor(
                        random_state=self._ml_random_state)
                    if p.load_model(model_path):
                        self._ml_clin_predictor = p
                        LOGGER.info(
                            "临床标签 ML 成员已加载（症状富集门控）: %s",
                            model_path)
                    else:
                        LOGGER.warning(
                            "临床标签 ML checkpoint 加载失败，门控退化为单 ML: %s",
                            model_path)
                else:
                    LOGGER.debug(
                        "临床标签 ML checkpoint 不存在，门控退化为单 ML: %s",
                        model_path)
            except Exception as e:  # noqa: BLE001 — 门控为可选增强，失败不阻断评估
                LOGGER.warning(
                    "临床标签 ML 成员初始化异常（%s: %s），门控退化为单 ML",
                    type(e).__name__, e)
        return self._ml_clin_predictor

    @property
    def ml_predictor(self):
        """ML 风险预测器（惰性创建 + 缓存）。

        单一实例供 GUI/CLI 共享，避免重复实例化与训练。
        首次访问时创建；后续访问返回同一实例。

        问题十-2：消除 GUI 自建 ``self.ml_predictor = MLRiskPredictor(...)``
        与 service 内部 ``MLRiskPredictor()`` 双路径并存的问题——
        GUI 现通过 ``service.ml_predictor`` 取引用，保证训练/预测上下文一致。
        """
        if self._ml_predictor is None:
            from ..scoring.predictor import MLRiskPredictor
            self._ml_predictor = MLRiskPredictor(random_state=self._ml_random_state)
        return self._ml_predictor

    def simulate_seir(
        self,
        population: int = 1000,
        days: int = int(SEIR_SIMULATION_DAYS),
        beta: float = SEIR_DEFAULT_BETA,
        trajectories: int = SEIR_N_TRAJECTORIES,
        sigma: Optional[float] = None,
        gamma: Optional[float] = None,
        initial_state: Optional[tuple] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """SEIR 传播动力学模拟（门面委托 StochasticSEIRModel）。

        问题十-1：CLI/GUI 通过此统一入口调用 SEIR 模拟，
        避免界面层直接 ``from tb_risk.seir.stochastic import StochasticSEIRModel``
        越层访问实现类。返回结构与原 CLI 命令保持兼容。

        参数：
            population: 模拟人口数
            days: 模拟天数（默认 SEIR_SIMULATION_DAYS=365 天）
            beta: 传播率（默认 SEIR_DEFAULT_BETA≈0.3，对应 R0≈2-4）
            trajectories: 轨迹数量（默认 SEIR_N_TRAJECTORIES=10）
            sigma: 潜伏→发病转换率（默认 SEIR_DEFAULT_SIGMA=1/30，文献：Vynnycky & Fine 1997）
            gamma: 恢复率（默认 SEIR_DEFAULT_GAMMA=1/180，文献：Ragonnet et al. 2021）
            initial_state: 初始状态（默认 (SEIR_INITIAL_S, SEIR_INITIAL_E, SEIR_INITIAL_I, SEIR_INITIAL_R) = (0.99, 0.01, 0, 0)）

        返回：
            dict: {'trajectories': ndarray, 'parameters': {...}}
        """
        try:
            from ..seir.stochastic import StochasticSEIRModel
        except ImportError as e:
            raise ImportError("SEIR 模块不可用（需要 numpy + scipy）") from e

        if sigma is None:
            sigma = SEIR_DEFAULT_SIGMA
        if gamma is None:
            gamma = SEIR_DEFAULT_GAMMA
        if initial_state is None:
            initial_state = (SEIR_INITIAL_S, SEIR_INITIAL_E, SEIR_INITIAL_I, SEIR_INITIAL_R)

        model = StochasticSEIRModel(
            population=population,
            seed=seed if seed is not None else SEIR_RANDOM_SEED,
        )
        trajectories_arr = model.simulate_multiple(
            initial_state=initial_state,
            t_span=(0.0, float(days)),
            dt=1.0,
            beta=beta,
            sigma=sigma,
            gamma=gamma,
            n_trajectories=trajectories,
        )
        return {
            'trajectories': trajectories_arr,
            'parameters': {
                'population': population, 'days': days, 'beta': beta,
                'trajectories': trajectories, 'sigma': sigma, 'gamma': gamma,
            },
        }

    def train_ml_models(
        self,
        n_samples: int = 2000,
        enable_hyperopt: bool = False,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """训练 ML 风险预测模型（门面委托 ml_predictor）。

        问题十-1：CLI/GUI 通过此统一入口调用训练，
        避免界面层直接 ``from tb_risk.scoring.predictor import MLRiskPredictor``
        越层访问实现类。

        参数：
            n_samples: 训练样本数
            enable_hyperopt: 是否启用超参数优化
            save_path: 模型保存路径（可选）

        返回：
            dict: {'model_count': int, 'is_trained': bool, 'save_path': str|None}
        """
        predictor = self.ml_predictor
        predictor.train_models(n_samples=n_samples, enable_hyperopt=enable_hyperopt)

        if save_path:
            predictor.save_model(save_path)

        return {
            'model_count': len(predictor.models),
            'is_trained': predictor.is_trained,
            'save_path': save_path,
        }

    def train_gnn_models(
        self,
        n_samples: int = 2000,
        enable_hyperopt: bool = False,
        n_epochs: int = 50,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """训练 GNN 网络风险模型（门面委托 ml_predictor）。

        M 级审计修复（CLI 暴露缺口）：--gnn-hyperopt 等 GNN 训练开关
        此前仅 GUI 可达，CLI 只有 --hyperopt（ML 路径）。此门面为
        CLI train-gnn 子命令提供统一入口（问题十-1 同语义：界面层
        不直接导入实现类）。

        参数：
            n_samples: 合成图样本量
            enable_hyperopt: 是否启用 GNN 超参搜索（受"样本量 ≥ 5000
                再调参"项目约束，门槛下自动跳过——见 gnn_training 实现）
            n_epochs: 训练轮数
            save_path: GNN 模型保存路径（可选）

        返回：
            dict: {'gnn_is_trained': bool, 'save_path': str|None,
                   'hyperopt': bool}
        """
        predictor = self.ml_predictor
        ok = predictor.train_gnn(
            n_samples=n_samples, enable_hyperopt=enable_hyperopt,
            n_epochs=n_epochs)

        if ok and save_path:
            predictor.save_gnn_model(save_path)

        return {
            'gnn_is_trained': bool(
                getattr(predictor, 'gnn_is_trained', False)),
            'save_path': save_path,
            'hyperopt': enable_hyperopt,
        }

    def _run_ml_predictions(
        self, patient_info: Dict,
        family_members: List[Dict],
        social_contacts: List[Dict],
        potential_patients: Optional[Dict] = None,
        base_infection_probability: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """运行结构化 ML 预测（CLI/GUI 统一入口）。

        统一编排：训练预测器 → 设置患者上下文 → 构建 integrator →
        运行 run_ml_prediction_loop，产出与 GUI 一致的结构化 ml_results
        (family/social/model_performance/gnn/ensemble)。

        GNN 方向：CLI 下预测器未训练 GNN，predict_gnn_risk 前置返回 None，
        GNN 字段为空列表；与 GUI 训练 GNN 后的结果结构仍保持一致（同键）。

        问题十-2：改用 ``self.ml_predictor`` 单一实例，避免每次调用新建并训练；
        若 GUI 已训练过同一实例，此处复用其训练成果。

        优先级3：potential_patients 与 base_infection_probability 作为显式参数
        传入，消除对 self._results 赋值时序的隐式依赖。未传时向后兼容回退到
        self._results（供 GUI 等既有调用方使用）。
        """
        try:
            predictor = self.ml_predictor
            if not predictor.is_trained:
                predictor.train_models(n_samples=1000, enable_hyperopt=False)
        except ImportError:
            LOGGER.info("ML 模块不可用，跳过预测")
            return None
        except Exception as e:
            LOGGER.warning("ML 预测失败: %s", e)
            return None

        if not predictor.is_trained:
            return None

        # 设置患者上下文（与 GUI 路径一致）
        basic_info = (patient_info or {}).get('basic_info', {})
        predictor.set_patient_context(
            patient_ftd=basic_info.get('delay_days', 0),
            patient_cough_freq=basic_info.get('cough_freq', 0),
            contact_count=len(family_members or []) + len(social_contacts or []),
        )

        integrator = self.get_integrator()

        # 优先级3：优先使用显式参数构建 results dict，消除对 self._results
        # 赋值时序的隐式依赖。integrator._get_seir_results 与
        # run_ml_prediction_loop 仅读取 potential_patients 和
        # base_infection_probability，这两个字段足以构建最小上下文。
        # 未传显式参数时回退到 self._results（向后兼容）。
        if potential_patients is not None:
            results_dict = {
                'potential_patients': potential_patients,
                'base_infection_probability': base_infection_probability,
            }
        else:
            results_dict = self._results

        # 轻量只读上下文：integrator._get_seir_results 仅读取 .results
        # GNN 方向因预测器未训练 GNN 而自动跳过
        context = _MLPredictionContext(results=results_dict)

        from .ml_runner import run_ml_prediction_loop
        return run_ml_prediction_loop(
            assessment=context, ml_predictor=predictor,
            integrator=integrator, results=results_dict,
            data_lock=self._results_lock)

    # ==================== 门面委托 API ====================
    # 以下方法将具体计算委托给协作类，保持对外 API 稳定，
    # 供 assessment.py（GUI）与测试用例直接调用。

    # --- SEIRIntegration 委托 ---

    def calculate_time_dependent_risk(self, contact, patient_treatment_days,
                                      is_mdr=False):
        """计算基于 SEIR 模型的时间依赖性基础风险系数（0-1）

        参数：
            contact: 接触者记录
            patient_treatment_days: 患者已治疗天数
            is_mdr: 源病例是否 MDR（genexpert_rif=resistant）
        """
        return self._seir_integration.calculate_time_dependent_risk(
            contact, patient_treatment_days, is_mdr=is_mdr)

    def compute_seir_probability(self, contact, patient_treatment_days,
                                 is_mdr=False):
        """从 SEIR 后验推断计算感染概率 P_seir (0-1)

        参数：
            contact: 接触者记录
            patient_treatment_days: 患者已治疗天数
            is_mdr: 源病例是否 MDR（genexpert_rif=resistant）
        """
        return self._seir_integration.compute_seir_probability(
            contact, patient_treatment_days, is_mdr=is_mdr)

    def compute_seir_confidence_weight(self):
        """计算 SEIR 推断的置信度权重 w_seir ∈ [0, 1]"""
        return self._seir_integration.compute_seir_confidence_weight()

    # --- ContactRiskCalculator 委托 ---

    def generate_potential_patients(
        self,
        patient_info: Dict,
        family_members: List[Dict],
        social_contacts: List[Dict],
        patient_treatment_days: float,
        individual_risks_result: Optional[Dict] = None,
    ) -> Dict[str, List[Dict]]:
        """生成潜在患者列表（公开 API，供 GUI 调用）"""
        return self._contact_risk_calc.generate_potential_patients(
            patient_info, family_members, social_contacts,
            patient_treatment_days, individual_risks_result
        )

    def compute_single_contact_risk(
        self, contact: Dict, patient_treatment_days: float,
        default_contact_distance: str = 'medium',
        is_social_contact: bool = False,
        is_mdr: bool = False,
    ) -> Dict[str, float]:
        """计算单个接触者的风险评分和概率（公开 API，供 GUI 调用）

        参数：
            contact: 接触者记录
            patient_treatment_days: 患者已治疗天数
            default_contact_distance: 默认接触距离
            is_social_contact: 是否社会接触者
            is_mdr: 源病例是否 MDR（genexpert_rif=resistant）
        """
        return self._contact_risk_calc.compute_single_contact_risk(
            contact, patient_treatment_days,
            default_contact_distance=default_contact_distance,
            is_social_contact=is_social_contact,
            is_mdr=is_mdr)

    def calculate_bcg_protection(self, age, bcg_vaccinated=True, years_since_vaccination=None):
        """计算 BCG 接种的保护效力"""
        return self._contact_risk_calc.calculate_bcg_protection(
            age, bcg_vaccinated, years_since_vaccination)

    def apply_karamay_enhancements(self, infection_prob, disease_prob, contact_data):
        """应用克拉玛依本土化增强"""
        return self._contact_risk_calc.apply_karamay_enhancements(
            infection_prob, disease_prob, contact_data)

    def enhance_recommendation_karamay(self, contact, priority, base_recommendation):
        """通过克拉玛依本土化增强建议文本"""
        return self._contact_risk_calc.enhance_recommendation_karamay(
            contact, priority, base_recommendation)

    # --- PatientScorer 委托 ---

    def calculate_fci_score(self, family_members, family_living_conditions):
        """计算家庭内密切接触强度（FCI）综合得分（0-10分）"""
        return self._patient_scorer.calculate_fci_score(
            family_members, family_living_conditions)

    def calculate_snc_score(self, social_contacts):
        """计算社会接触网络中心性（SNC）综合得分（0-10分）"""
        return self._patient_scorer.calculate_snc_score(social_contacts)

    def calculate_patient_risk_multiplier(self, sputum_smear, has_cavity, untreated):
        """计算患者类型的整体风险乘数（上限3.0）"""
        return self._patient_scorer.calculate_patient_risk_multiplier(
            sputum_smear, has_cavity, untreated)

    # ==================== 兼容层 ====================

    def set_seir_param_uncertainty(self, seir_param_uncertainty) -> None:
        """设置 SEIR 参数不确定性实例（同步到 SEIR 协作器）"""
        self._seir_integration.set_param_uncertainty(seir_param_uncertainty)

    def get_engine(self):
        """获取评分引擎"""
        return self._engine

    def get_localizer(self):
        """获取本土化适配器"""
        return self._localizer

    @staticmethod
    def create_default_localizer():
        """创建默认本土化适配器（问题十-3：统一 GUI/CLI 创建方式）。

        统一使用 ``KaramayLocalizer.get_instance()`` 单例模式：
        - GUI 长驻运行时复用缓存（避免每次评估重算贝叶斯校准）
        - CLI 单次执行内复用同一实例（批量评估场景性能更好）
        - 跨进程不共享（每次进程启动重建），无状态泄漏风险

        返回：
            KaramayLocalizer | None — 模块不可用时返回 None
        """
        try:
            from ..karamay.localizer import KaramayLocalizer
            return KaramayLocalizer.get_instance()
        except ImportError:
            LOGGER.info("克拉玛依本土化模块不可用，使用默认参数")
            return None

"""多层不确定性融合与决策引擎

将 SEIR/ML/GNN 各层不确定性统一融合，生成最终风险、不确定性等级与行动建议。
同时提供不确定性可视化数据准备。

文献：综合文献[1]-[7]
"""

import numpy as np


class UncertaintyFusionEngine:
    """多层不确定性融合与决策引擎

    将SEIR/ML/GNN各层不确定性统一融合：
      1. SEIR后验P(R0>1) → 全局预警
      2. ML预测区间 → 个体置信度
      3. GNN认知不确定性 → 模型熟悉度
      4. 加权融合 → 最终风险+不确定性等级+行动建议

    文献：综合文献[1]-[7]
    """

    UNCERTAINTY_LEVELS = {
        'low': {'label': '低', 'color': 'green', 'action': '常规随访'},
        'medium': {'label': '中', 'color': 'orange', 'action': '建议补充IGRA检测'},
        'high': {'label': '高', 'color': 'red', 'action': '建议分子检测+加强随访'},
    }

    def __init__(self, seir_uncertainty=None, ml_ensemble=None,
                 conformal=None, mc_dropout=None, swag=None):
        self.seir_uncertainty = seir_uncertainty
        self.ml_ensemble = ml_ensemble
        self.conformal = conformal
        self.mc_dropout = mc_dropout
        self.swag = swag
        self.fusion_history = []

    def compute_global_warning(self):
        """计算全局传播风险预警"""
        if self.seir_uncertainty is None:
            return {'warning': False, 'level': 'unknown'}
        return self.seir_uncertainty.get_high_confidence_warning()

    def fuse_individual(self, ml_result=None, gnn_result=None,
                         conformal_result=None, prior_risk=0.3,
                         weights=None, evidence_factor=1.0):
        """融合个体级别的不确定性估计

        参数：
            ml_result: ML集成结果
            gnn_result: GNN预测结果
            conformal_result: 共形预测结果
            prior_risk: 先验风险
            weights: 融合权重 {'ml': w1, 'gnn': w2}
            evidence_factor: 实验室证据自适应因子（≥1 时放大共形预测区间）。
                由 uncertainty_adaptive_factor(record) 计算。
                当 conformal_result 未提供时，仍通过标准误差放大 CI 宽度。

        返回：
            dict: {
                'risk_point': 点估计,
                'risk_ci_95': 95%CI,
                'uncertainty_level': str,
                'uncertainty_score': float,
                'action_recommendation': str,
                'confidence_flag': bool,
                'decision_support': str,
            }
        """
        if weights is None:
            weights = {'ml': 0.4, 'gnn': 0.4, 'prior': 0.2}

        ml_risk = prior_risk
        ml_std = 0.10

        if ml_result and 'error' not in ml_result:
            ml_mean = ml_result.get('mean', prior_risk)
            ml_std = ml_result.get('std', 0.10)
            if isinstance(ml_mean, (list, np.ndarray)):
                ml_risk = float(np.array(ml_mean).mean())
            else:
                ml_risk = float(ml_mean)

        gnn_risk = prior_risk
        gnn_std = 0.10

        if gnn_result and 'error' not in gnn_result:
            gnn_mean = gnn_result.get('mean', prior_risk)
            gnn_std = gnn_result.get('std', 0.10)
            if isinstance(gnn_mean, (list, np.ndarray)):
                gnn_risk = float(np.array(gnn_mean).mean())
            else:
                gnn_risk = float(gnn_mean)

        w_ml = weights['ml']
        w_gnn = weights['gnn']
        w_prior = weights.get('prior', 0.2)
        w_total = w_ml + w_gnn + w_prior
        if w_total > 0:
            w_ml /= w_total
            w_gnn /= w_total
            w_prior /= w_total

        risk_point = (
            w_ml * ml_risk +
            w_gnn * gnn_risk +
            w_prior * prior_risk
        )

        combined_std = float(np.sqrt(
            (w_ml * ml_std) ** 2 +
            (w_gnn * gnn_std) ** 2 +
            (w_prior * 0.10) ** 2
        ))

        # 低证据放大：无共形结果时，按证据因子扩大标准误差驱动的 CI 宽度
        # （有共形结果时，conformal_result 已在 predict_interval 中按
        #  adaptive_factor=evidence_factor 放大，此处不再重复放大）
        evidence_factor = max(1.0, float(evidence_factor))
        combined_std *= evidence_factor

        ci_95 = [
            max(0, risk_point - 1.96 * combined_std),
            min(1, risk_point + 1.96 * combined_std),
        ]

        ci_width = ci_95[1] - ci_95[0]
        if ci_width > 0.35:
            uncertainty_level = 'high'
        elif ci_width > 0.18:
            uncertainty_level = 'medium'
        else:
            uncertainty_level = 'low'

        if conformal_result and 'error' not in conformal_result:
            cp_lower = conformal_result.get('lower', ci_95[0])
            cp_upper = conformal_result.get('upper', ci_95[1])
            if isinstance(cp_lower, (list, np.ndarray)):
                cp_lower = float(np.array(cp_lower).flatten()[0])
                cp_upper = float(np.array(cp_upper).flatten()[0])
            ci_95 = [float(cp_lower), float(cp_upper)]
            ci_width = ci_95[1] - ci_95[0]
            if ci_width > 0.35:
                uncertainty_level = 'high'
            elif ci_width > 0.18:
                uncertainty_level = 'medium'
            else:
                uncertainty_level = 'low'

        action = self.UNCERTAINTY_LEVELS[uncertainty_level]['action']

        needs_enhanced = uncertainty_level != 'low' or risk_point > 0.25
        confidence_flag = not (uncertainty_level == 'high' and risk_point > 0.3)

        if risk_point > 0.4:
            action = '立即分子检测+预防治疗评估'
        elif risk_point > 0.25:
            if uncertainty_level == 'high':
                action = '分子检测+补充IGRA'
            else:
                action = '补充IGRA或TST检测'

        decision_support = (
            f"风险={risk_point:.1%}[{ci_95[0]:.1%}-{ci_95[1]:.1%}], "
            f"置信度={'高' if confidence_flag else '低'}, "
            f"建议: {action}"
        )

        result = {
            'risk_point': round(float(risk_point), 4),
            'risk_ci_95': [round(float(ci_95[0]), 4),
                           round(float(ci_95[1]), 4)],
            'ci_width': round(float(ci_width), 4),
            'uncertainty_level': uncertainty_level,
            'uncertainty_score': round(float(min(1.0, ci_width * 3.0)), 4),
            'action_recommendation': action,
            'confidence_flag': confidence_flag,
            'needs_enhanced_screening': needs_enhanced,
            'decision_support': decision_support,
            'contributions': {
                'ml_risk': round(float(ml_risk), 4),
                'gnn_risk': round(float(gnn_risk), 4),
                'ml_uncertainty': round(float(ml_std), 4),
                'gnn_uncertainty': round(float(gnn_std), 4),
            },
        }

        self.fusion_history.append(result)
        return result

    def fuse_batch(self, contacts_data, feature_extractor=None):
        """批量融合接触者不确定性

        参数：
            contacts_data: list[dict] 接触者数据列表
            feature_extractor: 特征提取函数

        返回：
            list[dict]: 各接触者的融合结果
        """
        results = []
        for contact in contacts_data:
            ml_result = None
            gnn_result = None

            if self.ml_ensemble and self.ml_ensemble.is_trained:
                if feature_extractor:
                    features = feature_extractor(contact)
                    ml_result = self.ml_ensemble.predict_with_uncertainty(
                        [features])

            # 实验室证据自适应因子：无实验室证据（grade=0）时放大置信区间
            evidence_factor = 1.0
            try:
                from tb_risk.core.lab_evidence import uncertainty_adaptive_factor
                evidence_factor = uncertainty_adaptive_factor(contact)
            except ImportError:
                evidence_factor = 1.0

            result = self.fuse_individual(
                ml_result=ml_result,
                gnn_result=gnn_result,
                prior_risk=contact.get('prior_risk', 0.3),
                evidence_factor=evidence_factor)

            result['contact_info'] = {
                'name': contact.get('name', ''),
                'type': contact.get('type', 'social'),
            }
            results.append(result)

        return results

    def generate_report(self):
        """生成不确定性量化综合报告"""
        global_warning = self.compute_global_warning()

        report = {
            'global_warning': global_warning,
            'fusion_count': len(self.fusion_history),
            'average_uncertainty': (
                float(np.mean([r['uncertainty_score']
                               for r in self.fusion_history]))
                if self.fusion_history else 0.0),
            'high_uncertainty_count': sum(
                1 for r in self.fusion_history
                if r['uncertainty_level'] == 'high'),
            'needs_enhanced_count': sum(
                1 for r in self.fusion_history
                if r['needs_enhanced_screening']),
        }

        return report



class UncertaintyVisualizer:
    """不确定性可视化数据准备

    为GUI风险列表生成不确定性展示数据，包含：
      - 风险概率（点估计）
      - 95% 预测区间
      - 不确定性等级及对应行动建议
    """

    HEADERS_CN = ['姓名', '类型', '风险概率', '95%预测区间',
                  '不确定性等级', '行动建议', '置信度']

    def __init__(self, fusion_engine=None):
        self.fusion_engine = fusion_engine

    def prepare_table_data(self, fusion_results, contact_info_map=None):
        """准备风险列表展示数据

        返回：
            list[dict]: [{'姓名', '类型', '风险概率', '95%预测区间',
                         '不确定性等级', '行动建议', '置信度'}, ...]
        """
        rows = []
        for i, result in enumerate(fusion_results):
            cinfo = result.get('contact_info', {})
            name = cinfo.get('name', f'接触者{i+1}')
            ctype = cinfo.get('type', '社会接触')

            risk = result.get('risk_point', 0)
            ci = result.get('risk_ci_95', [0, 0])
            ci_str = f"{ci[0]:.1%}–{ci[1]:.1%}"
            level = result.get('uncertainty_level', '')

            level_info = UncertaintyFusionEngine.UNCERTAINTY_LEVELS.get(
                level, {'label': '未知', 'action': ''})
            level_label = level_info.get('label', '未知')
            action = result.get('action_recommendation', '')

            confidence = '高' if result.get('confidence_flag', False) else '低'

            rows.append({
                '姓名': name,
                '类型': ctype,
                '风险概率': f"{risk:.1%}",
                '95%预测区间': ci_str,
                '不确定性等级': level_label,
                '行动建议': action,
                '置信度': confidence,
                'raw_risk': risk,
                'raw_ci': ci,
                'raw_level': level,
            })

        rows.sort(key=lambda r: r.get('raw_risk', 0), reverse=True)
        return rows

    def prepare_seir_plot_data(self, posterior_predictive):
        """准备SEIR曲线图的置信带数据"""
        if not posterior_predictive or 'error' in posterior_predictive:
            return None

        ci = posterior_predictive.get('ci_bands', {})
        return {
            'time': ci.get('time', []),
            'I_median': posterior_predictive.get('peak_I_mean', 0),
            'I_lower': ci.get('lower', []),
            'I_upper': ci.get('upper', []),
            'r0_mean': posterior_predictive.get('r0_mean', 0),
            'r0_ci': posterior_predictive.get('r0_ci', [0, 0]),
        }

from ..scoring.simulator import ScreeningDataSimulator
from .backtest import BacktestEngine
from .sensitivity import MorrisSensitivityAnalyzer
from .ablation import AblationAnalyzer
from .comparator import NationalModelComparator

import logging
import os

LOGGER = logging.getLogger("tb_risk.validation")


class KaramayValidator:
    """克拉玛依本土化模型验证主控制器

    整合四大分析模块的顶层接口，提供一键式验证流程。

    支持两种数据源：
    1. 合成数据（ScreeningDataSimulator.generate_dataset）—— 用于快速原型验证
    2. 外部真实数据（load_external_data）—— 用于打破循环验证，对接真实筛查数据
    """

    # 支持的外部数据格式
    SUPPORTED_FORMATS = ('.csv', '.json', '.jsonl', '.xlsx', '.xls')

    def __init__(self, localizer=None, random_state=42):
        self.localizer = localizer
        self.simulator = ScreeningDataSimulator(random_state=random_state)
        self._last_results = {}
        self._external_data_loaded = False

    # ========== V1: 外部真实数据加载接口 ==========

    def load_external_data(self, file_path, record_key='records',
                           label_field='is_confirmed', id_field='record_id',
                           verbose=True):
        """加载外部真实数据（V1：打破合成数据循环验证）

        支持 CSV、JSON、JSONL、Excel 格式。自动检测字段映射，
        将外部数据标准化为与 generate_dataset() 兼容的记录格式。

        参数：
            file_path (str): 数据文件路径
            record_key (str): JSON 中记录数组的键名（默认 'records'）
            label_field (str): 标签字段名（默认 'is_confirmed'）
            id_field (str): ID 字段名（默认 'record_id'）
            verbose (bool): 是否打印加载信息

        返回：
            list[dict]: 标准化后的记录列表
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f'不支持的文件格式 "{ext}"。'
                f'支持: {", ".join(self.SUPPORTED_FORMATS)}'
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'数据文件不存在: {file_path}')

        if verbose:
            LOGGER.info('[V1] 加载外部数据: %s', file_path)

        # 问题六-validator：委托 data_io.pipeline.import_pipeline 做完整导入管线
        # （加载 → 列名映射/同义词标准化 → 缺失值填充 → 质量评分 → 审计），
        # 消除 validator 与 data_io 平行存在的字段标准化逻辑。
        # validator 仅保留验证特定的标签/ID 字段重命名与类型归一化。
        from ..data_io.pipeline import import_pipeline
        pipeline_result = import_pipeline(
            file_path,
            enable_scoring=False,  # validator 不需要质量评分
            enable_audit=False,    # validator 不需要审计记录
            auto_map=True,         # 自动列名映射（同义词标准化）
            record_key=record_key,
        )
        if not pipeline_result.success:
            errs = '; '.join(pipeline_result.errors) or '未知错误'
            raise ValueError(f'导入管线失败: {errs}')
        records = pipeline_result.records
        meta = pipeline_result.meta

        if verbose and meta.get('encoding'):
            LOGGER.info('  检测到文件编码: %s', meta.get('encoding'))

        # 验证特定：重命名字段以匹配内部标准名（label_field → is_confirmed,
        # id_field → record_id）。与文件解析解耦，所有格式共用。
        records = self._rename_label_id_fields(records, label_field, id_field)

        # 标准化：确保每条记录包含必要字段
        records = self._normalize_records(records, label_field, id_field)

        self._external_data = records
        self._external_data_loaded = True

        if verbose:
            n_confirmed = sum(1 for r in records if r.get('is_confirmed', 0))
            LOGGER.info('  加载完成: %s 条记录, %s 确诊', len(records), n_confirmed)
            LOGGER.info('  数据来源: 外部文件 (非合成数据)')

        return records

    def _rename_label_id_fields(self, records, label_field, id_field):
        """重命名标签/ID 字段到内部标准名（验证特定逻辑）。

        问题六-6d：从原 _load_csv/_load_json/_load_excel 中提取的共性后处理，
        与 io_utils 的文件解析解耦。所有格式加载后统一应用。
        """
        for rec in records:
            if label_field in rec and label_field != 'is_confirmed':
                rec['is_confirmed'] = rec.pop(label_field)
            if id_field in rec and id_field != 'record_id':
                rec['record_id'] = rec.pop(id_field)
        return records

    def _normalize_records(self, records, label_field, id_field):
        """标准化记录：确保必要字段存在，类型转换"""
        for i, rec in enumerate(records):
            # 确保 is_confirmed 存在且为 int
            if 'is_confirmed' not in rec:
                rec['is_confirmed'] = int(rec.get(label_field, 0))
            else:
                try:
                    rec['is_confirmed'] = int(rec['is_confirmed'])
                except (ValueError, TypeError):
                    rec['is_confirmed'] = 0

            # 确保 record_id 存在
            if 'record_id' not in rec:
                rec['record_id'] = rec.get(id_field, f'ext_{i}')

        return records

    def run_validation_with_external_data(self, file_path, **kwargs):
        """使用外部真实数据运行完整验证（V1 便捷接口）

        等价于 load_external_data() + run_full_validation()，
        但跳过合成数据生成步骤。

        参数：
            file_path (str): 外部数据文件路径
            **kwargs: 传递给 load_external_data 和 run_full_validation 的参数

        返回：
            dict: 完整验证报告
        """
        load_kwargs = {k: v for k, v in kwargs.items()
                       if k in ('record_key', 'label_field', 'id_field', 'verbose')}
        records = self.load_external_data(file_path, **load_kwargs)

        run_kwargs = {k: v for k, v in kwargs.items()
                      if k in ('n_morris_sample', 'n_bootstrap', 'verbose')}
        return self._run_validation_on_records(records, **run_kwargs)

    def _run_validation_on_records(self, records, n_morris_sample=500,
                                    n_bootstrap=100, verbose=True):
        """在给定的记录集上运行验证分析（内部方法）"""
        n_confirmed = sum(1 for r in records if r.get('is_confirmed', 0))
        if verbose:
            LOGGER.info('  数据集: %s 条记录, %s 确诊', len(records), n_confirmed)
            LOGGER.info('  数据来源: %s', '外部真实数据' if self._external_data_loaded else '合成数据')

        # 1. 回测
        if verbose:
            LOGGER.info('[2/4] 运行本地数据回测...')
        backtest = BacktestEngine(self.localizer, use_localization=True)
        backtest_report = backtest.compare_localization_gain(records)

        if verbose:
            r = backtest_report['localized']
            LOGGER.info('  本土化: Top100检出 %s, 风险比 %.1fx',
                        r['top100_detected'], r['risk_ratio'])
            LOGGER.info('  检出率增益(Top5%%): %.1fpp',
                        backtest_report['detection_rate_gains'].get('top_5pct', 0) * 100)

        # 2. Morris敏感性分析
        if verbose:
            LOGGER.info('[3/4] 运行Morris敏感性分析...')
        morris = MorrisSensitivityAnalyzer(self.localizer)
        morris_report = morris.run_analysis(records, n_sample_ids=n_morris_sample)

        if verbose and morris_report.get('parameters'):
            top3 = morris_report['parameters'][:3]
            for p in top3:
                LOGGER.info('  %s: μ*=%.4f, σ=%.4f', p['label'], p['mu_star'], p['sigma'])

        # 3. 消融实验
        if verbose:
            LOGGER.info('[4/4] 运行消融实验 + 国家级模型对比...')
        ablation = AblationAnalyzer(self.localizer)
        ablation_report = ablation.run_ablation_with_metrics(records)

        comparator = NationalModelComparator(self.localizer)
        comparison_report = comparator.run_comparison(records, n_bootstrap=n_bootstrap)

        if verbose:
            gains = ablation_report.get('ablation_gains', {})
            sorted_gains = sorted(
                gains.items(),
                key=lambda x: x[1].get('absolute_delta', 0),
                reverse=True
            )
            LOGGER.info('  消融实验贡献排名:')
            for i, (mod, info) in enumerate(sorted_gains[:3]):
                LOGGER.info('    %d. %s: Δ=%.2fpp (%.1f%%)',
                            i + 1, info['label'],
                            info['absolute_delta'], info['relative_change_pct'])
            ci = comparison_report.get('bootstrap_confidence_intervals', {})
            if ci:
                LOGGER.info('  Bootstrap 95%%CI (Top5%%检出率增益): [%.1f%%, %.1f%%]',
                            ci.get('ci_95_lower', 0) * 100,
                            ci.get('ci_95_upper', 0) * 100)

        # 汇总
        self._last_results = {
            'backtest': backtest_report,
            'morris_sensitivity': morris_report,
            'ablation': ablation_report,
            'national_comparison': comparison_report,
            'dataset_info': {
                'n_total': len(records),
                'n_confirmed': n_confirmed,
                'positivity_rate': n_confirmed / max(len(records), 1),
                'data_source': 'external' if self._external_data_loaded else 'synthetic',
            },
        }

        if verbose:
            LOGGER.info('=' * 60)
            LOGGER.info('全部验证分析完成')
            LOGGER.info('=' * 60)

        return self._last_results

    def run_full_validation(self, n_samples=None, n_morris_sample=500,
                              n_bootstrap=100, verbose=True):
        """运行全部四项验证分析（合成数据路径）

        参数：
            n_samples (int|None): 数据集大小，None=完整48683
            n_morris_sample (int): Morris分析样本数
            n_bootstrap (int): Bootstrap重采样次数
            verbose (bool): 是否打印进度

        返回：
            dict: 完整验证报告
        """
        if verbose:
            LOGGER.info('=' * 60)
            LOGGER.info('克拉玛依本土化模型 - 验证与敏感性分析')
            LOGGER.info('=' * 60)

        # 生成合成数据集
        if verbose:
            LOGGER.info('[1/4] 生成筛查数据集...')
        self._external_data_loaded = False
        records = self.simulator.generate_dataset(
            n_samples=n_samples, include_labels=True
        )

        return self._run_validation_on_records(
            records, n_morris_sample=n_morris_sample,
            n_bootstrap=n_bootstrap, verbose=verbose
        )

    def get_last_results(self):
        return self._last_results

    def generate_summary_text(self, results=None):
        """生成可读的文本摘要报告"""
        r = results or self._last_results
        if not r:
            return '无验证结果。请先运行 run_full_validation()。'

        lines = []
        lines.append('=' * 60)
        lines.append('克拉玛依本土化模型 (KaramayLocalizer v2.0)')
        lines.append('验证与敏感性分析报告')
        lines.append('=' * 60)
        lines.append('')

        ds = r.get('dataset_info', {})
        lines.append(f'数据集: {ds.get("n_total", 0)} 样本, '
                      f'{ds.get("n_confirmed", 0)} 确诊')
        lines.append('')

        # 回测
        lines.append('--- 一、本地数据回测 ---')
        bt = r.get('backtest', {})
        loc = bt.get('localized', {})
        if loc:
            lines.append(f'  本土化模型 Top100 检出: {loc.get("top100_detected", "N/A")} 例')
            lines.append(f'  本土化模型 风险比: {loc.get("risk_ratio", 0):.1f}x')
        dg = bt.get('detection_rate_gains', {})
        for k, v in dg.items():
            lines.append(f'  检出率增益 ({k}): {v*100:.1f}pp')
        lines.append('')

        # Morris
        lines.append('--- 二、Morris敏感性分析 ---')
        mor = r.get('morris_sensitivity', {})
        params = mor.get('parameters', [])
        lines.append(f'  参数总数: {mor.get("n_parameters", 0)}, '
                      f'轨迹数: {mor.get("n_trajectories", 0)}')
        lines.append('  重要性排名 (μ*):')
        for i, p in enumerate(params[:5]):
            lines.append(f'    {i+1}. {p.get("label", p.get("name"))}'
                          f'  μ*={p.get("mu_star", 0):.4f}  '
                          f'σ={p.get("sigma", 0):.4f}')
        lines.append('')

        # 消融
        lines.append('--- 三、消融实验 ---')
        abl = r.get('ablation', {})
        gains = abl.get('ablation_gains', {})
        sorted_abl = sorted(
            gains.items(),
            key=lambda x: abs(x[1].get('absolute_delta', 0)),
            reverse=True,
        )
        for mod, info in sorted_abl:
            lines.append(f'  {info.get("label", mod)}: '
                          f'Δ={info.get("absolute_delta", 0):.4f}pp '
                          f'({info.get("relative_change_pct", 0):.1f}%)')
        lines.append('')

        # 国家级对比
        lines.append('--- 四、国家级模型对比 ---')
        nc = r.get('national_comparison', {})
        lm = nc.get('localized_model', {})
        bm = nc.get('baseline_model', {})
        lines.append(f'  本土化 Top100 检出: {lm.get("top100_detected", "N/A")}')
        lines.append(f'  基线    Top100 检出: {bm.get("top100_detected", "N/A")}')
        lines.append(f'  本土化 风险比:     {lm.get("risk_ratio", 0):.1f}x')
        lines.append(f'  基线    风险比:     {bm.get("risk_ratio", 0):.1f}x')
        ag = nc.get('absolute_gains', {})
        lines.append(f'  Top100 额外检出: {ag.get("top100_extra_detected", 0)} 例')
        lines.append(f'  Top500 额外检出: {ag.get("top500_extra_detected", 0)} 例')

        ci = nc.get('bootstrap_confidence_intervals', {})
        if ci:
            lines.append(f'  Bootstrap 95%CI (Top5%检出增益): '
                          f'[{ci.get("ci_95_lower", 0)*100:.1f}%, '
                          f'{ci.get("ci_95_upper", 0)*100:.1f}%]')

        impact = nc.get('clinical_impact_summary', {})
        if impact:
            lines.append(f'  临床影响: {impact.get("interpretation", "")}')
        lines.append('')

        return '\n'.join(lines)
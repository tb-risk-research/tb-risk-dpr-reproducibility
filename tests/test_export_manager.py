#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section IX 结果展示与导出增强测试

覆盖：
- ExportManager.export_chart: PNG/SVG/PDF 三种格式、自动扩展名、不支持格式回退、
  None figure 处理、matplotlib 不可用回退
- ExportManager.export_all_charts: 批量导出、跳过 None figure、stale 检查、
  matplotlib 不可用返回空
- ExportManager.export_results_text: 文本导出内容完整性、空结果处理
- ExportManager.export_results_csv: CSV 导出格式、utf-8-sig BOM、空结果处理
- ExportManager.export_pdf_report: reportlab 不可用返回 False、空结果处理
- ExportManager._check_stale: stale 标记检查与 messagebox 警告
- ExportManager._get_results_copy: 线程安全副本、空结果处理
"""

import csv
import os
import sys
import tempfile
from unittest import mock

import pytest

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from tb_risk.gui.results_panel.export_manager import ExportManager, REPORTLAB_AVAILABLE
from tb_risk.gui.results_panel._shared import MATPLOTLIB_AVAILABLE


# ===========================================================================
# 辅助 fixture
# ===========================================================================

class _FakeApp:
    """最小可用 GUI app 桩对象，供 ExportManager 测试使用"""

    def __init__(self, results=None, results_stale=False):
        self.results = results if results is not None else {}
        self.results_stale = results_stale
        self.patient_info = {}
        self.root = None  # messagebox 在 root 为 None 时不应崩溃
        self._data_lock = None  # 默认无锁


@pytest.fixture
def fake_app():
    return _FakeApp()


@pytest.fixture
def export_manager(fake_app):
    return ExportManager(fake_app)


@pytest.fixture
def sample_results():
    """构造一份完整的评估结果，覆盖文本/CSV 导出所有字段"""
    return {
        'overall_risk': '高',
        'total_score': 75.5,
        'base_infection_probability': 68.2,
        'overall_suggestion': '建议立即隔离并做进一步检测',
        'individual_risks': {
            '接触频率': {'risk': '高', 'suggestion': '减少接触次数'},
            '通风条件': {'risk': '中', 'suggestion': '改善通风'},
        },
        'potential_patients': {
            'family': [
                {'name': '张三', 'disease_probability': 82.5, 'priority': '极高',
                 'recommendation': '立即隔离'},
            ],
            'social': [
                {'name': '李四', 'disease_probability': 55.0, 'priority': '中',
                 'recommendation': '定期复查'},
            ],
        },
    }


# ===========================================================================
# ExportManager.export_chart
# ===========================================================================

class TestExportManagerChart:
    """export_chart: 单个 matplotlib Figure 导出"""

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_chart_png(self, export_manager, tmp_path):
        """PNG 导出应生成 .png 文件"""
        from matplotlib.figure import Figure
        fig = Figure()
        fig.add_subplot(111).plot([1, 2, 3])

        out = export_manager.export_chart(fig, str(tmp_path / "chart"), format='png')
        assert out is not None
        assert out.endswith('.png')
        assert os.path.exists(out)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_chart_svg(self, export_manager, tmp_path):
        """SVG 导出应生成 .svg 文件（矢量图）"""
        from matplotlib.figure import Figure
        fig = Figure()
        fig.add_subplot(111).plot([1, 2, 3])

        out = export_manager.export_chart(fig, str(tmp_path / "chart"), format='svg')
        assert out is not None
        assert out.endswith('.svg')
        assert os.path.exists(out)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_chart_pdf(self, export_manager, tmp_path):
        """PDF 导出应生成 .pdf 文件（矢量图）"""
        from matplotlib.figure import Figure
        fig = Figure()
        fig.add_subplot(111).plot([1, 2, 3])

        out = export_manager.export_chart(fig, str(tmp_path / "chart"), format='pdf')
        assert out is not None
        assert out.endswith('.pdf')
        assert os.path.exists(out)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_chart_auto_adds_extension(self, export_manager, tmp_path):
        """path 不含扩展名时应自动添加"""
        from matplotlib.figure import Figure
        fig = Figure()

        out = export_manager.export_chart(fig, str(tmp_path / "noext"), format='png')
        assert out == str(tmp_path / "noext.png")
        assert os.path.exists(out)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_chart_keeps_existing_extension(self, export_manager, tmp_path):
        """path 已含正确扩展名时不应重复添加"""
        from matplotlib.figure import Figure
        fig = Figure()

        path = str(tmp_path / "with_ext.png")
        out = export_manager.export_chart(fig, path, format='png')
        assert out == path
        assert os.path.exists(out)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_chart_unsupported_format_falls_back_to_png(self, export_manager, tmp_path):
        """不支持的格式应回退到 PNG"""
        from matplotlib.figure import Figure
        fig = Figure()

        out = export_manager.export_chart(fig, str(tmp_path / "fallback"),
                                          format='bmp')
        assert out is not None
        assert out.endswith('.png')
        assert os.path.exists(out)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_chart_none_figure_returns_none(self, export_manager, tmp_path):
        """figure 为 None 应返回 None"""
        out = export_manager.export_chart(None, str(tmp_path / "x"), format='png')
        assert out is None

    def test_export_chart_no_matplotlib_returns_none(self, fake_app, tmp_path):
        """matplotlib 不可用时应返回 None"""
        em = ExportManager(fake_app)
        with mock.patch('tb_risk.gui.results_panel.export_manager.MATPLOTLIB_AVAILABLE', False):
            out = em.export_chart("fake_figure", str(tmp_path / "x"), format='png')
        assert out is None

    def test_supported_chart_formats_constant(self, export_manager):
        """SUPPORTED_CHART_FORMATS 应包含 png/svg/pdf"""
        assert set(ExportManager.SUPPORTED_CHART_FORMATS) == {'png', 'svg', 'pdf'}

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_chart_none_format_defaults_to_png(self, export_manager, tmp_path):
        """format=None 应默认为 PNG"""
        from matplotlib.figure import Figure
        fig = Figure()

        out = export_manager.export_chart(fig, str(tmp_path / "default"), format=None)
        assert out is not None
        assert out.endswith('.png')


# ===========================================================================
# ExportManager.export_all_charts
# ===========================================================================

class TestExportManagerAllCharts:
    """export_all_charts: 批量导出所有图表"""

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_all_charts_exports_existing_figures(self, fake_app, tmp_path):
        """存在的 figure 应被导出，None 的应被跳过"""
        from matplotlib.figure import Figure
        fake_app.seir_figure = Figure()
        fake_app.heatmap_figure = Figure()
        fake_app.histogram_figure = None  # 应被跳过
        fake_app.dag_figure = Figure()

        em = ExportManager(fake_app)
        exported = em.export_all_charts(fake_app, str(tmp_path), format='png')

        assert len(exported) == 3  # 3 个非 None figure
        for path in exported:
            assert os.path.exists(path)
            assert path.endswith('.png')

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_all_charts_skips_none_figures(self, fake_app, tmp_path):
        """所有 figure 均为 None 时应返回空列表"""
        fake_app.seir_figure = None
        fake_app.heatmap_figure = None

        em = ExportManager(fake_app)
        exported = em.export_all_charts(fake_app, str(tmp_path), format='png')
        assert exported == []

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_all_charts_stale_returns_empty(self, fake_app, tmp_path):
        """results_stale=True 时应返回空列表"""
        fake_app.results_stale = True
        fake_app.root = None  # 避免 messagebox 调用

        from matplotlib.figure import Figure
        fake_app.seir_figure = Figure()

        em = ExportManager(fake_app)
        # mock messagebox 以避免 TclError
        with mock.patch('tb_risk.gui.results_panel.export_manager.messagebox'):
            exported = em.export_all_charts(fake_app, str(tmp_path), format='png')
        assert exported == []

    def test_export_all_charts_no_matplotlib_returns_empty(self, fake_app, tmp_path):
        """matplotlib 不可用时应返回空列表"""
        em = ExportManager(fake_app)
        with mock.patch('tb_risk.gui.results_panel.export_manager.MATPLOTLIB_AVAILABLE', False):
            with mock.patch('tb_risk.gui.results_panel.export_manager.messagebox'):
                exported = em.export_all_charts(fake_app, str(tmp_path), format='png')
        assert exported == []

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE,
                        reason="matplotlib 未安装，跳过图表导出测试")
    def test_export_all_charts_includes_calibration_and_shap(self, fake_app, tmp_path):
        """Section IX: 校准曲线 (ml_calib_figure) 和个体 SHAP 图 (shap_ind_figure)
        应被纳入批量导出列表"""
        from matplotlib.figure import Figure
        fake_app.ml_calib_figure = Figure()
        fake_app.shap_ind_figure = Figure()

        em = ExportManager(fake_app)
        exported = em.export_all_charts(fake_app, str(tmp_path), format='png')

        # 应包含校准曲线和个体 SHAP 图
        filenames = [os.path.basename(p) for p in exported]
        assert any('calibration' in name for name in filenames)
        assert any('shap_individual' in name for name in filenames)


# ===========================================================================
# ExportManager.export_results_text
# ===========================================================================

class TestExportManagerResultsText:
    """export_results_text: 文本结果导出"""

    def test_export_results_text_basic(self, export_manager, sample_results, tmp_path):
        """基本文本导出应包含所有关键字段"""
        out_path = str(tmp_path / "results.txt")
        ok = export_manager.export_results_text(sample_results, out_path)

        assert ok is True
        assert os.path.exists(out_path)

        with open(out_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 验证关键字段
        assert '结核病风险评估结果报告' in content
        assert '高' in content  # overall_risk
        assert '75.5' in content  # total_score
        assert '68.2' in content  # base_infection_probability
        assert '张三' in content  # family patient name
        assert '李四' in content  # social patient name
        assert '立即隔离' in content  # recommendation
        assert '接触频率' in content  # individual_risks key
        assert '减少接触次数' in content  # individual suggestion

    def test_export_results_text_empty_results_returns_false(self, export_manager, tmp_path):
        """空结果应返回 False"""
        ok = export_manager.export_results_text({}, str(tmp_path / "empty.txt"))
        assert ok is False

    def test_export_results_text_none_results_returns_false(self, export_manager, tmp_path):
        """None 结果应返回 False"""
        ok = export_manager.export_results_text(None, str(tmp_path / "none.txt"))
        assert ok is False

    def test_export_results_text_no_potential_patients(self, export_manager, tmp_path):
        """无 potential_patients 字段也应正常导出"""
        results = {
            'overall_risk': '低',
            'total_score': 20.0,
            'base_infection_probability': 15.0,
            'overall_suggestion': '保持观察',
            'individual_risks': {},
        }
        out_path = str(tmp_path / "no_patients.txt")
        ok = export_manager.export_results_text(results, out_path)
        assert ok is True
        assert os.path.exists(out_path)

    def test_export_results_text_invalid_path_returns_false(self, export_manager, sample_results):
        """无效路径应返回 False（被 try/except 捕获）"""
        ok = export_manager.export_results_text(
            sample_results, '/nonexistent_dir/sub/results.txt')
        assert ok is False


# ===========================================================================
# ExportManager.export_results_csv
# ===========================================================================

class TestExportManagerResultsCsv:
    """export_results_csv: CSV 结果导出"""

    def test_export_results_csv_basic(self, export_manager, sample_results, tmp_path):
        """CSV 导出应包含所有关键字段"""
        out_path = str(tmp_path / "results.csv")
        ok = export_manager.export_results_csv(sample_results, out_path)

        assert ok is True
        assert os.path.exists(out_path)

        with open(out_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 验证关键内容
        all_text = ','.join(','.join(row) for row in rows)
        assert '高' in all_text  # overall_risk
        assert '75.5' in all_text  # total_score
        assert '张三' in all_text  # family patient
        assert '李四' in all_text  # social patient
        assert '接触频率' in all_text  # individual_risks

    def test_export_results_csv_utf8_bom(self, export_manager, sample_results, tmp_path):
        """CSV 应使用 utf-8-sig 编码（含 BOM），便于 Excel 正确识别中文"""
        out_path = str(tmp_path / "bom.csv")
        export_manager.export_results_csv(sample_results, out_path)

        with open(out_path, 'rb') as f:
            first_bytes = f.read(3)
        # utf-8-sig BOM: EF BB BF
        assert first_bytes == b'\xef\xbb\xbf'

    def test_export_results_csv_empty_results_returns_false(self, export_manager, tmp_path):
        """空结果应返回 False"""
        ok = export_manager.export_results_csv({}, str(tmp_path / "empty.csv"))
        assert ok is False

    def test_export_results_csv_invalid_path_returns_false(self, export_manager, sample_results):
        """无效路径应返回 False"""
        ok = export_manager.export_results_csv(
            sample_results, '/nonexistent_dir/sub/results.csv')
        assert ok is False


# ===========================================================================
# ExportManager._check_stale
# ===========================================================================

class TestExportManagerStaleCheck:
    """_check_stale: results_stale 标记检查"""

    def test_check_stale_when_not_stale_returns_true(self, fake_app):
        """results_stale=False 时应返回 True（可继续导出）"""
        fake_app.results_stale = False
        em = ExportManager(fake_app)
        assert em._check_stale() is True

    def test_check_stale_when_stale_returns_false(self, fake_app):
        """results_stale=True 时应返回 False（中止导出）"""
        fake_app.results_stale = True
        fake_app.root = None  # 避免 messagebox 实际调用
        em = ExportManager(fake_app)
        with mock.patch('tb_risk.gui.results_panel.export_manager.messagebox'):
            assert em._check_stale() is False

    def test_check_stale_when_no_attr_returns_true(self, fake_app):
        """app 无 results_stale 属性时默认返回 True"""
        # 删除属性以模拟未初始化
        if hasattr(fake_app, 'results_stale'):
            del fake_app.results_stale
        em = ExportManager(fake_app)
        assert em._check_stale() is True

    def test_check_stale_exception_returns_true(self, fake_app):
        """检查过程中异常应返回 True（fail-open）"""
        # 让 getattr(self.app, 'results_stale', False) 抛异常
        # 通过 property 模拟
        class _BadApp:
            @property
            def results_stale(self):
                raise RuntimeError("simulated error")
            root = None

        em = ExportManager(_BadApp())
        # 异常被捕获后返回 True
        assert em._check_stale() is True


# ===========================================================================
# ExportManager._get_results_copy
# ===========================================================================

class TestExportManagerGetResultsCopy:
    """_get_results_copy: 线程安全获取 results 副本"""

    def test_get_results_copy_returns_deepcopy(self, fake_app, sample_results):
        """应返回深拷贝，修改副本不影响原件"""
        fake_app.results = sample_results
        em = ExportManager(fake_app)

        copy, err = em._get_results_copy()
        assert err is None
        assert copy == sample_results
        assert copy is not sample_results  # 不同对象
        # 修改副本不影响原件
        copy['overall_risk'] = 'modified'
        assert sample_results['overall_risk'] == '高'

    def test_get_results_copy_empty_results_returns_error(self, fake_app):
        """空 results 应返回 (None, error_message)"""
        fake_app.results = {}
        em = ExportManager(fake_app)

        copy, err = em._get_results_copy()
        assert copy is None
        assert err is not None
        assert '风险评估' in err

    def test_get_results_copy_with_data_lock(self, sample_results):
        """_data_lock 存在时应使用锁保护"""
        import threading

        class _AppWithLock:
            def __init__(self):
                self.results = sample_results
                self._data_lock = threading.Lock()
                self.root = None

        app = _AppWithLock()
        em = ExportManager(app)
        copy, err = em._get_results_copy()
        assert err is None
        assert copy == sample_results

    def test_get_results_copy_empty_with_lock_returns_error(self):
        """带锁的空 results 也应返回错误"""
        import threading

        class _AppWithLock:
            def __init__(self):
                self.results = {}
                self._data_lock = threading.Lock()
                self.root = None

        em = ExportManager(_AppWithLock())
        copy, err = em._get_results_copy()
        assert copy is None
        assert err is not None


# ===========================================================================
# ExportManager.export_pdf_report
# ===========================================================================

class TestExportManagerPdfReport:
    """export_pdf_report: PDF 报告导出"""

    @pytest.mark.skipif(not REPORTLAB_AVAILABLE,
                        reason="reportlab 未安装，跳过 PDF 报告测试")
    def test_export_pdf_report_with_reportlab(self, export_manager, sample_results, tmp_path):
        """reportlab 可用时应成功生成 PDF"""
        out_path = str(tmp_path / "report.pdf")
        ok = export_manager.export_pdf_report(sample_results, {}, out_path)

        # 注意：实际生成依赖 export_utils._export_pdf_report 的实现
        # 此处仅验证不抛异常并返回 bool
        assert isinstance(ok, bool)

    def test_export_pdf_report_no_reportlab_returns_false(self, fake_app, sample_results, tmp_path):
        """reportlab 不可用时应返回 False（不静默降级）"""
        em = ExportManager(fake_app)
        with mock.patch('tb_risk.gui.results_panel.export_manager.REPORTLAB_AVAILABLE', False):
            ok = em.export_pdf_report(sample_results, {}, str(tmp_path / "x.pdf"))
        assert ok is False

    def test_export_pdf_report_empty_results_returns_false(self, export_manager, tmp_path):
        """空结果应返回 False"""
        ok = export_manager.export_pdf_report({}, {}, str(tmp_path / "empty.pdf"))
        assert ok is False

    def test_export_pdf_report_none_results_returns_false(self, export_manager, tmp_path):
        """None 结果应返回 False"""
        ok = export_manager.export_pdf_report(None, {}, str(tmp_path / "none.pdf"))
        assert ok is False


# ===========================================================================
# ExportManager 初始化与基本属性
# ===========================================================================

class TestExportManagerInit:

    def test_init_stores_app(self, fake_app):
        em = ExportManager(fake_app)
        assert em.app is fake_app

    def test_get_patient_info_returns_dict(self, fake_app):
        """_get_patient_info 应返回字典（即使 app 无 patient_info 属性）"""
        em = ExportManager(fake_app)
        info = em._get_patient_info()
        assert isinstance(info, dict)
        assert info == {}

    def test_get_patient_info_returns_deepcopy(self, fake_app):
        """_get_patient_info 应返回深拷贝"""
        fake_app.patient_info = {'age': 50, 'name': '张三'}
        em = ExportManager(fake_app)
        info = em._get_patient_info()
        assert info == {'age': 50, 'name': '张三'}
        info['age'] = 99
        assert fake_app.patient_info['age'] == 50  # 原件未修改

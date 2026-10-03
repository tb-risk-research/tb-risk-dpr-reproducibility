#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 混合类 - 图形界面初始化、选项卡、图表、ML/GNN训练界面"""

from ._imports import *  # noqa: F401,F403 — 共享导入和常量（含项目内部依赖）

# Section X: 独立帮助窗口（仅依赖 tkinter，与 mixins 同生命周期加载）
from .help_system import HelpWindow


class BaseMixin:
    """GUI base methods"""

    def _cleanup_training_threads(self, timeout=5.0):
        """优雅退出：等待所有后台训练线程完成，避免资源泄漏

        每个线程最多等待 timeout 秒，超时则跳过。防止 daemon 线程被强制终止
        导致不完整的文件写入、数据库事务或 CUDA 上下文泄漏。
        """
        for t in getattr(self, '_training_threads', []):
            if t.is_alive():
                try:
                    t.join(timeout=timeout)
                except RuntimeError as e:
                    LOGGER.debug("等待训练线程结束失败: %s", e)
        # 清理已完成的线程引用
        if hasattr(self, '_training_threads'):
            self._training_threads = [t for t in self._training_threads if t.is_alive()]


    def _start_training_thread(self, target, daemon=True):
        """启动后台训练线程并跟踪，便于优雅退出时 join

        参数：
            target: 线程目标函数
            daemon: 是否为守护线程（默认 True）
        返回：
            threading.Thread 实例
        """
        t = threading.Thread(target=target, daemon=daemon)
        if not hasattr(self, '_training_threads'):
            self._training_threads = []
        self._training_threads.append(t)
        t.start()
        return t


    def init_gui(self):
        """初始化 GUI 界面"""
        self.ui_mode = self.UI_MODE_CLASSIC  # 设置为传统界面模式
        # 优先级六：创建 ClassicAdapter 实例
        from .ui_mode_adapter import create_adapter
        self.adapter = create_adapter(self.ui_mode, self)

        # 高 DPI 适配（Windows 下避免界面模糊）
        self._enable_high_dpi()

        self.root = tk.Tk()
        self.root.title("结核病传播风险评估系统")

        # 窗口图标：优先尝试 assets/icon.ico，失败时用内置 emoji 色块
        self._apply_window_icon()

        # 从配置恢复窗口几何信息（Section IV 配置持久化）
        geometry = None
        if hasattr(self, 'config') and self.config:
            geometry = self.config.window_geometry
        if geometry and 'x' in geometry and ('+' in geometry):
            self.root.geometry(geometry)
        else:
            # 默认尺寸 + 启动居中
            w, h = 1280, 800
            self.root.geometry(self._center_geometry(w, h))
        self.root.minsize(1100, 720)
        self.root.resizable(True, True)

        self._start_progress_polling()

        self._setup_styles()
        self._setup_keyboard_shortcuts()

        self.basic_info_vars = {}

        # 顶部品牌头部栏（标题 + 副标题 + 徽标）
        self._create_header(self.root)

        self._create_menu()

        # 底部状态栏必须先于主内容区 pack（side='bottom' 先占位）：
        # 若放在 fill+expand 的主内容区之后，内容区会按请求高度吃满剩余空间，
        # 底部栏被挤出窗口外（步骤指示器/快捷键提示不可见，内容直怼底边）
        self.bottom_bar = ttk.Frame(self.root, style='Bottom.TFrame')
        self.bottom_bar.pack(side='bottom', fill='x', padx=0, pady=0)

        # 改进4：左侧快速导航栏 + 主内容区域
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=0, pady=0)

        self._create_sidebar(main_container)

        content_frame = ttk.Frame(main_container)
        content_frame.pack(side='left', fill='both', expand=True)

        self.notebook = ttk.Notebook(content_frame, style='Main.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=20, pady=(12, 8))

        tab1 = ttk.Frame(self.notebook)
        tab2 = ttk.Frame(self.notebook)
        tab3 = ttk.Frame(self.notebook)
        tab4 = ttk.Frame(self.notebook)
        tab5 = ttk.Frame(self.notebook)

        self.notebook.add(tab1, text='患者基本信息')
        self.notebook.add(tab2, text='家庭成员信息')
        self.notebook.add(tab3, text='社会接触者信息')
        self.notebook.add(tab4, text='风险评估结果')
        self.notebook.add(tab5, text='部署与监控')

        self._init_basic_info_tab(tab1)
        self._init_family_tab(tab2)
        self._init_social_tab(tab3)
        self._init_result_tab(tab4)

        # 部署与监控面板（批量评估 / 模型对比 / 漂移监控）
        # 由 DeployPanelMixin 提供，若 Mixin 未接入则安全跳过
        if hasattr(self, '_init_deploy_tab'):
            try:
                self._init_deploy_tab(tab5)
            except Exception as e:
                LOGGER.debug("部署与监控标签页初始化失败（非致命）: %s", e)

        self.family_entries = []
        self.social_entries = []

        # 改进4：步骤进度指示器
        step_frame = ttk.Frame(self.bottom_bar, style='Bottom.TFrame')
        step_frame.pack(side='left', padx=12, pady=5)
        self._step_labels = []
        for i, step_name in enumerate(['1.基本信息', '2.家庭成员', '3.社会接触者', '4.评估结果']):
            step_lbl = ttk.Label(step_frame, text=f"● {step_name}",
                                font=self._FONT_SMALL,
                                style='Bottom.TLabel')
            step_lbl.pack(side='left', padx=(0, 8))
            self._step_labels.append(step_lbl)

        # P1 修复溢出：精简快捷键提示，弹性占位分离左右组件
        self.footer_label = ttk.Label(self.bottom_bar,
            text="F5 评估 | Ctrl+S 保存 | Ctrl+E 导出 | F1 帮助",
            font=self._FONT_SMALL, style='Bottom.TLabel')
        self.footer_label.pack(side='left', padx=10)

        # 改进7：字段帮助提示文本（焦点切换时动态更新）
        self.help_status_label = ttk.Label(self.bottom_bar,
            text="就绪 - F5开始评估，F1查看帮助",
            font=self._FONT_SMALL, style='Bottom.TLabel')
        self.help_status_label.pack(side='left', padx=8)

        # 弹性占位 Frame 避免窄窗口下左右组件碰撞
        spacer = ttk.Frame(self.bottom_bar, style='Bottom.TFrame')
        spacer.pack(side='left', fill='x', expand=True)

        # 改进7：下一步按钮
        ttk.Button(self.bottom_bar, text="下一步 ▶",
                   command=self._goto_next_tab).pack(side='right', padx=4, pady=2)

        fill_panel = ttk.Frame(self.bottom_bar, style='Bottom.TFrame')
        fill_panel.pack(side='right', padx=10)
        self.fill_progress_var = tk.IntVar(value=0)
        self.fill_progress_bar = ttk.Progressbar(fill_panel, variable=self.fill_progress_var,
                                                  length=100, mode='determinate')
        self.fill_progress_bar.pack(side='left', padx=2)
        self.fill_count_label = ttk.Label(fill_panel, text="0/0",
                                          font=self._FONT_SMALL, style='Bottom.TLabel')
        self.fill_count_label.pack(side='left', padx=2)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # 启动自动保存定时器（Section IV）
        if hasattr(self, 'auto_save'):
            self.auto_save.start()

        # 优先级一：启动时检查崩溃恢复数据
        if hasattr(self, '_check_autosave_recovery'):
            self._check_autosave_recovery()

        # 绑定标签页切换事件以更新步骤指示器
        self.notebook.bind('<<NotebookTabChanged>>', self._on_notebook_tab_changed)

    # ── 窗口初始化辅助方法 ──────────────────────────────────────

    def _enable_high_dpi(self):
        """Windows 高 DPI 适配（Per-Monitor V2），避免界面模糊。非 Windows 静默跳过。"""
        import sys as _sys
        if _sys.platform != 'win32':
            return
        try:
            import ctypes
            # 优先使用 Per-Monitor V2（Win10 1703+）
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                return
            except Exception:
                pass
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        except Exception as e:
            LOGGER.debug("高 DPI 适配失败（非致命）: %s", e)

    def _center_geometry(self, w, h):
        """根据屏幕尺寸计算居中 geometry 字符串。"""
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
        except Exception:
            return f"{w}x{h}"
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        return f"{w}x{h}+{x}+{y}"

    def _apply_window_icon(self):
        """设置窗口图标。尝试 assets/icon.ico，失败时使用 PhotoImage 生成的纯色图标。"""
        import os as _os
        ico_candidates = [
            _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'assets', 'icon.ico'),
            _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'icon.ico'),
        ]
        for path in ico_candidates:
            try:
                if _os.path.exists(path):
                    self.root.iconbitmap(path)
                    return
            except Exception:
                continue
        # 回退：无 ico 文件时不报错（tk 默认图标）
        LOGGER.debug("未找到 icon.ico，使用默认窗口图标")

    def _create_header(self, parent):
        """界面优化：顶部品牌头部栏

        深青蓝底 + 医疗十字徽标 + 系统标题/副标题 + 右侧功能徽标，
        为整个应用建立明确的品牌身份与视觉层级。
        """
        header = ttk.Frame(parent, style='Header.TFrame', padding=(16, 10))
        header.pack(side='top', fill='x')
        header.columnconfigure(1, weight=1)

        # 医疗十字徽标（tk.Label 色块，主题切换时同步更新背景）
        self._header_logo = tk.Label(
            header, text="＋", font=('Microsoft YaHei', 16, 'bold'),
            bg=self.COLORS['accent'], fg=self.COLORS['accent_fg'],
            relief='flat', padx=5, pady=1)
        self._header_logo.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky='ns')

        ttk.Label(header, text="结核病传播风险评估系统",
                  style='HeaderBrand.TLabel').grid(row=0, column=1, sticky='w')
        ttk.Label(header,
                  text="家庭与社会接触网络 · SEIR 传播动力学 · 智能风险分层",
                  style='HeaderSub.TLabel').grid(row=1, column=1, sticky='w',
                                                 pady=(1, 0))

        ttk.Label(header, text="评估工作台",
                  style='HeaderBadge.TLabel').grid(row=0, column=2, rowspan=2,
                                                   sticky='e', padx=(8, 0))

    def _create_sidebar(self, parent):
        """界面优化：创建左侧功能导航栏（含激活态高亮）"""
        sidebar = ttk.Frame(parent, width=170, style='Sidebar.TFrame')
        sidebar.pack(side='left', fill='y', padx=(0, 2))
        sidebar.pack_propagate(False)

        # 分区标题
        ttk.Label(sidebar, text="功能导航",
                  style='SidebarSection.TLabel').pack(padx=10, pady=(12, 6), anchor='w')

        # 导航按钮（左侧对齐，选中态以浅青绿底高亮）
        # 分组：评估流程（4 步工作流，编号与底部步骤指示器一致）
        # 与部署运维（部署期功能）视觉分区
        nav_items = [
            '1  患者信息',
            '2  家庭成员',
            '3  社会接触者',
            '4  评估结果',
            '⚙  部署监控',
        ]
        self._nav_buttons = []
        for idx, text in enumerate(nav_items):
            if idx == 4:
                # 部署运维分区
                ttk.Separator(sidebar, orient='horizontal').pack(
                    fill='x', padx=8, pady=(8, 2))
                ttk.Label(sidebar, text="部署运维",
                          style='SidebarSection.TLabel').pack(
                              padx=10, pady=(0, 2), anchor='w')
            btn = ttk.Button(sidebar, text=text, style='Nav.TButton',
                           cursor='hand2',
                           command=lambda i=idx: self._navigate_to_tab(i))
            btn.pack(fill='x', padx=6, pady=2)
            self._nav_buttons.append(btn)

        # 快速操作区
        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', padx=8, pady=8)
        self._quick_assess_btn = tk.Button(sidebar, text="▶ 快速评估",
                  font=self._FONT_DEFAULT,
                  bg=self.COLORS['accent'], fg='white',
                  activebackground=self.COLORS['accent_hover'], activeforeground='white',
                  relief='flat', bd=0, cursor='hand2',
                  padx=8, pady=6,
                  command=self._run_assessment)
        self._quick_assess_btn.pack(fill='x', padx=6, pady=2)

    def _navigate_to_tab(self, idx):
        """切换到指定标签页"""
        try:
            tabs = self.notebook.tabs()
            if 0 <= idx < len(tabs):
                self.notebook.select(tabs[idx])
        except Exception as e:
            LOGGER.debug("_navigate_to_tab(%d) 失败: %s", idx, e)

    def _goto_next_tab(self):
        """改进7：切换到下一个标签页"""
        try:
            tabs = self.notebook.tabs()
            current = self.notebook.select()
            idx = tabs.index(current) if current else -1
            next_idx = (idx + 1) % len(tabs)
            self.notebook.select(tabs[next_idx])
        except Exception as e:
            LOGGER.debug("_goto_next_tab 切换标签页失败: %s", e)

    def _set_help_status(self, text):
        """改进7：更新底部栏帮助提示文本，空文本时显示默认就绪提示"""
        if hasattr(self, 'help_status_label'):
            try:
                if text:
                    self.help_status_label.configure(text=text)
                else:
                    self.help_status_label.configure(text="就绪 - F5开始评估，F1查看帮助")
            except Exception as e:
                LOGGER.debug("_set_help_status 更新状态栏失败: %s", e)

    def _on_notebook_tab_changed(self, event=None):
        """标签页切换时更新步骤指示器与侧边栏激活态"""
        self._update_step_indicator()
        self._update_nav_active()
        if hasattr(self, '_update_assess_button_state'):
            try:
                self._update_assess_button_state()
            except Exception as e:
                LOGGER.debug("_on_notebook_tab_changed 更新按钮状态失败: %s", e)

    def _update_nav_active(self):
        """界面优化：高亮当前标签页对应的侧边栏导航按钮"""
        nav_buttons = getattr(self, '_nav_buttons', None)
        if not nav_buttons:
            return
        try:
            current_idx = self.notebook.index(self.notebook.select())
        except Exception as e:
            LOGGER.debug("_update_nav_active 获取当前标签页索引失败: %s", e)
            return
        for i, btn in enumerate(nav_buttons):
            try:
                btn.configure(style='NavActive.TButton' if i == current_idx else 'Nav.TButton')
            except Exception as e:
                LOGGER.debug("_update_nav_active 更新按钮 %d 样式失败: %s", i, e)

    def _update_step_indicator(self):
        """改进4：更新底部步骤指示器样式（当前步骤高亮）"""
        if not hasattr(self, '_step_labels') or not self._step_labels:
            return
        try:
            current_idx = self.notebook.index(self.notebook.select())
        except Exception as e:
            LOGGER.debug("_update_step_indicator 获取当前标签页索引失败: %s", e)
            return
        for i, lbl in enumerate(self._step_labels):
            try:
                if current_idx >= len(self._step_labels):
                    # 部署与监控等流程外标签页：步骤指示器全部置灰
                    lbl.configure(style='StepInactive.TLabel')
                elif i < current_idx:
                    lbl.configure(style='StepDone.TLabel')
                elif i == current_idx:
                    lbl.configure(style='StepActive.TLabel')
                else:
                    lbl.configure(style='StepInactive.TLabel')
            except Exception as e:
                LOGGER.debug("_update_step_indicator 更新步骤 %d 样式失败: %s", i, e)


    def _setup_styles(self):
        """配置全局视觉风格：主题、调色板、字体、控件样式"""
        self.COLORS = {
            # 医疗青绿主色系（中性色 bg_light/text/text_light 受测试约束，勿改）
            'primary': '#164e63',        # 深青蓝 —— 头部/底部导航底色
            'primary_dark': '#0f3a4a',   # primary 更深的变体
            'accent': '#0f766e',         # 医疗青绿 —— 主行动色（WCAG AA）
            'accent_hover': '#115e59',   # accent 悬停/按下加深
            'accent_soft': '#ccfbf1',    # accent 浅色底（选中/悬停背景）
            'accent_fg': '#ffffff',      # accent 底色上的文字色
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'bg_light': '#f8f9fa',
            'bg_card': '#ffffff',
            'text': '#2c3e50',
            # Section X: WCAG AA 对比度（≥4.5:1），将灰色加深
            'text_light': '#5a6c7d',
            'border': '#dee2e6',
            # 优先级十：matplotlib 图表语义化颜色（主题切换时同步更新）
            'risk_high': '#e74c3c',      # 高风险（红）
            'risk_medium': '#f39c12',    # 中风险（橙）
            'risk_low': '#2ecc71',       # 低风险（绿）
            'risk_high_bg': '#ffe0e0',   # 高风险背景
            'risk_medium_bg': '#fff3cd', # 中风险背景
            'risk_low_bg': '#e0ffe0',    # 低风险背景
            'risk_high_fg': '#c0392b',   # 高风险前景（深红，用于浅色背景）
            'risk_medium_fg': '#856404', # 中风险前景（深黄）
            'risk_low_fg': '#27ae60',    # 低风险前景（深绿）
            'chart_series_1': '#FF6B6B', # 图表系列 1
            'chart_series_2': '#4ECDC4', # 图表系列 2
            'chart_series_3': '#45B7D1', # 图表系列 3
            'chart_series_4': '#FFA07A', # 图表系列 4
            'chart_series_5': '#98D8C8', # 图表系列 5
            'chart_facecolor': '#ffffff', # 图表背景色
            'chart_edgecolor': '#2c3e50', # 图表文字/边框色
            'chart_grid_color': '#cccccc', # 图表网格线色
            # 改进6：新增语义色，替换硬编码颜色
            'warning_bg': '#fff3cd',     # 警告背景
            'warning_fg': '#856404',     # 警告文字
            'info': '#17a2b8',           # 信息色
            'info_bg': '#d1ecf1',        # 信息背景
            'danger_bg': '#f8d7da',      # 危险背景
            'danger_fg': '#721c24',      # 危险文字
            'success_bg': '#d4edda',     # 成功背景
            'success_fg': '#155724',     # 成功文字
            'disabled': '#999999',       # 禁用色
            'required': '#e74c3c',       # 必填标记色
            'step_active': '#0f766e',    # 步骤激活色（青绿）
            'step_inactive': '#aab7c4',  # 步骤未激活色
            'step_done': '#2ecc71',      # 步骤完成色
            # 头部品牌栏
            'header_fg': '#ffffff',      # 头部标题文字
            'header_sub_fg': '#cbd5e1',  # 头部副标题文字
            # 侧边栏
            'sidebar_bg': '#ffffff',     # 侧边栏背景
        }

        style = ttk.Style()
        available_themes = style.theme_names()
        # Windows上优先使用vista原生主题（复选框/单选框显示标准✓/●，不会出现×）
        # 其他平台优先clam跨平台主题
        import sys
        if sys.platform == 'win32':
            preferred_themes = ('vista', 'winnative', 'clam', 'alt', 'default')
        elif sys.platform == 'darwin':
            preferred_themes = ('aqua', 'clam', 'alt', 'default')
        else:
            preferred_themes = ('clam', 'alt', 'default')
        for preferred in preferred_themes:
            if preferred in available_themes:
                style.theme_use(preferred)
                break

        # 原生 Windows 主题（vista/winnative/xpnative）的按钮面由系统绘制，
        # 会忽略 background 配置 —— 彩色按钮（Accent/Nav）的文字落在浅灰
        # 原生按钮面上不可读（实测：开始评估按钮呈"空白框"）。
        # 从 clam 借按钮元素替换布局，使背景/悬停色真正生效。
        if style.theme_use() in ('vista', 'winnative', 'xpnative'):
            try:
                style.element_create('ColorButton', 'from', 'clam')
                for _color_btn in ('Accent.TButton', 'Nav.TButton',
                                   'NavActive.TButton'):
                    style.layout(_color_btn, [
                        ('ColorButton', {'children': [
                            ('Button.focus', {'children': [
                                ('Button.padding', {'children': [
                                    ('Button.label', {'sticky': 'nswe'})],
                                 'sticky': 'nswe'})],
                             'sticky': 'nswe'})],
                         'border': 1, 'sticky': 'nswe'})])
            except Exception as e:
                LOGGER.debug("彩色按钮 clam 元素注入失败（回退原生样式）: %s", e)

        default_font = ('Microsoft YaHei', 10)
        heading_font = ('Microsoft YaHei', 11, 'bold')
        small_font = ('Microsoft YaHei', 9)

        # 改进8：记录系统 DPI 缩放（仅记录，不再手动放大字体）。
        # 开启高 DPI 感知后，Tk 会依据 tk scaling 自动放大控件与字体，
        # 若此处再手动乘 scale_factor 会造成"双重放大"，导致字体过大。
        try:
            dpi_scale = self.root.tk.call('tk', 'scaling')
            self._dpi_scale = float(dpi_scale) if dpi_scale else 1.0
        except Exception:
            self._dpi_scale = 1.0

        style.configure('TFrame', background=self.COLORS['bg_light'])
        style.configure('TLabel', font=default_font, background=self.COLORS['bg_light'],
                       foreground=self.COLORS['text'])
        style.configure('TButton', font=default_font, padding=(12, 7),
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text'],
                        borderwidth=1, relief='flat')
        style.map('TButton',
                  background=[('active', self.COLORS['accent_soft']),
                              ('pressed', self.COLORS['accent_soft']),
                              ('!disabled', self.COLORS['bg_card'])],
                  foreground=[('active', self.COLORS['accent']),
                              ('!disabled', self.COLORS['text'])])
        style.configure('Accent.TButton', font=default_font, padding=(14, 8),
                       background=self.COLORS['accent'], foreground=self.COLORS['accent_fg'],
                       borderwidth=0, relief='flat')
        style.map('Accent.TButton',
                  background=[('active', self.COLORS['accent_hover']),
                              ('pressed', self.COLORS['primary_dark']),
                              ('!disabled', self.COLORS['accent'])],
                  foreground=[('active', self.COLORS['accent_fg']),
                              ('pressed', self.COLORS['accent_fg']),
                              ('!disabled', self.COLORS['accent_fg'])])
        style.configure('Danger.TButton', font=default_font, padding=(12, 7))
        style.configure('Success.TButton', font=default_font, padding=(12, 7))
        style.configure('TCombobox', font=default_font, padding=6,
                        fieldbackground=self.COLORS['bg_card'],
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text'],
                        bordercolor=self.COLORS['border'],
                        lightcolor=self.COLORS['border'],
                        darkcolor=self.COLORS['border'],
                        selectbackground=self.COLORS['accent'],
                        selectforeground='white')
        style.map('TCombobox',
                  fieldbackground=[('readonly', self.COLORS['bg_card']),
                                   ('focus', self.COLORS['bg_card'])],
                  selectbackground=[('focus', self.COLORS['accent'])],
                  selectforeground=[('focus', 'white')],
                  bordercolor=[('focus', self.COLORS['accent'])],
                  lightcolor=[('focus', self.COLORS['accent'])],
                  darkcolor=[('focus', self.COLORS['accent'])])
        style.configure('TSpinbox', font=default_font, padding=6,
                        fieldbackground=self.COLORS['bg_card'],
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text'],
                        insertcolor=self.COLORS['text'],
                        arrowcolor=self.COLORS['text'],
                        bordercolor=self.COLORS['border'],
                        lightcolor=self.COLORS['border'],
                        darkcolor=self.COLORS['border'],
                        selectbackground=self.COLORS['accent'],
                        selectforeground='white')
        style.map('TSpinbox',
                  bordercolor=[('focus', self.COLORS['accent'])],
                  lightcolor=[('focus', self.COLORS['accent'])],
                  darkcolor=[('focus', self.COLORS['accent'])])
        style.configure('TEntry', font=default_font, padding=6,
                        fieldbackground=self.COLORS['bg_card'],
                        foreground=self.COLORS['text'],
                        insertcolor=self.COLORS['text'],
                        bordercolor=self.COLORS['border'],
                        lightcolor=self.COLORS['border'],
                        darkcolor=self.COLORS['border'],
                        selectbackground=self.COLORS['accent'],
                        selectforeground='white')
        style.map('TEntry',
                  bordercolor=[('focus', self.COLORS['accent'])],
                  lightcolor=[('focus', self.COLORS['accent'])],
                  darkcolor=[('focus', self.COLORS['accent'])])
        style.configure('TNotebook', background=self.COLORS['bg_light'], tabposition='n',
                        borderwidth=0)
        # Tab 选中态：主色文字 + 白底，形成浮起效果
        # 子 Notebook（评估结果 20+ 子标签/部署面板）标签多，用小字号 + 紧凑
        # padding 换取更多完整可见标签，缓解一行塞不下导致的截断
        style.configure('TNotebook.Tab', font=small_font,
                        padding=(10, 5),
                        background=self.COLORS['bg_light'],
                        foreground=self.COLORS['text_light'],
                        borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', self.COLORS['bg_card']),
                              ('active', self.COLORS['bg_card'])],
                  foreground=[('selected', self.COLORS['accent']),
                              ('active', self.COLORS['text'])])
        # 主 Notebook 隐藏顶部标签条：侧边栏是唯一主导航（消除双导航冗余），
        # 子 Notebook（评估结果/部署面板内部）仍保留标签条。
        # 功能不受影响：.tabs()/.select()/<<NotebookTabChanged>>/Ctrl+Tab 均可用。
        style.layout('Main.TNotebook.Tab', [])
        style.configure('Main.TNotebook', background=self.COLORS['bg_light'],
                        borderwidth=0)
        style.configure('TLabelframe', font=default_font, padding=(14, 12),
                       background=self.COLORS['bg_light'],
                       borderwidth=1, relief='solid',
                       bordercolor=self.COLORS['border'])
        style.configure('TLabelframe.Label', font=heading_font, padding=(6, 2),
                       background=self.COLORS['bg_light'], foreground=self.COLORS['primary'])
        style.configure('Heading.TLabel', font=heading_font,
                       background=self.COLORS['bg_light'], foreground=self.COLORS['primary'])
        style.configure('Card.TFrame', background=self.COLORS['bg_card'], relief='solid',
                       borderwidth=1, bordercolor=self.COLORS['border'])
        style.configure('Card.TLabel', font=default_font, background=self.COLORS['bg_card'],
                       foreground=self.COLORS['text'])
        style.configure('CardTitle.TLabel', font=heading_font, background=self.COLORS['bg_card'],
                       foreground=self.COLORS['primary'])
        style.configure('CardValue.TLabel', font=('Microsoft YaHei', 14, 'bold'),
                       background=self.COLORS['bg_card'], foreground=self.COLORS['accent'])
        style.configure('Step.TButton', font=small_font, padding=2, width=2)
        style.configure('StepActive.TButton', font=('Microsoft YaHei', 8, 'bold'),
                       padding=2, width=2)
        style.configure('TProgressbar', thickness=8,
                        background=self.COLORS['accent'],
                        troughcolor=self.COLORS['bg_light'])
        style.configure('TSeparator', background=self.COLORS['border'])

        # 滚动条样式：浅灰槽 + 青绿滑块（悬停加深）
        style.configure('Vertical.TScrollbar',
                        background=self.COLORS['border'],
                        troughcolor=self.COLORS['bg_light'],
                        arrowcolor=self.COLORS['text_light'],
                        borderwidth=0, relief='flat',
                        width=10)
        style.configure('Horizontal.TScrollbar',
                        background=self.COLORS['border'],
                        troughcolor=self.COLORS['bg_light'],
                        arrowcolor=self.COLORS['text_light'],
                        borderwidth=0, relief='flat',
                        thickness=10)
        style.map('Vertical.TScrollbar',
                  background=[('active', self.COLORS['accent']),
                              ('pressed', self.COLORS['accent_hover'])],
                  arrowcolor=[('active', self.COLORS['accent_fg']),
                              ('pressed', self.COLORS['accent_fg'])])
        style.map('Horizontal.TScrollbar',
                  background=[('active', self.COLORS['accent']),
                              ('pressed', self.COLORS['accent_hover'])],
                  arrowcolor=[('active', self.COLORS['accent_fg']),
                              ('pressed', self.COLORS['accent_fg'])])
        # Treeview：表头 + 行高 + 斑马纹基础样式
        style.configure('Treeview',
                        font=default_font,
                        rowheight=32,
                        background=self.COLORS['bg_card'],
                        fieldbackground=self.COLORS['bg_card'],
                        foreground=self.COLORS['text'],
                        borderwidth=0, relief='flat')
        style.configure('Treeview.Heading',
                        font=('Microsoft YaHei', 10, 'bold'),
                        background=self.COLORS['bg_light'],
                        foreground=self.COLORS['primary'],
                        relief='flat', borderwidth=0,
                        padding=(10, 8))
        style.map('Treeview',
                  background=[('selected', self.COLORS['accent'])],
                  foreground=[('selected', 'white')])
        style.map('Treeview.Heading',
                  background=[('active', self.COLORS['accent_soft'])],
                  foreground=[('active', self.COLORS['accent'])])
        # 斑马纹 tag（插入数据时需为行打 even/odd tag）
        self._zebra_colors = {
            'even': self.COLORS['bg_card'],
            'odd': '#f1f5f9',  # 浅青灰（比 bg_light 略深，不与卡片底冲突）
        }
        style.configure('Bottom.TFrame', background=self.COLORS['primary'])
        style.configure('Bottom.TLabel', font=small_font, background=self.COLORS['primary'],
                       foreground='white')
        style.configure('Validate.TLabel', font=small_font, foreground=self.COLORS['danger'],
                       background=self.COLORS['bg_light'])
        style.configure('Tip.TLabel', font=small_font, foreground=self.COLORS['text_light'],
                       background=self.COLORS['bg_light'])
        style.configure('CardTip.TLabel', font=small_font, foreground=self.COLORS['text_light'],
                       background=self.COLORS['bg_card'])
        # 错误输入样式（初始配置，主题切换时在 _switch_theme 中更新）
        style.configure('Error.TEntry', fieldbackground=self.COLORS['danger_bg'])
        style.configure('Error.TSpinbox', fieldbackground=self.COLORS['danger_bg'])

        # ── 头部品牌栏样式 ──
        style.configure('Header.TFrame', background=self.COLORS['primary'])
        style.configure('HeaderBrand.TLabel',
                        font=('Microsoft YaHei', 13, 'bold'),
                        background=self.COLORS['primary'],
                        foreground=self.COLORS['header_fg'])
        style.configure('HeaderSub.TLabel', font=small_font,
                        background=self.COLORS['primary'],
                        foreground=self.COLORS['header_sub_fg'])
        style.configure('HeaderBadge.TLabel', font=small_font,
                        background=self.COLORS['accent'],
                        foreground=self.COLORS['accent_fg'], padding=(8, 2))

        # ── 侧边栏样式 ──
        style.configure('Sidebar.TFrame', background=self.COLORS['sidebar_bg'])
        style.configure('SidebarSection.TLabel', font=small_font,
                        background=self.COLORS['sidebar_bg'],
                        foreground=self.COLORS['text_light'])
        nav_pad = (12, 7)
        style.configure('Nav.TButton', font=default_font, padding=nav_pad, anchor='w',
                        background=self.COLORS['sidebar_bg'],
                        foreground=self.COLORS['text'])
        style.map('Nav.TButton',
                  background=[('active', self.COLORS['accent_soft']),
                              ('pressed', self.COLORS['accent_soft']),
                              ('!disabled', self.COLORS['sidebar_bg'])],
                  foreground=[('active', self.COLORS['accent']),
                              ('!disabled', self.COLORS['text'])])
        style.configure('NavActive.TButton', font=default_font, padding=nav_pad, anchor='w',
                        background=self.COLORS['accent_soft'],
                        foreground=self.COLORS['accent'])
        style.map('NavActive.TButton',
                  background=[('active', self.COLORS['accent_soft']),
                              ('!disabled', self.COLORS['accent_soft'])],
                  foreground=[('active', self.COLORS['accent']),
                              ('!disabled', self.COLORS['accent'])])

        # ── 步骤指示器（底部栏）样式 ──
        style.configure('StepActive.TLabel', font=small_font,
                        background=self.COLORS['primary'],
                        foreground=self.COLORS['step_active'])
        style.configure('StepDone.TLabel', font=small_font,
                        background=self.COLORS['primary'],
                        foreground=self.COLORS['step_done'])
        style.configure('StepInactive.TLabel', font=small_font,
                        background=self.COLORS['primary'],
                        foreground=self.COLORS['step_inactive'])

        self.style = style
        self._FONT_DEFAULT = default_font
        self._FONT_HEADING = heading_font
        self._FONT_SMALL = small_font

        # 跟踪所有 tk.Text 控件，便于主题切换时统一更新颜色
        self._text_widgets = []

        # 跟踪所有 ttk.Treeview 控件，便于主题切换时统一更新标签颜色
        self._tree_widgets = []

        # Section X: 图表缩放因子与当前主题（供快捷键与主题切换使用）
        self._chart_zoom_factors = {
            'risk_canvas': 1.0,
            'seir_canvas': 1.0,
            'heatmap_canvas': 1.0,
            'ml_compare_canvas': 1.0,
        }
        self._current_theme = 'light'


    def _setup_keyboard_shortcuts(self):
        """Section X: 快捷键绑定集中管理，使用字典映射

        将原本散落的 root.bind 调用聚合为字典，便于维护与扩展。
        保持向后兼容：原有快捷键功能不变。
        """
        shortcuts = {
            '<Control-n>': self._shortcut_add_contact,
            '<Control-N>': self._shortcut_add_contact,
            '<Control-o>': lambda e: self._load_from_csv(),
            '<Control-s>': lambda e: self._save_to_csv(),
            '<Control-e>': lambda e: self._export_results(),  # 新增：导出结果
            '<F1>': lambda e: self._show_help_window(),  # 新增：上下文帮助
            '<F5>': lambda e: self._run_assessment(),
            '<Control-q>': lambda e: self._on_closing(),
            '<Control-Tab>': lambda e: self._switch_tab(1),  # 新增：下一个标签页
            '<Control-Shift-Tab>': lambda e: self._switch_tab(-1),  # 新增：上一个标签页
            '<Control-plus>': lambda e: self._zoom_chart(1.2),  # 新增：图表放大
            '<Control-minus>': lambda e: self._zoom_chart(1 / 1.2),  # 新增：图表缩小
            '<Control-z>': lambda e: self._undo(),
            '<Control-y>': lambda e: self._redo(),
            '<Control-Z>': lambda e: self._undo(),
            '<Control-Y>': lambda e: self._redo(),
            '<Escape>': lambda e: self._close_top_dialog(),
            '<Control-w>': lambda e: (
                self._close_survival_dialog()
                if hasattr(self, '_survival_dialog') and self._survival_dialog
                else None),
            '<Delete>': lambda e: self._delete_selected_contact(),  # 新增：删除选中接触者
            '<Return>': lambda e: self._edit_selected_contact(),  # 新增：编辑选中行
        }
        for binding, callback in shortcuts.items():
            try:
                self.root.bind(binding, callback)
            except (tk.TclError, RuntimeError):
                # 个别绑定在不支持的平台上可能失败，忽略以保证其余快捷键可用
                pass

    def _shortcut_add_contact(self, event=None):
        """Ctrl+N: 根据当前标签页名称（非硬编码索引）添加接触者

        Section X: 通过标签页 text 属性判断类型，避免索引漂移导致误判。
        """
        try:
            current_tab = self.notebook.select()
            if not current_tab:
                return
            tab_text = self.notebook.tab(current_tab, 'text')
            if '家庭' in tab_text:
                self._add_family_member()
            elif '社会' in tab_text:
                self._add_social_contact()
        except Exception as e:
            LOGGER.debug("_add_contact_shortcut 添加接触者失败: %s", e)

    def _switch_tab(self, direction):
        """Ctrl+Tab/Ctrl+Shift+Tab: 切换标签页

        Args:
            direction: 1 表示下一个，-1 表示上一个
        """
        try:
            tabs = self.notebook.tabs()
            if not tabs:
                return
            current = self.notebook.select()
            if not current:
                return
            try:
                idx = tabs.index(current)
            except ValueError:
                return
            new_idx = (idx + direction) % len(tabs)
            self.notebook.select(tabs[new_idx])
        except Exception as e:
            LOGGER.debug("_switch_tab 切换标签页失败: %s", e)

    def _zoom_chart(self, factor):
        """Ctrl+加号/减号: 图表缩放

        遍历所有 matplotlib canvas，更新缩放因子并触发重绘。
        实际缩放效果由各 _draw_* 方法在重绘时根据 _chart_zoom_factors 决定。

        Args:
            factor: 缩放倍数（>1 放大，<1 缩小）
        """
        try:
            redraw_map = {
                'risk_canvas': '_draw_risk_chart',
                'seir_canvas': '_draw_seir_curve',
                'heatmap_canvas': '_draw_risk_heatmap',
                'ml_compare_canvas': '_update_ml_charts',
            }
            zoom_factors = getattr(self, '_chart_zoom_factors', None)
            if not zoom_factors:
                return
            for canvas_attr, redraw_method_name in redraw_map.items():
                canvas = getattr(self, canvas_attr, None)
                if canvas is None:
                    continue
                current = zoom_factors.get(canvas_attr, 1.0)
                new_factor = max(0.2, min(5.0, current * factor))
                zoom_factors[canvas_attr] = new_factor
                redraw_method = getattr(self, redraw_method_name, None)
                if redraw_method is not None:
                    try:
                        redraw_method()
                    except Exception as e:
                        LOGGER.debug("图表缩放重绘 %s 失败: %s", redraw_method_name, e)
        except Exception as e:
            LOGGER.debug("_zoom_chart 图表缩放失败: %s", e)

    def _delete_selected_contact(self):
        """Delete键: 删除选中接触者

        根据当前标签页名称判断调用家庭/社会删除方法。
        """
        try:
            current_tab = self.notebook.select()
            if not current_tab:
                return
            tab_text = self.notebook.tab(current_tab, 'text')
            if '家庭' in tab_text and hasattr(self, '_delete_family_member'):
                self._delete_family_member()
            elif '社会' in tab_text and hasattr(self, '_delete_social_contact'):
                self._delete_social_contact()
        except Exception as e:
            LOGGER.debug("_delete_selected_contact 删除接触者失败: %s", e)

    def _edit_selected_contact(self):
        """Enter键: 编辑选中行

        根据当前标签页名称判断调用家庭/社会编辑方法。
        """
        try:
            current_tab = self.notebook.select()
            if not current_tab:
                return
            tab_text = self.notebook.tab(current_tab, 'text')
            if '家庭' in tab_text and hasattr(self, '_edit_family_member'):
                self._edit_family_member()
            elif '社会' in tab_text and hasattr(self, '_edit_social_contact'):
                self._edit_social_contact()
        except Exception as e:
            LOGGER.debug("_edit_selected_contact 编辑接触者失败: %s", e)

    def _switch_theme(self, theme_name='light'):
        """Section X: 主题切换（浅色/深色/跟随系统）

        更新 self.COLORS 与 ttk Style 配置，刷新已有控件外观。
        已存在的 Tkinter 控件颜色需手动 refresh，ttk Style 修改可即时生效。

        Args:
            theme_name: 'light' 或 'dark'
        """
        try:
            if not hasattr(self, 'style'):
                return
            themes = {
                'light': {
                    'bg_light': '#f8f9fa', 'bg_card': '#ffffff',
                    'text': '#2c3e50', 'text_light': '#5a6c7d',
                    'border': '#dee2e6', 'primary': '#164e63',
                    'primary_dark': '#0f3a4a',
                    'accent': '#0f766e', 'accent_hover': '#115e59',
                    'accent_soft': '#ccfbf1', 'accent_fg': '#ffffff',
                    'success': '#2ecc71',
                    'warning': '#f39c12', 'danger': '#e74c3c',
                    'info': '#17a2b8', 'required': '#e74c3c',
                    'disabled': '#999999',
                    'warning_bg': '#fff3cd', 'warning_fg': '#856404',
                    'info_bg': '#d1ecf1',
                    'danger_bg': '#f8d7da', 'danger_fg': '#721c24',
                    'success_bg': '#d4edda', 'success_fg': '#155724',
                    'risk_high': '#e74c3c', 'risk_medium': '#f39c12', 'risk_low': '#2ecc71',
                    'risk_high_bg': '#ffe0e0', 'risk_medium_bg': '#fff3cd', 'risk_low_bg': '#e0ffe0',
                    'risk_high_fg': '#c0392b', 'risk_medium_fg': '#856404', 'risk_low_fg': '#27ae60',
                    'step_active': '#0f766e', 'step_inactive': '#aab7c4', 'step_done': '#2ecc71',
                    'header_fg': '#ffffff', 'header_sub_fg': '#cbd5e1',
                    'sidebar_bg': '#ffffff',
                    'chart_series_1': '#FF6B6B', 'chart_series_2': '#4ECDC4',
                    'chart_series_3': '#45B7D1', 'chart_series_4': '#FFA07A', 'chart_series_5': '#98D8C8',
                    'chart_facecolor': '#ffffff',
                    'chart_edgecolor': '#2c3e50',
                    'chart_grid_color': '#cccccc',
                },
                'dark': {
                    'bg_light': '#2d2d2d', 'bg_card': '#3d3d3d',
                    'text': '#e0e0e0', 'text_light': '#b0b0b0',
                    'border': '#555555', 'primary': '#111827',
                    'primary_dark': '#0b1220',
                    'accent': '#2dd4bf', 'accent_hover': '#5eead4',
                    'accent_soft': '#134e4a', 'accent_fg': '#062a22',
                    'success': '#58d68d',
                    'warning': '#f4d03f', 'danger': '#ec7063',
                    'info': '#5dade2', 'required': '#ec7063',
                    'disabled': '#666666',
                    'warning_bg': '#4a3a00', 'warning_fg': '#f4d03f',
                    'info_bg': '#1a3a4a',
                    'danger_bg': '#4a2020', 'danger_fg': '#ec7063',
                    'success_bg': '#1a3a1a', 'success_fg': '#58d68d',
                    'risk_high': '#ec7063', 'risk_medium': '#f4d03f', 'risk_low': '#58d68d',
                    'risk_high_bg': '#4a2020', 'risk_medium_bg': '#4a3a00', 'risk_low_bg': '#1a3a1a',
                    'risk_high_fg': '#ec7063', 'risk_medium_fg': '#f4d03f', 'risk_low_fg': '#58d68d',
                    'step_active': '#2dd4bf', 'step_inactive': '#555555', 'step_done': '#58d68d',
                    'header_fg': '#f3f4f6', 'header_sub_fg': '#9ca3af',
                    'sidebar_bg': '#3d3d3d',
                    'chart_series_1': '#FF6B6B', 'chart_series_2': '#4ECDC4',
                    'chart_series_3': '#45B7D1', 'chart_series_4': '#FFA07A', 'chart_series_5': '#98D8C8',
                    'chart_facecolor': '#3d3d3d',
                    'chart_edgecolor': '#e0e0e0',
                    'chart_grid_color': '#555555',
                },
            }
            colors = themes.get(theme_name, themes['light'])
            self.COLORS.update(colors)

            # 更新 ttk Style：覆盖与主题相关的样式
            self.style.configure('TFrame', background=colors['bg_light'])
            self.style.configure('TLabel', background=colors['bg_light'],
                                 foreground=colors['text'])
            self.style.configure('Heading.TLabel', background=colors['bg_light'],
                                 foreground=colors['primary'])
            self.style.configure('Card.TFrame', background=colors['bg_card'],
                                relief='solid', borderwidth=1, bordercolor=colors['border'])
            self.style.configure('Card.TLabel', background=colors['bg_card'],
                                 foreground=colors['text'])
            self.style.configure('CardTitle.TLabel', background=colors['bg_card'],
                                 foreground=colors['primary'])
            self.style.configure('Tip.TLabel', foreground=colors['text_light'],
                                 background=colors['bg_light'])
            self.style.configure('CardTip.TLabel', foreground=colors['text_light'],
                                 background=colors['bg_card'])
            self.style.configure('TSeparator', background=colors['border'])
            self.style.configure('Bottom.TFrame', background=colors['primary'])
            self.style.configure('Bottom.TLabel', background=colors['primary'],
                                 foreground='white')
            self.style.configure('CardValue.TLabel', background=colors['bg_card'],
                                 foreground=colors['accent'])
            self.style.configure('Validate.TLabel', foreground=colors['danger'],
                                 background=colors['bg_light'])

            # 错误输入样式跟随主题
            self.style.configure('Error.TEntry', fieldbackground=colors['danger_bg'])
            self.style.configure('Error.TSpinbox', fieldbackground=colors['danger_bg'])

            # 按钮样式跟随主题
            self.style.configure('TButton', background=colors['bg_card'],
                                 foreground=colors['text'])
            self.style.map('TButton',
                          background=[('active', colors['accent']),
                                     ('!disabled', colors['bg_card'])],
                          foreground=[('active', 'white'), ('!disabled', colors['text'])])

            # 单选框/复选框样式跟随主题
            # 注意：不显式设置background，让其继承父容器背景色，避免Windows原生控件渲染异常
            # （显式设置background会导致Windows主题下复选框选中标记显示为×而非✓）
            self.style.configure('TCheckbutton',
                                 foreground=colors['text'])
            self.style.map('TCheckbutton',
                          foreground=[('active', colors['text']),
                                     ('disabled', colors['text_light'])])
            self.style.configure('TRadiobutton',
                                 foreground=colors['text'])
            self.style.map('TRadiobutton',
                          foreground=[('active', colors['text']),
                                     ('disabled', colors['text_light'])])

            # 进度条样式跟随主题
            self.style.configure('Horizontal.TProgressbar',
                                 background=colors['accent'],
                                 troughcolor=colors['bg_light'])
            self.style.configure('TProgressbar',
                                 background=colors['accent'],
                                 troughcolor=colors['bg_light'])

            # 滚动条样式跟随主题
            self.style.configure('Vertical.TScrollbar',
                                 background=colors['border'],
                                 troughcolor=colors['bg_light'],
                                 arrowcolor=colors['text_light'])
            self.style.configure('Horizontal.TScrollbar',
                                 background=colors['border'],
                                 troughcolor=colors['bg_light'],
                                 arrowcolor=colors['text_light'])
            self.style.map('Vertical.TScrollbar',
                          background=[('active', colors['accent']),
                                      ('pressed', colors['accent_hover'])],
                          arrowcolor=[('active', colors['accent_fg']),
                                      ('pressed', colors['accent_fg'])])
            self.style.map('Horizontal.TScrollbar',
                          background=[('active', colors['accent']),
                                      ('pressed', colors['accent_hover'])],
                          arrowcolor=[('active', colors['accent_fg']),
                                      ('pressed', colors['accent_fg'])])

            # 滑块样式跟随主题
            self.style.configure('TScale', background=colors['bg_light'],
                                 troughcolor=colors['border'])

            # 通用按钮样式跟随主题
            self.style.configure('TButton', background=colors['bg_card'],
                                 foreground=colors['text'])
            self.style.map('TButton',
                          background=[('active', colors['accent_soft']),
                                      ('pressed', colors['accent_soft']),
                                      ('!disabled', colors['bg_card'])],
                          foreground=[('active', colors['accent']),
                                      ('!disabled', colors['text'])])

            # Accent.TButton 跟随主题切换（含 hover/pressed 状态）
            self.style.configure('Accent.TButton', background=colors['accent'],
                                 foreground=colors['accent_fg'])
            self.style.map('Accent.TButton',
                          background=[('active', colors['accent_hover']),
                                      ('pressed', colors['primary_dark']),
                                      ('!disabled', colors['accent'])],
                          foreground=[('active', colors['accent_fg']),
                                      ('pressed', colors['accent_fg']),
                                      ('!disabled', colors['accent_fg'])])

            # 输入控件：边框色 + 聚焦高亮
            for entry_style in ('TEntry', 'TCombobox', 'TSpinbox'):
                self.style.configure(entry_style,
                                     fieldbackground=colors['bg_card'],
                                     background=colors['bg_card'],
                                     foreground=colors['text'],
                                     bordercolor=colors['border'],
                                     lightcolor=colors['border'],
                                     darkcolor=colors['border'],
                                     selectbackground=colors['accent'],
                                     selectforeground='white')
                self.style.map(entry_style,
                              bordercolor=[('focus', colors['accent'])],
                              lightcolor=[('focus', colors['accent'])],
                              darkcolor=[('focus', colors['accent'])])

            # LabelFrame 跟随主题
            self.style.configure('TLabelframe', background=colors['bg_light'],
                                 bordercolor=colors['border'])
            self.style.configure('TLabelframe.Label', background=colors['bg_light'],
                                 foreground=colors['primary'])

            # 侧边栏快速评估按钮（tk.Button）跟随主题
            if hasattr(self, '_quick_assess_btn') and self._quick_assess_btn.winfo_exists():
                try:
                    self._quick_assess_btn.configure(
                        bg=colors['accent'], fg=colors['accent_fg'],
                        activebackground=colors['accent_hover'],
                        activeforeground=colors['accent_fg'])
                except Exception as e:
                    LOGGER.debug("更新快速评估按钮主题失败: %s", e)

            # 头部品牌栏跟随主题
            self.style.configure('Header.TFrame', background=colors['primary'])
            self.style.configure('HeaderBrand.TLabel', background=colors['primary'],
                                 foreground=colors['header_fg'])
            self.style.configure('HeaderSub.TLabel', background=colors['primary'],
                                 foreground=colors['header_sub_fg'])
            self.style.configure('HeaderBadge.TLabel', background=colors['accent'],
                                 foreground=colors['accent_fg'])
            if hasattr(self, '_header_logo') and self._header_logo.winfo_exists():
                try:
                    self._header_logo.configure(bg=colors['accent'], fg=colors['accent_fg'])
                except Exception as e:
                    LOGGER.debug("更新头部徽标主题失败: %s", e)

            # 侧边栏跟随主题
            self.style.configure('Sidebar.TFrame', background=colors['sidebar_bg'])
            self.style.configure('SidebarSection.TLabel', background=colors['sidebar_bg'],
                                 foreground=colors['text_light'])
            self.style.configure('Nav.TButton', background=colors['sidebar_bg'],
                                 foreground=colors['text'])
            self.style.map('Nav.TButton',
                          background=[('active', colors['accent_soft']),
                                      ('pressed', colors['accent_soft']),
                                      ('!disabled', colors['sidebar_bg'])],
                          foreground=[('active', colors['accent']),
                                      ('!disabled', colors['text'])])
            self.style.configure('NavActive.TButton', background=colors['accent_soft'],
                                 foreground=colors['accent'])
            self.style.map('NavActive.TButton',
                          background=[('active', colors['accent_soft']),
                                      ('!disabled', colors['accent_soft'])],
                          foreground=[('active', colors['accent']),
                                      ('!disabled', colors['accent'])])

            # 步骤指示器跟随主题
            self.style.configure('StepActive.TLabel', background=colors['primary'],
                                 foreground=colors['step_active'])
            self.style.configure('StepDone.TLabel', background=colors['primary'],
                                 foreground=colors['step_done'])
            self.style.configure('StepInactive.TLabel', background=colors['primary'],
                                 foreground=colors['step_inactive'])

            if hasattr(self, '_seir_param_sliders'):
                for slider in self._seir_param_sliders.values():
                    try:
                        if slider.winfo_exists():
                            slider.configure(
                                bg=colors.get('bg_card', '#ffffff'),
                                fg=colors.get('text', '#2c3e50'),
                                troughcolor=colors.get('bg_light', '#f0f4f8'),
                                activebackground=colors.get('accent', '#3498db'))
                    except Exception as e:
                        LOGGER.debug("更新 SEIR 滑块主题失败: %s", e)

            if hasattr(self, '_seir_r0_label') and self._seir_r0_label.winfo_exists():
                try:
                    self._seir_r0_label.configure(foreground=colors.get('accent', '#3498db'))
                except Exception as e:
                    LOGGER.debug("更新 SEIR R0 标签主题失败: %s", e)

            # 斑马纹 odd 颜色（根据实际 bg_card 亮度自适应）
            def _is_light(bg_hex):
                try:
                    h = bg_hex.lstrip('#')
                    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    return (r * 299 + g * 587 + b * 114) / 1000 > 180
                except Exception:
                    return True
            zebra_odd = '#f1f5f9' if _is_light(colors['bg_card']) else '#454a52'
            self.style.configure('Treeview', background=colors['bg_card'],
                                 fieldbackground=colors['bg_card'],
                                 foreground=colors['text'])
            self.style.configure('Treeview.Heading', background=colors['bg_light'],
                                 foreground=colors['primary'])
            self.style.map('Treeview',
                          background=[('selected', colors['accent'])],
                          foreground=[('selected', 'white')])
            self.style.map('Treeview.Heading',
                          background=[('active', colors['accent_soft'])],
                          foreground=[('active', colors['accent'])])
            # 同步更新斑马纹 tag（通过 _update_tree_widgets_theme 已处理，但这里确保
            # 未被注册的 treeview 仍能通过 tag_configure 获得新色——实际刷新由 _update_tree_widgets_theme 完成）

            # Notebook 样式跟随主题
            self.style.configure('TNotebook', background=colors['bg_light'],
                                 borderwidth=0)
            self.style.configure('TNotebook.Tab', background=colors['bg_light'],
                                 foreground=colors['text_light'])
            self.style.map('TNotebook.Tab',
                          background=[('selected', colors['bg_card']),
                                     ('active', colors['bg_card'])],
                          foreground=[('selected', colors['accent']),
                                     ('active', colors['text'])])

            # 更新所有 tk.Text 控件颜色
            self._update_text_widgets_theme()

            # 更新所有 Treeview 控件的标签颜色
            self._update_tree_widgets_theme()

            # 更新结果页风险大横幅
            self._update_banner_theme()

            # 优先级十：同步 matplotlib rcParams 与当前主题匹配
            self._apply_matplotlib_theme()

            try:
                if HelpWindow.is_open():
                    HelpWindow._instance.apply_theme(self.COLORS)
            except Exception as e:
                LOGGER.debug("同步帮助窗口主题失败: %s", e)

            self._current_theme = theme_name

            if hasattr(self, '_update_all_charts'):
                try:
                    self._update_all_charts()
                except Exception as e:
                    LOGGER.debug("主题切换后重绘图表失败: %s", e)
        except Exception as e:
            LOGGER.warning("_switch_theme 切换主题失败: %s", e)

    def _update_banner_theme(self):
        """主题切换后同步结果页风险大横幅的颜色。

        若横幅已处于"已评估"状态（含风险等级），重新调用 _update_risk_banner
        用最新 COLORS 刷新；否则重置为初始未评估态配色。
        """
        colors = getattr(self, 'COLORS', {})
        if not colors:
            return
        # 若有评估结果，通过 _update_risk_banner 刷新（它由结果页 Mixin 提供）
        if getattr(self, 'results', None) and hasattr(self, '_update_risk_banner'):
            try:
                self._update_risk_banner(self.results)
                return
            except Exception as e:
                LOGGER.debug("_update_banner_theme 刷新评估横幅失败: %s", e)
        # 未评估态：重置为 bg_card 配色
        for attr, is_label, fg in (
            ('_risk_banner', False, None),
            ('_risk_banner_icon', True, colors.get('accent')),
            ('_risk_banner_title', True, colors.get('text')),
            ('_risk_banner_subtitle', True, colors.get('text_light')),
            ('_risk_banner_pct', True, colors.get('accent')),
        ):
            w = getattr(self, attr, None)
            if w is None:
                continue
            try:
                if w.winfo_exists():
                    kw = {'bg': colors.get('bg_card', '#ffffff')}
                    if is_label and fg is not None:
                        kw['fg'] = fg
                    if not is_label:
                        kw['highlightbackground'] = colors.get('border', '#dee2e6')
                    w.configure(**kw)
            except Exception as e:
                LOGGER.debug("_update_banner_theme 更新 %s 失败: %s", attr, e)
        # 重置文字内容
        if hasattr(self, '_risk_banner_title') and self._risk_banner_title:
            try:
                self._risk_banner_title.configure(text="尚未进行风险评估")
            except Exception:
                pass
        if hasattr(self, '_risk_banner_subtitle') and self._risk_banner_subtitle:
            try:
                self._risk_banner_subtitle.configure(
                    text="请填写患者基本信息并添加至少一名接触者，然后点击上方「开始风险评估」按钮")
            except Exception:
                pass
        if hasattr(self, '_risk_banner_pct') and self._risk_banner_pct:
            try:
                self._risk_banner_pct.configure(text="")
            except Exception:
                pass
        if hasattr(self, '_risk_banner_icon') and self._risk_banner_icon:
            try:
                self._risk_banner_icon.configure(text="◉", fg=colors.get('accent'))
            except Exception:
                pass

    def _apply_matplotlib_theme(self):
        """优先级十：将 matplotlib rcParams 与当前 GUI 主题同步

        深色主题下图表背景改为深色、文字改为浅色、网格线改为半透明。
        """
        try:
            import matplotlib.pyplot as plt
            colors = getattr(self, 'COLORS', {})
            plt.rcParams['axes.facecolor'] = colors.get('chart_facecolor', '#ffffff')
            plt.rcParams['figure.facecolor'] = colors.get('chart_facecolor', '#ffffff')
            plt.rcParams['text.color'] = colors.get('chart_edgecolor', '#2c3e50')
            plt.rcParams['axes.labelcolor'] = colors.get('chart_edgecolor', '#2c3e50')
            plt.rcParams['xtick.color'] = colors.get('chart_edgecolor', '#2c3e50')
            plt.rcParams['ytick.color'] = colors.get('chart_edgecolor', '#2c3e50')
            plt.rcParams['axes.edgecolor'] = colors.get('chart_grid_color', '#cccccc')
            plt.rcParams['grid.color'] = colors.get('chart_grid_color', '#cccccc')
            plt.rcParams['grid.alpha'] = 0.3
        except Exception as e:
            LOGGER.debug("_apply_matplotlib_theme 应用 matplotlib 主题失败: %s", e)

    def _register_text_widget(self, text_widget):
        """注册 tk.Text 控件，主题切换时自动更新其颜色

        同时立即为控件应用当前主题的颜色配置。
        """
        if text_widget is None:
            return text_widget
        try:
            text_widget.configure(
                bg=self.COLORS['bg_card'],
                fg=self.COLORS['text'],
                insertbackground=self.COLORS['text'],
                selectbackground=self.COLORS['accent'],
                selectforeground='white'
            )
        except Exception as e:
            LOGGER.debug("_register_text_widget 配置文本控件颜色失败: %s", e)
        if not hasattr(self, '_text_widgets'):
            self._text_widgets = []
        if text_widget not in self._text_widgets:
            self._text_widgets.append(text_widget)
        return text_widget

    def _update_text_widgets_theme(self):
        """主题切换后更新所有已注册 tk.Text 控件的颜色"""
        colors = getattr(self, 'COLORS', {})
        if not colors or not hasattr(self, '_text_widgets'):
            return
        bg_card = colors.get('bg_card', '#ffffff')
        fg_text = colors.get('text', '#2c3e50')
        fg_warning = colors.get('warning', '#f39c12')
        fg_danger = colors.get('danger', '#e74c3c')
        fg_success = colors.get('success', '#2ecc71')
        fg_info = colors.get('info', '#17a2b8')
        fg_accent = colors.get('accent', '#3498db')
        is_dark = getattr(self, '_current_theme', 'light') == 'dark'
        # 深色主题下warning/info/success/danger可能与浅色主题同名但色值不同
        # 统一映射常见tag名到主题色
        tag_color_map = {
            'warning': fg_warning,
            'error': fg_danger,
            'danger': fg_danger,
            'success': fg_success,
            'info': fg_info,
            'title': fg_text,
            'heading': fg_accent,
            'subheading': fg_text,
        }
        for txt in list(self._text_widgets):
            try:
                if txt.winfo_exists():
                    txt.configure(
                        bg=bg_card,
                        fg=fg_text,
                        insertbackground=fg_text,
                        selectbackground=fg_accent,
                        selectforeground='white'
                    )
                    # 更新常见tag的前景色（忽略已设置特殊背景的tag）
                    for tag_name, tag_fg in tag_color_map.items():
                        try:
                            txt.tag_configure(tag_name, foreground=tag_fg)
                        except Exception as e:
                            LOGGER.debug("更新文本控件 tag %r 颜色失败: %s", tag_name, e)
            except Exception as e:
                LOGGER.debug("_update_text_widgets_theme 更新文本控件失败，将移除: %s", e)
                try:
                    self._text_widgets.remove(txt)
                except ValueError:
                    pass

    def _register_tree_widget(self, tree_widget):
        """注册 ttk.Treeview 控件，主题切换时自动更新其标签颜色"""
        if not hasattr(self, '_tree_widgets'):
            self._tree_widgets = []
        if tree_widget not in self._tree_widgets:
            self._tree_widgets.append(tree_widget)
        return tree_widget

    def _update_tree_widgets_theme(self):
        """主题切换后更新所有已注册 Treeview 控件的风险等级标签颜色 + 斑马纹"""
        colors = getattr(self, 'COLORS', {})
        if not colors or not hasattr(self, '_tree_widgets'):
            return
        bg_card = colors.get('bg_card', '#ffffff')
        try:
            h = bg_card.lstrip('#')
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            is_light = (r * 299 + g * 587 + b * 114) / 1000 > 180
        except Exception:
            is_light = True
        zebra_odd = '#f1f5f9' if is_light else '#454a52'
        for tree in list(self._tree_widgets):
            try:
                if tree.winfo_exists():
                    tree.tag_configure('risk_high',
                                       background=colors.get('risk_high_bg', '#ffe0e0'),
                                       foreground=colors.get('risk_high_fg', '#c0392b'))
                    tree.tag_configure('risk_medium',
                                       background=colors.get('risk_medium_bg', '#fff3cd'),
                                       foreground=colors.get('risk_medium_fg', '#856404'))
                    tree.tag_configure('risk_low',
                                       background=colors.get('risk_low_bg', '#e0ffe0'),
                                       foreground=colors.get('risk_low_fg', '#27ae60'))
                    # 排名树使用的标签（无背景色，仅前景色）
                    tree.tag_configure('rank_high', foreground=colors.get('risk_high', '#c0392b'))
                    tree.tag_configure('rank_medium', foreground=colors.get('risk_medium', '#856404'))
                    tree.tag_configure('rank_low', foreground=colors.get('risk_low', '#27ae60'))
                    # 斑马纹
                    tree.tag_configure('even', background=colors.get('bg_card', '#ffffff'))
                    tree.tag_configure('odd', background=zebra_odd)
            except Exception as e:
                LOGGER.debug("_update_tree_widgets_theme 更新树控件失败，将移除: %s", e)
                try:
                    self._tree_widgets.remove(tree)
                except ValueError:
                    pass

    def _show_help_window(self):
        """Section X: 显示独立帮助窗口（Toplevel），替代 messagebox

        HelpWindow 内部保证单例：若已打开则聚焦已有窗口。
        """
        try:
            if not hasattr(self, 'root'):
                return
            HelpWindow(self.root)
        except Exception as e:
            LOGGER.warning("_show_help_window 打开帮助窗口失败，将使用 messagebox 回退: %s", e)
            try:
                messagebox.showinfo("帮助", "帮助窗口暂时不可用，请参考项目 README。")
            except Exception as e2:
                LOGGER.debug("帮助回退 messagebox 也失败: %s", e2)

    def _undo(self):
        """撤销最近一次操作"""
        if hasattr(self, 'undo_manager') and self.undo_manager.undo():
            # 优先级六：通过 adapter 统一刷新（消除 ui_mode 分支）
            if self.adapter is not None:
                self.adapter.refresh_after_undo()
            if hasattr(self, 'auto_save'):
                self.auto_save.mark_dirty()

    def _redo(self):
        """重做最近一次撤销的操作"""
        if hasattr(self, 'undo_manager') and self.undo_manager.redo():
            # 优先级六：通过 adapter 统一刷新（消除 ui_mode 分支）
            if self.adapter is not None:
                self.adapter.refresh_after_undo()
            if hasattr(self, 'auto_save'):
                self.auto_save.mark_dirty()


    def _close_top_dialog(self):
        """关闭顶层弹窗"""
        for child in self.root.winfo_children():
            if isinstance(child, tk.Toplevel) and child.winfo_exists():
                child.destroy()
                break


    # ============ 数据安全与合规（Section: security） ============

    def _get_security(self):
        """获取数据安全管理器（懒初始化，缓存复用）。

        Returns:
            DataSecurityManager 实例；失败时返回 None。
        """
        try:
            if not hasattr(self, '_security_manager') or self._security_manager is None:
                from ..security import DataSecurityManager
                self._security_manager = DataSecurityManager()
            return self._security_manager
        except Exception:
            LOGGER.error("初始化数据安全管理器失败", exc_info=True)
            return None

    def _mask_for_export(self, data):
        """对导出数据应用统一强化脱敏（基于当前角色与导出脱敏开关）。

        底层导出器保持纯净；脱敏在应用层统一入口执行。
        """
        try:
            mgr = self._get_security()
            if mgr is None:
                return copy.deepcopy(data)
            return mgr.guard.prepare(data)
        except Exception:
            LOGGER.debug("导出脱敏失败（非致命），返回原始数据", exc_info=True)
            return copy.deepcopy(data)

    def _log_security_operation(self, operation, **kwargs):
        """记录关键操作审计（评估/导出/导入）。"""
        try:
            mgr = self._get_security()
            if mgr is None:
                return
            mgr.auditor.log_operation(operation, role=mgr.current_role, **kwargs)
        except Exception:
            LOGGER.debug("记录安全审计失败（非致命）", exc_info=True)

    def _open_security_panel(self):
        """打开数据安全与合规配置面板（Toplevel 对话框）。"""
        try:
            from .security_panel import SecurityPanel
            SecurityPanel(self.root, app=self)
        except Exception as e:
            LOGGER.error("打开数据安全面板失败: %s", e, exc_info=True)
            messagebox.showerror("错误", f"无法打开数据安全面板: {e}")

    def _open_ops_panel(self):
        """打开运维中心面板（离线状态/热更新/部署打包）。"""
        try:
            from .ops_panel import OpsPanel
            OpsPanel(self.root, app=self)
        except Exception as e:
            LOGGER.error("打开运维中心面板失败: %s", e, exc_info=True)
            messagebox.showerror("错误", f"无法打开运维中心面板: {e}")


    def _close_survival_dialog(self):
        """关闭生存分析对话框"""
        if hasattr(self, '_survival_dialog') and self._survival_dialog:
            try:
                self._survival_dialog.destroy()
            except (tk.TclError, RuntimeError, AttributeError):
                pass
            self._survival_dialog = None


    def _update_fill_progress(self):
        """更新底部填写进度条"""
        total = len(self.family_entries) + len(self.social_entries)

        valid_count = 0
        for entry in self.family_entries + self.social_entries:
            if all(entry.get(f) for f in ['name', 'age']):
                valid_count += 1

        pct = int(valid_count / max(total, 1) * 100) if total > 0 else 0
        self.fill_progress_var.set(pct)
        self.fill_count_label.config(text=f"{valid_count}/{total}")


    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导出 CSV 模板", command=self._export_csv_template)
        file_menu.add_separator()
        file_menu.add_command(label="从 CSV 导入数据", command=self._load_from_csv)
        file_menu.add_command(label="从 Excel 导入数据", command=self._load_from_excel)
        file_menu.add_command(label="从 JSON 导入数据", command=self._load_from_json)
        file_menu.add_command(label="从 API 导入数据", command=self._load_from_api)
        file_menu.add_command(label="从数据库导入数据", command=self._load_from_database)
        file_menu.add_separator()
        file_menu.add_command(label="从 HIS/EMR 标准接口接入...",
                              command=self._load_from_health_interop)
        file_menu.add_command(label="生成传染病报告卡...",
                              command=self._generate_report_card_safe)
        file_menu.add_separator()
        # 优先级一：最近打开子菜单（动态生成）
        self._recent_files_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="最近打开", menu=self._recent_files_menu)
        self._update_recent_files_menu()
        file_menu.add_separator()
        file_menu.add_command(label="保存数据到 CSV", command=self._save_to_csv)
        file_menu.add_separator()
        file_menu.add_command(label="清空表单", command=self._clear_form)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # 改进4：导出子菜单（将分散的导出功能集中）
        export_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="导出", menu=export_menu)
        export_menu.add_command(label="导出评估结果", command=self._export_results)
        export_menu.add_separator()
        export_menu.add_command(label="导出 Excel 报告", command=self._export_excel_report)
        export_menu.add_command(label="导出 JSON 报告", command=self._export_json_report)
        export_menu.add_command(label="导出 PDF 报告", command=self._export_pdf_report)
        export_menu.add_command(label="导出模型训练报告", command=self._export_ml_training_report)
        export_menu.add_separator()
        export_menu.add_command(label="导出到数据库", command=self._export_to_database)

        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="接口服务管理...", command=self._open_interface_service_panel)
        tools_menu.add_separator()
        tools_menu.add_command(label="数据安全与合规...", command=self._open_security_panel)
        tools_menu.add_command(label="运维中心...", command=self._open_ops_panel)
        tools_menu.add_separator()
        tools_menu.add_command(label="AI 设置...", command=self._open_ai_settings)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self._show_help)

        if KARAMAY_LOCALIZER_AVAILABLE and KARAMAY_VALIDATOR_AVAILABLE:
            kar_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="克拉玛依", menu=kar_menu)
            kar_menu.add_command(label="运行验证与敏感性分析",
                                 command=self._run_karamay_validation)
            kar_menu.add_command(label="显示本土化模块信息",
                                 command=self._show_karamay_info)

    def _open_interface_service_panel(self):
        """打开接口服务管理面板（HL7 MLLP监听、CDC Webhook等）"""
        try:
            from .interface_service_panel import InterfaceServicePanel
            panel = InterfaceServicePanel(self.root, app=self)
            # 添加初始日志
            panel._add_log('system', 'info', "接口服务管理面板已打开")
        except Exception as e:
            LOGGER.error("打开接口服务管理面板失败: %s", e, exc_info=True)
            messagebox.showerror("错误", f"无法打开接口服务管理面板: {e}")

    def _open_ai_settings(self):
        """Layer 4: 打开 AI 设置对话框

        用户提供 GUI 内配置 AI API Key 等敏感字段的入口，避免手动编辑
        ~/.tb_risk/ai_config.json 或设置环境变量。

        流程：
        1. 构造 AISettingsDialog（传入当前 AppConfig 与 on_save_callback）
        2. 用户点确定 → 对话框 _save 原地更新 AppConfig 并 save()，
           同时把 api_key 写入 ai_config.json
        3. on_save_callback 调用 ai.client._clear_client_cache() 让下次
           AI 调用重建客户端（避免使用旧 api_key/配置的缓存客户端）
        4. 用户点取消 → 不修改任何配置

        对话框关闭后 self.config（AppConfig）已被原地更新，无需额外刷新。
        """
        try:
            from .ai_settings_dialog import AISettingsDialog
            from ..ai.client import _clear_client_cache
        except ImportError:
            LOGGER.warning("AI 设置模块加载失败", exc_info=True)
            messagebox.showerror("错误", "AI 设置模块加载失败，请检查安装。")
            return

        cfg = getattr(self, 'config', None)
        if cfg is None:
            messagebox.showerror("错误", "配置未初始化，无法打开 AI 设置。")
            return

        def _on_ai_settings_saved():
            """保存成功后刷新 AI 客户端缓存"""
            try:
                _clear_client_cache()
                LOGGER.info("AI 设置已更新，客户端缓存已清空")
            except Exception:
                LOGGER.debug("清空 AI 客户端缓存失败", exc_info=True)

        try:
            dialog = AISettingsDialog(
                self.root, cfg, on_save_callback=_on_ai_settings_saved)
            self.root.wait_window(dialog)
        except Exception:
            LOGGER.warning("打开 AI 设置对话框失败", exc_info=True)
            messagebox.showerror("错误", "打开 AI 设置对话框失败。")

    def _generate_report_card_safe(self):
        """安全包装：调用HealthInteropMixin的传染病报告卡生成（兼容Mixin未加载情况）"""
        if hasattr(self, 'generate_tb_report_card'):
            try:
                self.generate_tb_report_card()
            except Exception as e:
                LOGGER.error("生成传染病报告卡失败: %s", e, exc_info=True)
                messagebox.showerror("错误", f"生成传染病报告卡失败: {e}")
        else:
            messagebox.showinfo(
                "提示",
                "医疗标准接口模块未加载，请确认 HealthInteropMixin 已正确注册。")

    def _update_recent_files_menu(self):
        """优先级一：动态刷新最近打开文件子菜单

        从 config.recent_files 读取路径列表，为每条路径创建菜单项。
        点击菜单项直接触发对应格式的导入管线并跳过文件选择阶段。
        """
        try:
            if not hasattr(self, '_recent_files_menu'):
                return
            # 清空现有菜单项
            self._recent_files_menu.delete(0, 'end')
            recent = getattr(self.config, 'recent_files', []) if hasattr(self, 'config') else []
            if not recent:
                self._recent_files_menu.add_command(
                    label="（无）", state='disabled')
                return
            for file_path in recent[:10]:
                # 截断过长的路径用于显示
                display = file_path if len(file_path) <= 60 else '...' + file_path[-57:]
                # 利用默认参数绑定当前路径，避免闭包延迟绑定问题
                self._recent_files_menu.add_command(
                    label=display,
                    command=lambda p=file_path: self._open_recent_file(p))
        except Exception:
            LOGGER.debug("刷新最近文件菜单失败", exc_info=True)

    def _open_recent_file(self, file_path: str):
        """优先级一：打开最近文件列表中的文件

        根据文件扩展名自动选择对应的导入管线，跳过文件选择阶段
        （直接传入已知路径）。
        """
        try:
            if not file_path or not os.path.exists(file_path):
                messagebox.showwarning("文件不存在",
                                       f"文件 \"{file_path}\" 已被移动或删除，将从最近列表中移除。")
                # 从最近列表中移除失效路径
                if hasattr(self, 'config'):
                    recent = getattr(self.config, 'recent_files', []) or []
                    recent = [p for p in recent if p != file_path]
                    self.config.recent_files = recent
                    try:
                        self.config.save()
                    except Exception as e:
                        LOGGER.warning("_open_file 保存配置移除失效路径失败: %s", e)
                    self._update_recent_files_menu()
                return
            # 根据扩展名选择管线
            ext = os.path.splitext(file_path)[1].lower()
            from .import_panel.pipeline import (
                CSVImportPipeline, JSONImportPipeline, ExcelImportPipeline)
            pipeline_cls = None
            if ext == '.csv':
                pipeline_cls = CSVImportPipeline
            elif ext == '.json':
                pipeline_cls = JSONImportPipeline
            elif ext in ('.xlsx', '.xls'):
                pipeline_cls = ExcelImportPipeline
            if pipeline_cls is None:
                messagebox.showwarning("不支持的格式",
                                       f"无法识别文件类型：{ext}（支持 CSV/JSON/Excel）")
                return
            # 创建管线并直接设置源路径（跳过文件选择阶段）
            pipeline = pipeline_cls(self)
            pipeline._source_file_path = file_path
            # 跳过 select_source，直接执行后续阶段
            mode = 'replace'
            if pipeline_cls.SUPPORTS_MODE_SELECTION:
                mode = pipeline._select_mode()
                if mode is None:
                    return
            # 进度对话框
            progress_win, progress_var, update_progress, close_progress = \
                self._show_progress_dialog(pipeline.get_progress_title())
            try:
                update_progress(10, "正在解析数据...")
                result = pipeline.parse_data({'file_path': file_path}, update_progress)
            except Exception as e:
                LOGGER.error("最近文件导入解析失败: %s", e, exc_info=True)
                close_progress()
                messagebox.showerror("错误", f"导入数据解析失败：{e}")
                return
            if result is None or (not result[0] and not result[1]):
                close_progress()
                messagebox.showinfo("提示", "没有可导入的数据")
                return
            family_entries, social_entries, metadata = result
            close_progress()
            if not pipeline._preview_and_confirm(family_entries, social_entries, metadata):
                return
            pipeline._load_data(family_entries, social_entries, mode)
        except Exception as e:
            LOGGER.error("打开最近文件失败: %s", e, exc_info=True)
            messagebox.showerror("错误", f"打开文件失败：{e}")


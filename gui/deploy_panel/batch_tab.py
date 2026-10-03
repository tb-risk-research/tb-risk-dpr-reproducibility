#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署与监控面板 - 批量接触者评估子标签

流程：导入接触者（CSV / JSON / 当前已录入）→ 后台批量打分
（三层集成 → ML 集成 → 轻量评分三级回退，core.deploy_services）→
风险排序表 → 导出 CSV。每次批量评估自动写入评估日志（漂移监控数据源）。
"""

from ._shared import *  # noqa: F401,F403
# 星号导入不携带下划线名，显式补充
from ._shared import _deploy  # noqa: F401
from ...utils import _is_yes  # noqa: F401
from ...core.ppv import percentile_ranks  # noqa: F401
from ...constants import PREVALENCE_CALIBERS  # noqa: F401

_SOURCE_LABELS = {
    'three_layer': '三层集成',
    'ml_ensemble': 'ML 集成',
    'lightweight': '轻量评分',
    'unavailable': '未评分',
}


class BatchAssessTabMixin:
    """批量接触者评估：导入 → 批量打分 → 排序 → 导出"""

    # ==================== 界面构建 ====================

    def _build_batch_assess_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text='批量接触者评估')

        self._batch_contacts = []
        self._batch_results = []
        self._batch_running = False

        # ── 顶部操作栏 ──
        top = ttk.Frame(tab)
        top.pack(fill='x', padx=10, pady=(10, 4))
        ttk.Button(top, text='导入 CSV',
                   command=self._batch_import_csv).pack(side='left', padx=(0, 4))
        ttk.Button(top, text='导入 JSON',
                   command=self._batch_import_json).pack(side='left', padx=4)
        ttk.Button(top, text='使用当前接触者',
                   command=self._batch_use_current).pack(side='left', padx=4)
        self._batch_source_label = ttk.Label(
            top, text='尚未加载数据 — 请导入接触者文件或使用当前已录入的接触者',
            style='Tip.TLabel')
        self._batch_source_label.pack(side='left', padx=12)

        bar = ttk.Frame(tab)
        bar.pack(fill='x', padx=10, pady=(0, 6))
        self._batch_start_btn = ttk.Button(
            bar, text='▶ 开始批量评估', style='Accent.TButton',
            command=self._batch_start)
        self._batch_start_btn.pack(side='left', padx=(0, 6))
        self._batch_export_btn = ttk.Button(
            bar, text='导出结果 CSV', command=self._batch_export)
        self._batch_export_btn.pack(side='left', padx=4)
        self._batch_progress = ttk.Progressbar(
            bar, length=180, mode='determinate')
        self._batch_progress.pack(side='left', padx=(14, 6))
        self._batch_status_label = ttk.Label(bar, text='', style='Tip.TLabel')
        self._batch_status_label.pack(side='left')

        # ── 汇总卡片 ──
        cards = ttk.Frame(tab)
        cards.pack(fill='x', padx=10, pady=(0, 6))
        self._batch_cards = {}
        for i, (key, label, color_key) in enumerate([
                ('n_total', '接触者总数', 'text'),
                ('n_high', '高风险', 'risk_high'),
                ('n_medium', '中风险', 'risk_medium'),
                ('n_low', '低风险', 'risk_low'),
                ('mean_score', '平均评分', 'accent'),
                ('source', '打分来源', 'primary')]):
            cards.columnconfigure(i, weight=1, uniform='batch_card')
            card = ttk.Frame(cards, style='Card.TFrame', padding=8)
            card.grid(row=0, column=i, sticky='nsew', padx=2)
            ttk.Label(card, text=label, font=self._FONT_SMALL,
                      style='CardTitle.TLabel').pack(anchor='w')
            val = ttk.Label(card, text='--', style='CardValue.TLabel',
                            foreground=self.COLORS.get(color_key, '#2c3e50'))
            val.pack(anchor='w')
            self._batch_cards[key] = val

        # ── 结果排序表 ──
        table_frame = ttk.LabelFrame(tab, text='风险排序（按评分降序）', padding=6)
        table_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # ── 决策口径标注（问题4：排序+截断点为主要决策形式）──
        # 分位列 = 队列内风险分位（不依赖概率口径的绝对正确性）；
        # PPV 按人群阳性率换算，双口径来自单一真值源，不可混用。
        # 注意：必须先于 Treeview pack（side='bottom' 预留底部空间，
        # 否则被 expand 的树挤没）
        _cc = PREVALENCE_CALIBERS['close_contact']
        _gp = PREVALENCE_CALIBERS['general_population']
        caliber_note = (
            '决策口径：以排序+截断点为主（分位 = 队列内风险分位），'
            '绝对概率仅作参考；PPV 按人群阳性率换算 — '
            f"密接人群 {_cc['display']} / 全人群 {_gp['display']}，"
            '两套口径不可混用')
        ttk.Label(table_frame, text=caliber_note,
                  style='Tip.TLabel', wraplength=720,
                  justify='left').pack(side='bottom', fill='x', pady=(4, 0))

        cols = ('rank', 'name', 'ctype', 'age', 'symptoms',
                'score', 'percentile', 'level', 'source')
        self._batch_tree = ttk.Treeview(table_frame, columns=cols,
                                        show='headings', height=14)
        self._register_tree_widget(self._batch_tree)
        for col, text, width in (
                ('rank', '排名', 50), ('name', '姓名', 110),
                ('ctype', '类型', 60), ('age', '年龄', 55),
                ('symptoms', '症状', 60), ('score', '风险评分', 85),
                ('percentile', '分位', 60), ('level', '风险等级', 75),
                ('source', '打分来源', 90)):
            self._batch_tree.heading(col, text=text)
            self._batch_tree.column(col, width=width,
                                    anchor='center' if col != 'name' else 'w')
        zebra = getattr(self, '_zebra_colors',
                        {'even': '#ffffff', 'odd': '#f1f5f9'})
        self._batch_tree.tag_configure('even', background=zebra['even'])
        self._batch_tree.tag_configure('odd', background=zebra['odd'])
        self._batch_tree.tag_configure(
            'risk_high', foreground=self.COLORS['risk_high'])
        self._batch_tree.tag_configure(
            'risk_medium', foreground=self.COLORS['risk_medium'])
        self._batch_tree.tag_configure(
            'risk_low', foreground=self.COLORS['risk_low'])
        scroll = ttk.Scrollbar(table_frame, orient='vertical',
                               command=self._batch_tree.yview)
        self._batch_tree.configure(yscrollcommand=scroll.set)
        self._batch_tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

    # ==================== 数据加载 ====================

    def _batch_set_contacts(self, contacts, source_desc):
        """设置批量评估数据并刷新状态显示。"""
        self._batch_contacts = contacts
        self._batch_results = []
        for item in self._batch_tree.get_children():
            self._batch_tree.delete(item)
        n_family = sum(1 for c in contacts
                       if c.get('_contact_type') == 'family')
        self._batch_source_label.configure(
            text=f'{source_desc}：共 {len(contacts)} 人'
                 f'（家庭 {n_family} / 社会 {len(contacts) - n_family}）')
        self._batch_status_label.configure(text='')
        self._batch_progress.configure(value=0, maximum=max(len(contacts), 1))
        for key in self._batch_cards:
            self._batch_cards[key].configure(text='--')

    @staticmethod
    def _batch_normalize_type(raw):
        """把原始类型字段归一化为 family/social。"""
        text = str(raw or '').strip().lower()
        if text in ('family', '家庭', '家庭成员', '家', 'fam', '1'):
            return 'family'
        return 'social'

    def _batch_tag_contacts(self, entries, default_type):
        """为 entry 列表打上 _contact_type 标记（不修改原 dict）。"""
        tagged = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            c = dict(e)
            raw = c.get('_contact_type') or c.get('contact_type')
            c['_contact_type'] = (self._batch_normalize_type(raw) if raw
                                  else default_type)
            tagged.append(c)
        return tagged

    def _batch_import_csv(self):
        """从 CSV 导入接触者（复用导入管线的三阶模糊字段映射）。"""
        file_path = filedialog.askopenfilename(
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='批量评估 — 从 CSV 导入接触者')
        if not file_path:
            return
        try:
            from ..import_panel.pipeline import ImportPipeline
            with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                mappings, _low = ImportPipeline.map_columns(list(columns))
                entries = [ImportPipeline.remap_entry(row, mappings)
                           for row in reader]
        except (OSError, csv.Error) as e:
            messagebox.showerror('导入失败', f'CSV 解析失败：{e}')
            return
        contacts = self._batch_tag_contacts(entries, 'social')
        if not contacts:
            messagebox.showinfo('提示', 'CSV 中没有可用的接触者记录')
            return
        self._batch_set_contacts(contacts, f'CSV：{os.path.basename(file_path)}')

    def _batch_import_json(self):
        """从 JSON 导入接触者（{family: [...], social: [...]} 或列表）。"""
        file_path = filedialog.askopenfilename(
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            title='批量评估 — 从 JSON 导入接触者')
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror('导入失败', f'JSON 解析失败：{e}')
            return
        contacts = []
        if isinstance(data, dict):
            contacts += self._batch_tag_contacts(
                data.get('family', data.get('family_members', [])), 'family')
            contacts += self._batch_tag_contacts(
                data.get('social', data.get('social_contacts', [])), 'social')
        elif isinstance(data, list):
            contacts = self._batch_tag_contacts(data, 'family')
        if not contacts:
            messagebox.showinfo('提示', 'JSON 中没有可用的接触者记录')
            return
        self._batch_set_contacts(contacts, f'JSON：{os.path.basename(file_path)}')

    def _batch_use_current(self):
        """使用当前表单中已录入的家庭成员与社会接触者。"""
        family = self._batch_tag_contacts(
            getattr(self, 'family_entries', []) or [], 'family')
        social = self._batch_tag_contacts(
            getattr(self, 'social_entries', []) or [], 'social')
        contacts = family + social
        if not contacts:
            messagebox.showinfo(
                '提示', '当前表单中没有接触者 — 请先在「家庭成员信息」或'
                '「社会接触者信息」标签页录入')
            return
        self._batch_set_contacts(contacts, '当前表单')

    # ==================== 批量打分 ====================

    def _batch_start(self):
        if self._batch_running:
            messagebox.showinfo('提示', '批量评估正在进行中，请稍候')
            return
        if not self._batch_contacts:
            messagebox.showinfo('提示', '请先导入或加载接触者数据')
            return
        self._batch_running = True
        self._batch_start_btn.configure(state='disabled')
        self._batch_status_label.configure(text='评估中...')
        contacts = list(self._batch_contacts)
        self._start_training_thread(
            lambda: self._batch_worker(contacts))

    def _batch_worker(self, contacts):
        """后台线程：逐条打分（三级回退），完成后回主线程刷新。"""
        ml_predictor = getattr(self, 'ml_predictor', None)
        integrator = None
        service = getattr(self, '_assessment_service', None)
        if service is not None and hasattr(service, 'get_integrator'):
            try:
                integrator = service.get_integrator()
            except Exception as e:  # noqa: BLE001
                LOGGER.debug('获取集成器失败，批量评估降级为 ML/轻量: %s', e)

        total = len(contacts)

        def _progress(done, _total):
            # 节流：每 5 条或最后一条更新一次进度
            if done % 5 and done != total:
                return
            try:
                self.root.after(0, lambda d=done: (
                    self._batch_progress.configure(value=d),
                    self._batch_status_label.configure(
                        text=f'评估中... {d}/{total}')))
            except Exception:  # noqa: BLE001 — 窗口关闭后静默
                pass

        try:
            result = _deploy.score_contacts_batch(
                contacts, ml_predictor=ml_predictor, integrator=integrator,
                assessment=None, source='batch', progress_cb=_progress)
        except Exception as e:  # noqa: BLE001 — 绝不向 GUI 抛异常
            LOGGER.error('批量评估失败: %s', e, exc_info=True)
            result = {'available': False, 'results': [],
                      'summary': {}, 'reason': str(e)}
        try:
            self.root.after(0, lambda r=result: self._batch_finish(r))
        except Exception:  # noqa: BLE001
            self._batch_running = False

    def _batch_finish(self, result):
        """主线程：渲染批量评估结果。"""
        self._batch_running = False
        self._batch_start_btn.configure(state='normal')
        summary = result.get('summary') or {}
        results = result.get('results') or []
        self._batch_results = results

        if not result.get('available'):
            self._batch_status_label.configure(
                text='评估失败：' + (result.get('reason') or '未知原因'))
            return

        self._batch_status_label.configure(
            text=f"完成 — {summary.get('n_scored', 0)}/{summary.get('n_total', 0)} 人已评分")

        # 汇总卡片
        mean = summary.get('mean_score')
        self._batch_cards['n_total'].configure(
            text=str(summary.get('n_total', 0)))
        self._batch_cards['n_high'].configure(
            text=str(summary.get('n_high', 0)))
        self._batch_cards['n_medium'].configure(
            text=str(summary.get('n_medium', 0)))
        self._batch_cards['n_low'].configure(
            text=str(summary.get('n_low', 0)))
        self._batch_cards['mean_score'].configure(
            text=f'{mean:.1f}' if isinstance(mean, (int, float)) else '--')
        self._batch_cards['source'].configure(
            text=_SOURCE_LABELS.get(summary.get('score_source'), '--'))

        # 排序表（含队列内分位列：排序决策形式的数据基础，问题4）
        for item in self._batch_tree.get_children():
            self._batch_tree.delete(item)
        scores_all = [r.get('final_score') if isinstance(r.get('final_score'),
                                                          (int, float))
                      else 0.0 for r in results]
        percentiles = percentile_ranks(scores_all) if results else []
        level_tag = {'高风险': 'risk_high', '中风险': 'risk_medium',
                     '低风险': 'risk_low'}
        for idx, r in enumerate(results):
            score = r.get('final_score')
            level = r.get('risk_level', '未知')
            tags = ['even' if idx % 2 == 0 else 'odd']
            if level in level_tag:
                tags.append(level_tag[level])
            self._batch_tree.insert('', 'end', tags=tags, values=(
                idx + 1,
                r.get('name', ''),
                '家庭' if r.get('contact_type') == 'family' else '社会',
                r.get('age') if r.get('age') not in (None, '') else '--',
                '是' if _is_yes(r.get('has_symptoms', 0)) else '否',
                f'{score:.1f}' if isinstance(score, (int, float)) else '--',
                f'{percentiles[idx]:.0f}%',
                level,
                _SOURCE_LABELS.get(r.get('score_source'), '--'),
            ))

    # ==================== 导出 ====================

    def _batch_export(self):
        if not self._batch_results:
            messagebox.showinfo('提示', '暂无批量评估结果可导出')
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv')],
            initialfile='批量评估结果.csv',
            title='导出批量评估结果')
        if not file_path:
            return
        try:
            with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['排名', '姓名', '类型', '年龄', '症状',
                                 '风险评分', '分位%', '风险等级', '打分来源',
                                 'ML评分', 'GNN评分'])
                scores_all = [
                    r.get('final_score') if isinstance(r.get('final_score'),
                                                       (int, float)) else 0.0
                    for r in self._batch_results]
                percentiles = percentile_ranks(scores_all)
                for idx, r in enumerate(self._batch_results):
                    score = r.get('final_score')
                    ml_prob = r.get('ml_prob')
                    gnn_prob = r.get('gnn_prob')
                    writer.writerow([
                        idx + 1,
                        r.get('name', ''),
                        '家庭' if r.get('contact_type') == 'family' else '社会',
                        r.get('age') or '',
                        '是' if _is_yes(r.get('has_symptoms', 0)) else '否',
                        f'{score:.2f}' if isinstance(score, (int, float)) else '',
                        f'{percentiles[idx]:.1f}',
                        r.get('risk_level', ''),
                        _SOURCE_LABELS.get(r.get('score_source'), ''),
                        f'{ml_prob:.2f}' if isinstance(ml_prob, (int, float)) else '',
                        f'{gnn_prob:.2f}' if isinstance(gnn_prob, (int, float)) else '',
                    ])
        except OSError as e:
            messagebox.showerror('导出失败', f'文件写入失败：{e}')
            return
        messagebox.showinfo('成功', f'已导出 {len(self._batch_results)} 条结果：\n{file_path}')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 结果面板 - 结果显示 Mixin（Classic / Wizard）

Section IX 增强：
- ``_display_results_classic`` 在纯文本结果之后追加可排序 Treeview 表格，
  展示潜在患者列表（姓名 / 类型 / 年龄 / 发病概率 / 优先级 / 建议），支持
  点击列头按概率/风险等级/姓名排序。
- 新增 ``_add_results_treeview`` / ``_sort_treeview_column`` 方法。
- 新增 ``_show_stale_banner`` / ``_hide_stale_banner`` / ``_mark_results_stale``
  方法，配合 init_ui.py 中的 stale 横幅容器显示"数据已修改，请重新评估"提示。
"""

from ._shared import *

# 证据等级短标签（Treeview 列用；完整说明见 LAB_EVIDENCE_GRADES）
_EVIDENCE_SHORT_LABEL = {
    0: '症状推断', 1: '影像学', 2: '涂片分级', 3: 'GeneXpert',
}


def _evidence_label(grade):
    return _EVIDENCE_SHORT_LABEL.get(int(grade or 0), '症状推断')


def _format_three_layer_text(three_layer):
    """把三层递进架构报告渲染为文本（防御性，缺字段回退）。"""
    if not isinstance(three_layer, dict):
        return ''
    lines = []
    layers = three_layer.get('layers', {})
    summary = three_layer.get('summary', {})
    decision = three_layer.get('decision', {})

    l1 = layers.get('layer1_individual', {})
    l2 = layers.get('layer2_network', {})
    if not l1 and not l2:
        return ''

    lines.append("--- 三层递进架构评估（个体基础层 → 网络增强层 → 社区干预层） ---")
    # 第 1 层
    lines.append(f"第1层 个体基线概率 P_base：{l1.get('p_base_percent', 'N/A'):.1f}%"
                 if isinstance(l1.get('p_base_percent'), (int, float))
                 else "第1层 个体基线概率 P_base：N/A")
    # 第 2 层
    incr = l2.get('network_increment')
    net_risk = l2.get('network_risk')
    aware = l2.get('network_aware', False)
    if isinstance(incr, (int, float)) and isinstance(net_risk, (int, float)):
        incr_str = f"{incr:+.1f}%"
        aware_str = "（使用网络结构）" if aware else "（未使用网络结构）"
        lines.append(f"第2层 网络感知风险 {net_risk:.1f}%，网络增量 {incr_str}{aware_str}")
    # 第 3 层
    l3 = layers.get('layer3_intervention') or {}
    if l3.get('summary'):
        best = l3['summary'].get('best_strategy_label') or 'N/A'
        averted = l3['summary'].get('best_averted_percent')
        averted_str = f"{averted:.1f}%" if isinstance(averted, (int, float)) else "N/A"
        lines.append(f"第3层 最佳干预策略：{best}（可避免病例 {averted_str}）")
    # 决策层
    combined = decision.get('combined_probability')
    gating = decision.get('gating')
    if isinstance(combined, (int, float)):
        gating_str = f"，门控 g={gating:.2f}" if isinstance(gating, (int, float)) else ""
        lines.append(f"决策层 联合决策风险 {combined:.1f}%{gating_str}")
    if summary.get('best_strategy'):
        lines.append(f"社区干预建议：{summary.get('best_strategy')}")
    lines.append("")
    return "\n".join(lines)


class DisplayMixin:
    """结果显示方法（Classic 界面 + Wizard 界面）"""

    def _insert_three_layer_summary(self, text_widget):
        """在结果文本区插入三层递进架构 + 任务分解摘要（防御性）。"""
        try:
            three_layer = self.results.get('three_layer') if isinstance(
                self.results, dict) else None
            if three_layer:
                text = _format_three_layer_text(three_layer)
                if text:
                    text_widget.insert('end', text, 'subheading')

            tasks = self.results.get('task_reports') if isinstance(
                self.results, dict) else None
            if isinstance(tasks, dict) and tasks:
                text_widget.insert(
                    'end', "--- 任务分解与临床终点 ---\n", 'subheading')
                for tid in ('A', 'B', 'C'):
                    rep = tasks.get(tid) or {}
                    name = rep.get('task_name') or f'任务 {tid}'
                    decision = rep.get('decision_meaning') or ''
                    text_widget.insert(
                        'end', f"任务{tid} {name}\n  决策含义：{decision}\n")
                text_widget.insert('end', "\n")
        except Exception:
            LOGGER.debug("三层/任务摘要展示失败（不影响主结果）", exc_info=True)


    def _display_results_classic(self):
        """在传统 GUI 中显示评估结果（防御性编程）"""
        if not self.results:
            return

        # 防御性编程：检查控件是否存在
        if not hasattr(self, 'result_text') or self.result_text is None:
            return

        # 更新顶部风险等级大横幅
        self._update_risk_banner(self.results)

        self.result_text.delete('1.0', 'end')

        # 综合风险评估结果
        self.result_text.insert('end', "=== 结核病传播风险评估结果 ===\n\n", 'title')
        self.result_text.insert('end', f"综合风险等级：{self.results.get('overall_risk', 'N/A')}\n", 'heading')
        self.result_text.insert('end', f"综合风险概率：{self.results.get('base_infection_probability', 0):.1f}%\n")
        self.result_text.insert('end', f"综合建议：{self.results.get('overall_suggestion', '')}\n\n")

        # 三层递进架构 + 任务分解摘要（若有）
        self._insert_three_layer_summary(self.result_text)

        # 各项指标风险评估结果
        self.result_text.insert('end', "--- 各项风险指标评估结果 ---\n", 'heading')
        for indicator, risk_info in self.results.get('individual_risks', {}).items():
            self.result_text.insert('end', f"{indicator}: {risk_info.get('risk', 'N/A')}\n")
            self.result_text.insert('end', f"  建议：{risk_info.get('suggestion', '')}\n")

        # 潜在患者识别结果
        self.result_text.insert('end', "\n--- 潜在患者识别结果 ---\n", 'heading')

        # 家庭成员
        family_patients = self.results.get('potential_patients', {}).get('family', [])
        if family_patients:
            self.result_text.insert('end', "\n家庭接触者风险评估:\n", 'subheading')
            for member in family_patients:
                name = member.get('name', '未知')
                rel = member.get('relationship', '家庭成员')
                age = member.get('age', 'N/A')
                prob = member.get('disease_probability', 0)
                priority = member.get('priority', 'N/A')
                rec = member.get('recommendation', '')
                self.result_text.insert('end', f"- {name} ({rel}, {age}岁): ")
                self.result_text.insert('end', f"发病概率 {prob:.1f}%, ")
                self.result_text.insert('end', f"优先级：{priority}\n")
                latent = member.get('latent_infection_prob')
                short_term = member.get('short_term_active_risk')
                if latent is not None:
                    self.result_text.insert('end',
                        f"  潜伏感染概率 {latent*100:.1f}%，"
                        f"近期(1-2年)发病风险 {short_term*100:.2f}%\n")
                self.result_text.insert('end', f"  建议：{rec}\n")

        # 社会接触者
        social_patients = self.results.get('potential_patients', {}).get('social', [])
        if social_patients:
            self.result_text.insert('end', "\n社会接触者风险评估:\n", 'subheading')
            for contact in social_patients:
                name = contact.get('name', '未知')
                age = contact.get('age', None)
                prob = contact.get('disease_probability', 0)
                priority = contact.get('priority', 'N/A')
                rec = contact.get('recommendation', '')
                age_str = f" ({age}岁)" if age is not None else ""
                self.result_text.insert('end', f"- {name}{age_str}: ")
                self.result_text.insert('end', f"发病概率 {prob:.1f}%, ")
                self.result_text.insert('end', f"优先级：{priority}\n")
                latent = contact.get('latent_infection_prob')
                short_term = contact.get('short_term_active_risk')
                if latent is not None:
                    self.result_text.insert('end',
                        f"  潜伏感染概率 {latent*100:.1f}%，"
                        f"近期(1-2年)发病风险 {short_term*100:.2f}%\n")
                self.result_text.insert('end', f"  建议：{rec}\n")

        # 配置文本标签
        self.result_text.tag_configure('title', font=('Microsoft YaHei', 12, 'bold'))
        self.result_text.tag_configure('heading', font=('Microsoft YaHei', 10, 'bold'))
        self.result_text.tag_configure('subheading', font=('Microsoft YaHei', 10, 'bold', 'italic'))

        # Section IX: 在纯文本结果之后追加可排序 Treeview 表格
        # 优先使用 init_ui.py 缓存的 _results_text_tab（标签页容器），
        # 这样 Treeview 作为 text_frame 的兄弟节点，布局更合理
        try:
            tree_parent = getattr(self, '_results_text_tab', None)
            if tree_parent is None:
                # 回退到 result_text 的祖父节点（防御性编程）
                tree_parent = self.result_text.master.master
            self._add_results_treeview(tree_parent)
        except Exception:
            LOGGER.error("添加结果 Treeview 失败", exc_info=True)

    def _display_results_wizard(self):
        """在向导界面中显示评估结果（仪表盘模式）"""
        if not self.results:
            return

        try:
            # 更新仪表盘卡片
            if hasattr(self, 'risk_card') and 'value_label' in self.risk_card:
                risk_level = self.results.get('overall_risk', '未知')
                risk_icon = {'高风险': '🔴', '中风险': '🟡', '低风险': '🟢'}.get(risk_level, '🟢')
                self.risk_card['value_label'].configure(text=f"{risk_icon} {risk_level}")

            # 更新高风险接触者卡片
            if hasattr(self, 'high_risk_card') and 'value_label' in self.high_risk_card:
                high_risk_count = 0
                for member in self.results.get('potential_patients', {}).get('family', []):
                    if member.get('priority', '') in ('高', '极高') or member.get('disease_probability', 0) > 50:
                        high_risk_count += 1
                for contact in self.results.get('potential_patients', {}).get('social', []):
                    if contact.get('priority', '') in ('高', '极高') or contact.get('disease_probability', 0) > 50:
                        high_risk_count += 1
                self.high_risk_card['value_label'].configure(text=f"⚠️ {high_risk_count}人")

            # 更新ML集成预测卡片
            if hasattr(self, 'ml_card') and 'value_label' in self.ml_card:
                avg_prob = self.results.get('base_infection_probability', 0)
                self.ml_card['value_label'].configure(text=f"{avg_prob:.1f}%")

            # 更新 Treeview 数据（如果存在）
            self._refresh_wizard_trees()

        except Exception as e:
            LOGGER.error("捕获未处理异常", exc_info=True)
            if not getattr(self, '_closing', False):
                self.root.after(0, lambda: messagebox.showerror(
                    "显示错误", f"向导界面结果显示失败:\n{str(e)}"))

    # ==================== Section IX: 可排序 Treeview ====================

    # 优先级中文到数值的映射（用于排序）
    _PRIORITY_ORDER = {'极高': 4, '高': 3, '中': 2, '低': 1}

    def _add_results_treeview(self, parent):
        """在结果区域下方创建可排序 Treeview 表格

        展示潜在患者列表，支持点击列头按概率/风险等级/姓名排序。
        若已存在 Treeview，先清空再重建，避免重复堆积。
        使用 ``after=`` 参数将 Treeview 插入到文本结果区之后、风险评分分布图之前。

        Args:
            parent: 父容器（通常是 result_text 的 master，即 text_tab）
        """
        try:
            # 若已存在 Treeview 容器，先销毁旧实例，避免重复堆积
            existing = getattr(self, '_results_treeview_frame', None)
            if existing is not None:
                try:
                    existing.destroy()
                except Exception:
                    pass
                self._results_treeview_frame = None
                self._results_treeview = None

            # 创建新的容器框架（放在父容器内）
            tree_frame = ttk.LabelFrame(parent, text="潜在患者列表（点击列头排序）", padding=5)
            # 使用 after= 参数将 Treeview 插入到 text_frame 之后，
            # 确保它出现在纯文本结果区之后、风险评分分布图之前
            text_frame = getattr(self, 'result_text', None)
            if text_frame is not None:
                text_frame = text_frame.master  # text_frame（LabelFrame）
            pack_kwargs = {'fill': 'both', 'expand': False, 'padx': 5, 'pady': 5}
            if text_frame is not None and text_frame is not parent:
                pack_kwargs['after'] = text_frame
            tree_frame.pack(**pack_kwargs)
            self._results_treeview_frame = tree_frame

            # 优先级二：搜索框 + 概率范围筛选控件
            filter_frame = ttk.Frame(tree_frame)
            filter_frame.pack(fill='x', pady=(0, 5))

            ttk.Label(filter_frame, text="🔍 搜索:").pack(side='left', padx=(0, 5))
            self._results_search_var = tk.StringVar()
            self._results_search_var.trace_add('write', lambda *_: self._filter_results_treeview())
            ttk.Entry(filter_frame, textvariable=self._results_search_var, width=20).pack(side='left', padx=(0, 10))

            ttk.Label(filter_frame, text="概率范围:").pack(side='left', padx=(5, 5))
            self._results_prob_filter_var = tk.StringVar(value="全部")
            prob_combo = ttk.Combobox(filter_frame, textvariable=self._results_prob_filter_var,
                                       values=["全部", "> 50%", "20%-50%", "< 20%"],
                                       width=10, state='readonly')
            prob_combo.pack(side='left', padx=(0, 5))
            prob_combo.bind('<<ComboboxSelected>>', lambda _: self._filter_results_treeview())

            # 列定义：姓名 / 类型 / 年龄 / 发病概率 / 优先级 / 证据等级 / 建议
            columns = ('name', 'type', 'age', 'probability', 'priority',
                       'evidence', 'suggestion')
            tree = ttk.Treeview(
                tree_frame, columns=columns, show='headings', height=8
            )

            tree.heading('name', text='姓名',
                         command=lambda c='name': self._sort_treeview_column(tree, c, False))
            tree.heading('type', text='类型',
                         command=lambda c='type': self._sort_treeview_column(tree, c, False))
            tree.heading('age', text='年龄',
                         command=lambda c='age': self._sort_treeview_column(tree, c, False))
            tree.heading('probability', text='发病概率(%)',
                         command=lambda c='probability': self._sort_treeview_column(tree, c, False))
            tree.heading('priority', text='优先级',
                         command=lambda c='priority': self._sort_treeview_column(tree, c, False))
            tree.heading('evidence', text='证据等级',
                         command=lambda c='evidence': self._sort_treeview_column(tree, c, False))
            tree.heading('suggestion', text='建议',
                         command=lambda c='suggestion': self._sort_treeview_column(tree, c, False))

            tree.column('name', width=100, anchor='w')
            tree.column('type', width=70, anchor='center')
            tree.column('age', width=60, anchor='center')
            tree.column('probability', width=100, anchor='center')
            tree.column('priority', width=80, anchor='center')
            tree.column('evidence', width=90, anchor='center')
            tree.column('suggestion', width=300, anchor='w')

            # 垂直滚动条
            tree_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
            tree.configure(yscrollcommand=tree_scroll.set)
            tree.pack(side='left', fill='both', expand=True)
            tree_scroll.pack(side='right', fill='y')

            self._results_treeview = tree

            # 填充数据
            self._populate_results_treeview(tree)
        except Exception:
            LOGGER.error("创建结果 Treeview 失败", exc_info=True)

    def _populate_results_treeview(self, tree):
        """从 self.results 填充 Treeview 数据

        Args:
            tree: ttk.Treeview 实例
        """
        try:
            # 清空已有项
            for item in tree.get_children():
                tree.delete(item)

            if not self.results:
                return

            potential = self.results.get('potential_patients', {}) or {}
            family_patients = potential.get('family', []) or []
            social_patients = potential.get('social', []) or []

            # 优先级二：保存原始数据列表用于筛选后恢复
            raw_data = []

            for member in family_patients:
                name = member.get('name', '未知')
                age = member.get('age', 'N/A')
                prob = member.get('disease_probability', 0)
                priority = member.get('priority', 'N/A')
                rec = member.get('recommendation', '') or member.get('suggestion', '')
                ev_label = _evidence_label(
                    int(member.get('lab_evidence_grade', 0) or 0))
                values = (name, '家庭', age, f"{prob:.1f}", priority, ev_label, rec)
                tree.insert('', 'end', values=values)
                raw_data.append(values)

            for contact in social_patients:
                name = contact.get('name', '未知')
                age = contact.get('age', 'N/A')
                prob = contact.get('disease_probability', 0)
                priority = contact.get('priority', 'N/A')
                rec = contact.get('recommendation', '') or contact.get('suggestion', '')
                ev_label = _evidence_label(
                    int(contact.get('lab_evidence_grade', 0) or 0))
                values = (name, '社会', age, f"{prob:.1f}", priority, ev_label, rec)
                tree.insert('', 'end', values=values)
                raw_data.append(values)

            # 保存原始数据供筛选使用
            self._results_raw_data = raw_data
            # 数据变更后重置筛选状态
            if hasattr(self, '_results_search_var'):
                self._results_search_var.set('')
            if hasattr(self, '_results_prob_filter_var'):
                self._results_prob_filter_var.set('全部')
        except Exception:
            LOGGER.error("填充结果 Treeview 数据失败", exc_info=True)

    def _filter_results_treeview(self):
        """优先级二：根据搜索框和概率范围筛选结果 Treeview

        搜索框按姓名模糊匹配（不区分大小写），概率范围下拉按数值区间筛选。
        两个条件叠加生效。清除筛选后恢复完整列表。
        """
        tree = getattr(self, '_results_treeview', None)
        if tree is None:
            return

        raw_data = getattr(self, '_results_raw_data', [])
        if not raw_data:
            return

        # 获取筛选条件
        search_query = ''
        if hasattr(self, '_results_search_var'):
            search_query = self._results_search_var.get().strip().lower()

        prob_filter = '全部'
        if hasattr(self, '_results_prob_filter_var'):
            prob_filter = self._results_prob_filter_var.get()

        # 清空 Treeview
        for item in tree.get_children():
            tree.delete(item)

        # 根据条件筛选并重新填充
        for values in raw_data:
            name = str(values[0])
            prob_str = str(values[3])

            # 姓名模糊匹配
            if search_query and search_query not in name.lower():
                continue

            # 概率范围筛选
            if prob_filter != '全部':
                try:
                    prob_val = float(prob_str)
                    if prob_filter == '> 50%' and prob_val <= 50:
                        continue
                    elif prob_filter == '20%-50%' and (prob_val < 20 or prob_val > 50):
                        continue
                    elif prob_filter == '< 20%' and prob_val >= 20:
                        continue
                except (ValueError, TypeError):
                    continue

            tree.insert('', 'end', values=values)

    def _sort_treeview_column(self, tree, col, descending):
        """Treeview 列排序回调

        点击列头时按该列排序，再次点击切换升序/降序。
        - 姓名/类型/建议：按字符串自然顺序排序
        - 年龄：按数值排序
        - 发病概率：按数值排序（列值为 "12.3" 格式字符串）
        - 优先级：按中文优先级映射（极高>高>中>低）排序

        Args:
            tree: ttk.Treeview 实例
            col (str): 列标识符
            descending (bool): 是否降序
        """
        try:
            data = [(tree.set(child, col), child) for child in tree.get_children('')]

            if col == 'probability':
                # 发病概率列存储为 "12.3" 格式字符串，按浮点数排序
                def _key(item):
                    try:
                        return float(item[0])
                    except (ValueError, TypeError):
                        return -1.0
            elif col == 'age':
                # 年龄按整数排序（可能为 'N/A'）
                def _key(item):
                    try:
                        return int(item[0])
                    except (ValueError, TypeError):
                        return -1
            elif col == 'priority':
                # 优先级按中文映射排序
                def _key(item):
                    return self._PRIORITY_ORDER.get(item[0], 0)
            elif col == 'evidence':
                # 证据等级按等级数值排序（GeneXpert > 涂片 > 影像 > 症状推断）
                def _key(item):
                    rev = {v: k for k, v in _EVIDENCE_SHORT_LABEL.items()}
                    return rev.get(item[0], -1)
            else:
                # 姓名/类型/建议按字符串自然顺序
                def _key(item):
                    return item[0]

            data.sort(key=_key, reverse=descending)

            for idx, (_val, child) in enumerate(data):
                tree.move(child, '', idx)

            # 切换排序方向（下次点击同一列时反向）
            tree.heading(col,
                         command=lambda c=col: self._sort_treeview_column(tree, c, not descending))
        except Exception:
            LOGGER.error("Treeview 列排序失败 (col=%s)", col, exc_info=True)

    # ==================== Section IX: results_stale 横幅 ====================

    def _show_stale_banner(self):
        """显示"数据已修改，请重新评估"横幅

        横幅容器在 init_ui.py 中创建（``self._stale_banner_frame``），
        初始隐藏。本方法仅负责显示横幅，不修改 ``results_stale`` 标记
        （标记的设置由 ``_mark_results_stale`` 负责，清除由
        ``_run_assessment`` 完成时负责）。
        """
        try:
            banner = getattr(self, '_stale_banner_frame', None)
            if banner is None:
                # 容器尚未创建，无法显示
                return
            banner.pack(fill='x', padx=10, pady=(5, 0))
        except Exception:
            LOGGER.error("显示 stale 横幅失败", exc_info=True)

    def _hide_stale_banner(self):
        """隐藏"数据已修改，请重新评估"横幅

        仅隐藏横幅的视觉显示，不重置 ``results_stale`` 标记。
        用户通过 × 按钮关闭横幅后，数据仍是 stale 状态，导出操作
        仍会检测到 ``results_stale=True`` 并提示。标记的清除只能
        由 ``_run_assessment`` 完成时执行。
        """
        try:
            banner = getattr(self, '_stale_banner_frame', None)
            if banner is None:
                return
            banner.pack_forget()
        except Exception:
            LOGGER.error("隐藏 stale 横幅失败", exc_info=True)

    def _mark_results_stale(self):
        """标记结果为 stale（被导入管线等调用）

        当数据导入、表单清空等操作使当前评估结果失效时调用。
        会设置 ``results_stale`` 标记并显示 stale 横幅，
        后续导出操作会检测该标记并提示用户先重新评估。
        """
        try:
            self.results_stale = True
            self._show_stale_banner()
        except Exception:
            LOGGER.error("标记 results_stale 失败", exc_info=True)

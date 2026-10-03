#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试：构建模型证据 13 个子标签页并在真实 mainloop 下触发后台证据渲染。

注意：必须用 root.mainloop()（与生产一致），后台线程的 root.after 才能生效；
手动 update() 循环会触发 'main thread is not in main loop'。
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tkinter as tk
from tkinter import ttk

import tb_risk.gui.results_panel as rp
from tb_risk.gui.results_panel.model_evidence import ModelEvidenceMixin

assert any(c.__name__ == 'ModelEvidenceMixin' for c in rp.ResultsMixin.__mro__), \
    "ModelEvidenceMixin 未接入 ResultsMixin"
print("[OK] ModelEvidenceMixin 已在 ResultsMixin MRO 中")


class Host(ModelEvidenceMixin):
    def __init__(self, root):
        self.root = root
        self._closing = False
        self.results = {}
        self.ml_results = None
        self.ml_results_lock = threading.Lock()
        self.adapter = None
        self.COLORS = {
            'accent': '#0f766e', 'chart_series_3': '#45B7D1',
            'bg_card': '#ffffff', 'text': '#2c3e50',
        }
        self._text_widgets = []
        self._tree_widgets = []
        self._threads = []

    def _get_chart_figsize(self, w=8, h=3, **kw):
        return (w, h)

    def _bind_configure_redraw(self, widget, redraw_func=None, chart_name='default',
                               figure=None, base_height=None):
        pass

    def _register_text_widget(self, w):
        self._text_widgets.append(w)
        return w

    def _register_tree_widget(self, w):
        self._tree_widgets.append(w)
        return w

    def _start_training_thread(self, target, daemon=True):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        self._threads.append(t)
        return t


root = tk.Tk()
root.withdraw()
host = Host(root)
nb = ttk.Notebook(root)
nb.pack(fill='both', expand=True)

host._init_model_evidence_tabs(nb)
n_tabs = nb.index('end')
tab_names = [nb.tab(i, 'text') for i in range(n_tabs)]
print(f"[OK] 标签页创建完成，共 {n_tabs} 个子标签页: {tab_names}")
assert n_tabs == 13, "应有 13 个模型证据子标签页"
assert "阴性发现" in tab_names, "第 11 个子标签页（阴性发现）未创建"
assert "组粒度指南" in tab_names, "第 12 个子标签页（组粒度指南）未创建"
assert "部署度量" in tab_names, "第 13 个子标签页（部署度量）未创建"

state = {'done': False, 'error': None}


def poll():
    # 后台证据就绪后渲染并校验
    cache = host._me_cache
    if cache.get('tasks') is not None and cache.get('seir') is not None \
            and cache.get('granularity') is not None \
            and cache.get('deployment') is not None:
        try:
            host._on_static_evidence_ready(cache)
            for key in ('tasks', 'ablation', 'public', 'dgp', 'seir',
                        'negative', 'granularity', 'deployment'):
                entry = cache.get(key) or {}
                assert entry.get('available'), \
                    f"{key} 证据不可用: {entry.get('reason')}"
                print(f"[OK] {key} 证据就绪")
            host._draw_three_layer_waterfall()
            print("[OK] 三层递进占位渲染正常（无 ML 结果时）")
            from tb_risk.core import model_evidence as me
            series = me.extract_three_layer_series(None)
            assert not series.get('available')
            print("[OK] extract_three_layer_series(None) 防御性返回不可用")
            print("[ALL PASS] 模型证据面板冒烟测试通过")
        except Exception as e:  # noqa: BLE001
            state['error'] = e
        state['done'] = True
        root.after(50, root.destroy)
        return
    root.after(300, poll)


# 90 秒超时保护
root.after(90000, lambda: (print("[TIMEOUT] 后台证据未就绪"), root.destroy()))
root.after(300, poll)

try:
    root.mainloop()
except Exception:
    pass

if state['error'] is not None:
    raise state['error']
if not state['done']:
    raise AssertionError("冒烟测试超时未完成")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练档案迁移：data/training_archive → ~/.tb_risk（追加式合并，不覆盖历史）

用法（需在沙箱放行 ~/.tb_risk 或由用户手动执行）：
    python data/migrate_training_archive.py
"""
import json
import os
import shutil

ARCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_archive")
DEST = os.path.join(os.path.expanduser("~"), ".tb_risk")


def migrate():
    os.makedirs(DEST, exist_ok=True)
    report = []

    src_log = os.path.join(ARCH, "training_log.jsonl")
    dst_log = os.path.join(DEST, "training_log.jsonl")
    if os.path.exists(src_log):
        existing = set()
        if os.path.exists(dst_log):
            with open(dst_log, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            existing.add(json.loads(line).get("run_id"))
                        except json.JSONDecodeError:
                            pass
        added = 0
        with open(src_log, encoding="utf-8") as f, open(dst_log, "a", encoding="utf-8") as out:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rid = json.loads(line).get("run_id")
                except json.JSONDecodeError:
                    continue
                if rid not in existing:
                    out.write(line + "\n")
                    existing.add(rid)
                    added += 1
        report.append(f"training_log.jsonl: 追加 {added} 条（按 run_id 去重）")

    src_best = os.path.join(ARCH, "best.json")
    dst_best = os.path.join(DEST, "best.json")
    if os.path.exists(src_best):
        if not os.path.exists(dst_best):
            shutil.copy2(src_best, dst_best)
            report.append("best.json: 拷入（目标原不存在）")
        else:
            with open(src_best, encoding="utf-8") as f:
                s = json.load(f)
            with open(dst_best, encoding="utf-8") as f:
                d = json.load(f)
            s_metric = (s.get("metrics") or {}).get("AUROC", 0)
            d_metric = (d.get("metrics") or {}).get("AUROC", 0)
            if s_metric > d_metric:
                shutil.copy2(src_best, dst_best)
                report.append(f"best.json: 采用回退档案版本（AUROC {s_metric:.3f} > {d_metric:.3f}）")
            else:
                report.append(f"best.json: 保留现有版本（AUROC {d_metric:.3f} ≥ {s_metric:.3f}）")

    for sub in ["results", "models"]:
        src_dir = os.path.join(ARCH, sub)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(DEST, sub)
        os.makedirs(dst_dir, exist_ok=True)
        copied = 0
        for fn in os.listdir(src_dir):
            dst_fp = os.path.join(dst_dir, fn)
            if not os.path.exists(dst_fp):
                shutil.copy2(os.path.join(src_dir, fn), dst_fp)
                copied += 1
        report.append(f"{sub}/: 拷入 {copied} 个文件（不覆盖已有）")

    print("== 迁移报告 ==")
    for r_ in report:
        print("  ", r_)
    if os.path.exists(dst_log):
        with open(dst_log, encoding="utf-8") as f:
            print("   ~/.tb_risk/training_log.jsonl 总条数:",
                  sum(1 for l in f if l.strip()))


if __name__ == "__main__":
    migrate()

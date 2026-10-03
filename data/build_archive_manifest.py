#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归档 JSON 哈希清单（TRIPOD+AI 条目 18f，代码/归档共享配套）

产出：
1. processed/archive_manifest.sha256 —— 标准格式（<sha256>  <filename>），
   可用 `sha256sum -c archive_manifest.sha256` 校验
2. processed/archive_manifest.json —— 机器可读（含 size/date/来源脚本）

范围：data/processed/*.json（全部结论归档）。manifest 自身与 CSV 输出
不纳入（避免自引用）。每次归档更新后重跑本脚本刷新清单；稿件投稿时
随 GitHub 公开/Zenodo 存档一起发布，实现"第三方可校验"。

用法：
    python data/build_archive_manifest.py
"""
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "processed")
DOCS = os.path.join(os.path.dirname(HERE), "docs")


def sha256_of(path, buf_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    files = sorted(
        f for f in os.listdir(PROC)
        if f.endswith(".json")
        and f not in ("archive_manifest.json",))
    entries = []
    for fn in files:
        p = os.path.join(PROC, fn)
        st = os.stat(p)
        entries.append({
            "file": fn,
            "sha256": sha256_of(p),
            "size_bytes": st.st_size,
            "mtime": datetime.datetime.fromtimestamp(
                st.st_mtime).isoformat(timespec="seconds"),
        })

    # 标准格式清单（sha256sum -c 可校验）
    txt_path = os.path.join(PROC, "archive_manifest.sha256")
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['file']}\n")

    # 机器可读版
    js = {
        "generated_at": datetime.datetime.now().isoformat(
            timespec="seconds"),
        "algorithm": "sha256",
        "n_files": len(entries),
        "verify_cmd": "sha256sum -c archive_manifest.sha256",
        "files": entries,
    }
    js_path = os.path.join(PROC, "archive_manifest.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=1)

    print(f"[OK] {txt_path} ({len(entries)} 个归档)")
    print(f"[OK] {js_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tier-1 真实 IPD 数据集探测 + 下载（2026-08-25 第五轮）。

目标（global_ipd_scan_20260825.json tier_1 中尚未到位的五个）：
  1. Barcelona BCG 接触者纵向队列（Dryad dd7h4）——唯一免审批含前瞻性
     活动性 TB 进展结局的数据集（任务 B 外部验证的关键）
  2. Uganda Kampala 接触者 QFT-Plus（Dryad k3j9kd5bg）——此前文件下载
     401，本轮用 API v2 files 通道重试
  3. India NFHS-5（rchiips.org）——约 220 万个体，宿主通路大规模训练
  4. Brazil SINAN-TB（ftp.datasus.gov.br）——132 万通报个案
  5. TREATS（LSHTM Data Compass）——此前 CSV 401，复核状态

用法（从 Desktop cwd 运行，防 tb_risk 命名空间遮蔽）：
    python tb_risk/data/probe_download_tier1.py --probe   # 只探测
    python tb_risk/data/probe_download_tier1.py --download barcelona
"""
import argparse
import datetime
import ftplib
import hashlib
import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) tb-risk-research/1.0"}
TIMEOUT = 60

DRYAD_API = "https://datadryad.org/api/v2"
DRYAD_SETS = {
    "barcelona": "doi:10.5061/dryad.dd7h4",
    "uganda": "doi:10.5061/dryad.k3j9kd5bg",
}

NFHS_PAGE = "http://rchiips.org/nfhs/dllinfo.shtml"

SINAN_FTP = "ftp.datasus.gov.br"
SINAN_DIR = "/dissemin/publicos/SINAN/DADOS/FINAIS"

TREATS_DOI = "https://doi.org/10.17037/DATA.00003627"


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- Dryad
def dryad_probe(name, doi):
    """Dryad API v2：元数据 + 文件列表（含真实下载链接）。"""
    out = {"name": name, "doi": doi, "ok": False}
    r = requests.get(
        DRYAD_API + "/datasets/" + requests.utils.quote(doi, safe=""),
        headers=UA, timeout=TIMEOUT)
    out["meta_status"] = r.status_code
    if r.status_code != 200:
        out["error"] = r.text[:300]
        return out
    meta = r.json()
    out["title"] = meta.get("title", "")
    out["ok"] = True
    # 文件列表：优先走元数据 _links 给出的端点
    links = meta.get("_links") or {}
    files_href = (links.get("stash:files") or {}).get("href")
    if not files_href:
        files_href = DRYAD_API + "/datasets/%s/files" % requests.utils.quote(doi, safe="")
    if files_href.startswith("/"):
        files_href = "https://datadryad.org" + files_href
    fl = requests.get(files_href, headers=UA, timeout=TIMEOUT)
    out["files_status"] = fl.status_code
    out["files_href"] = files_href
    if fl.status_code != 200:
        out["files_error"] = fl.text[:200]
        out["files"] = []
        return out
    try:
        payload = fl.json()
    except ValueError:
        out["files_error"] = "non-json: " + fl.text[:200]
        out["files"] = []
        return out
    files = payload.get("_embedded", {}).get("stash:files", [])
    out["files"] = [
        {"path": f.get("path"), "size": f.get("size"),
         "id": f.get("id"),
         "download": ((f.get("_links") or {}).get("stash:versions:download") or {}).get("href")}
        for f in files]
    return out


def dryad_download(name, doi, dest_dir):
    info = dryad_probe(name, doi)
    if not info.get("ok"):
        return info
    os.makedirs(dest_dir, exist_ok=True)
    for f in info.get("files", []):
        fname = os.path.basename(f["path"] or "file")
        dest = os.path.join(dest_dir, fname)
        url = DRYAD_API + "/files/%s/download" % f["id"]
        print("[%s] 下载 %s (%.1f kB) ..." % (name, fname, (f["size"] or 0) / 1024.0))
        with requests.get(url, headers=UA, timeout=300, stream=True) as r:
            if r.status_code != 200:
                print("  失败 HTTP %s: %s" % (r.status_code, r.text[:200]))
                f["download_status"] = r.status_code
                continue
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
        f["downloaded"] = dest
        f["sha256"] = sha256_of(dest)
        print("  OK -> %s (%d bytes)" % (dest, os.path.getsize(dest)))
    return info


# ---------------------------------------------------------------- NFHS-5
def nfhs_probe():
    """NFHS-5 数据下载页链接盘点。"""
    out = {"url": NFHS_PAGE, "ok": False}
    try:
        r = requests.get(NFHS_PAGE, headers=UA, timeout=TIMEOUT)
        out["status"] = r.status_code
        if r.status_code != 200:
            return out
        out["ok"] = True
        hrefs = sorted(set(re.findall(r'href="([^"]+)"', r.text, re.I)))
        out["links"] = [h for h in hrefs if any(
            k in h.lower() for k in ('.zip', '.dta', '.sav', '.csv', 'data', 'dll'))][:80]
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


# ---------------------------------------------------------------- SINAN
def sinan_probe():
    """巴西 SINAN FTP：列出 TUBE*.dbc 文件与大小。"""
    out = {"ftp": SINAN_FTP, "dir": SINAN_DIR, "ok": False}
    try:
        ftp = ftplib.FTP(SINAN_FTP, timeout=60)
        ftp.login()
        entries = []
        ftp.cwd(SINAN_DIR)
        ftp.retrlines("LIST", lambda line: entries.append(line))
        tubes = [l for l in entries if "TUBE" in l.upper()]
        out["ok"] = True
        out["n_entries"] = len(entries)
        out["tube_files"] = tubes[:40]
        ftp.quit()
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


# ---------------------------------------------------------------- TREATS
def treats_probe():
    """LSHTM Data Compass TREATS DOI 复核。"""
    out = {"doi": TREATS_DOI, "ok": False}
    try:
        r = requests.get(TREATS_DOI, headers=UA, timeout=TIMEOUT, allow_redirects=True)
        out["status"] = r.status_code
        out["final_url"] = r.url
        if r.status_code == 200:
            out["ok"] = True
            out["mentions_request_access"] = bool(
                re.search(r"request|access|restricted", r.text, re.I))
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--download", default=None,
                    help="dryad 数据集名：barcelona / uganda")
    args = ap.parse_args()

    if args.download:
        name = args.download
        if name not in DRYAD_SETS:
            print("未知数据集: %s" % name)
            sys.exit(2)
        dest = os.path.join(RAW, "barcelona_bcg" if name == "barcelona" else "uganda_qft")
        info = dryad_download(name, DRYAD_SETS[name], dest)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    report = {
        "date": datetime.date.today().isoformat(),
        "purpose": "tier-1 IPD 探测（第五轮：大规模真实数据训练与外部验证准备）",
    }
    report["barcelona_dryad"] = dryad_probe("barcelona", DRYAD_SETS["barcelona"])
    report["uganda_dryad"] = dryad_probe("uganda", DRYAD_SETS["uganda"])
    report["nfhs5"] = nfhs_probe()
    report["sinan_ftp"] = sinan_probe()
    report["treats_lshtm"] = treats_probe()

    out_path = os.path.join(RAW, "tier1_probe_20260826.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:6000])
    print("\n已保存: %s" % out_path)


if __name__ == "__main__":
    main()

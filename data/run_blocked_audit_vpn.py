#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""封锁数据集定向审计·VPN 复测轮（2026-09-06 晚）

背景：用户接入 VPN 后出口 IP 变化，重测 2026-09-06 白天审计
（blocked_audit_20260906.json：1 到手 / 9 封锁）中仍封锁的 9 个。
预探（本脚本运行前实测）：
- CDC Socrata download 通道 200 octet-stream → 解封
- Dataverse S3 重定向 200 xlsx → 解封
- figshare CDN 仍 403 / zenodo API 403（代理出口仍被 WAF 拒）
  / Dryad downloads 200 但 text/html（Anubis JS 挑战）
- socket.getaddrinfo("zenodo.org") 失败但 requests 可连——VPN 设置了
  系统代理，requests 走代理（代理端 DNS），socket 直连不走 →
  本轮 DNS 检查改用 HTTP 可达性而非 socket 解析

设计（与 run_blocked_audit.py 同构）：
- 成功 → raw/ 对应目录 + manifest 登记（downloaded 追加 +
  unavailable 清除，与 download_public_datasets.record 同构）+ sha256；
- 失败 → 精确记录 HTTP 状态/错误类型；
- 归档独立文件（不覆盖白天审计）：blocked_audit_20260906_vpn.json。

用法：
    python data/run_blocked_audit_vpn.py
"""
import datetime
import hashlib
import json
import os

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
PROC = os.path.join(HERE, "processed")
MANIFEST = os.path.join(RAW, "_manifest.json")
OUT = os.path.join(PROC, "blocked_audit_20260906_vpn.json")

TIMEOUT = 90
UA_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
UA_PLAIN = {"User-Agent": "python-requests tb-risk-audit/1.0"}

TARGETS = [
    # (tier, relpath, display, channels)
    ("T1", "figshare/uganda_household_igra.xlsx",
     "乌干达家庭接触者 IGRA（Muchuro 2022, n=355, 户ID+暴露梯度）",
     [("figshare-ndownloader-browser",
       "https://figshare.com/ndownloader/files/36177084", UA_BROWSER),
      ("figshare-ndownloader-cdn",
       "https://ndownloader.figshare.com/files/36177084", UA_BROWSER)]),
    ("T1", "dryad/uganda_qftplus_contacts.xlsx",
     "乌干达坎帕拉成人接触者 QFT-Plus（Chunda/Mayito 2022, n=202）",
     [("dryad-api-file-download",
       "https://datadryad.org/api/v2/files/1883732/download", UA_BROWSER),
      ("dryad-classic-downloads",
       "https://datadryad.org/downloads/1883732", UA_BROWSER)]),
    ("T2", "zenodo/spain_qftgit_contacts_5y.sav",
     "西班牙 BCG 成人接触者 QFT-GIT + 5 年随访（n=661, CC0，SOP 第四国候选）",
     [("zenodo-api-record",
       "https://zenodo.org/api/records/4931404", UA_BROWSER)]),
    ("T3", "cdc/tbesc2_partA_tst_igra.zip",
     "CDC TBESC-II Part A：TST vs IGRA 头对头 + 进展预测（~2,121 接触者）",
     [("cdc-download-zip",
       "https://data.cdc.gov/download/5hpj-p74g/"
       "application/x-zip-compressed", UA_BROWSER),
      ("cdc-soda-api-csv",
       "https://data.cdc.gov/resource/5hpj-p74g.csv?$limit=500000",
       UA_BROWSER)]),
    ("T3", "dataverse/korea_hcw_tst_igra.xlsx",
     "韩国 458 名医护人员 TST/IGRA 串行检测（Park 2018, 2009-2013）",
     [("dataverse-api",
       "https://dataverse.harvard.edu/api/access/datafile/3210719",
       UA_BROWSER)]),
    ("T4", "dataverse/carabayllo_contact_evaluation.tab",
     "秘鲁 Carabayllo 家庭接触者评估级联（Yuen 2019, CC0）",
     [("dataverse-api",
       "https://dataverse.harvard.edu/api/access/datafile/3425482",
       UA_BROWSER)]),
    ("T5", "figshare/german_students_dual_igra.sav",
     "德国留学生 QFT-GIT vs QFT-Plus 双平台配对（n=134）",
     [("figshare-ndownloader-browser",
       "https://figshare.com/ndownloader/files/6989846", UA_BROWSER),
      ("figshare-ndownloader-cdn",
       "https://ndownloader.figshare.com/files/6989846", UA_BROWSER)]),
    ("T5", "zenodo/lima_treatment_default.xls",
     "秘鲁利马涂阳患者治疗中断队列（n=1,233, 127 中断, CC0）",
     [("zenodo-api-record",
       "https://zenodo.org/api/records/4992464", UA_BROWSER)]),
    ("T5", "cdc/tbesc2_partB_ltbi_cascade.zip",
     "CDC TBESC-II Part B：LTBI 预防级联监测",
     [("cdc-download-zip",
       "https://data.cdc.gov/download/epap-ayij/"
       "application/x-zip-compressed", UA_BROWSER),
      ("cdc-soda-api-csv",
       "https://data.cdc.gov/resource/epap-ayij.csv?$limit=500000",
       UA_BROWSER)]),
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def try_channels(session, channels):
    attempts = []
    for name, url, ua in channels:
        try:
            r = session.get(url, headers=ua, timeout=TIMEOUT,
                            allow_redirects=True)
            ctype = r.headers.get("Content-Type", "")
            attempts.append({
                "channel": name, "url": url,
                "http_status": r.status_code,
                "final_url": r.url[:160],
                "content_type": ctype[:60],
                "bytes": len(r.content),
            })
            if (r.status_code == 200 and len(r.content) > 1000
                    and "text/html" not in ctype.lower()):
                return r.content, name, attempts
            if r.status_code == 200 and "text/html" in ctype.lower():
                attempts[-1]["note"] = "HTML 挑战页（Anubis/WAF），非文件"
        except Exception as e:  # noqa: BLE001
            attempts.append({"channel": name,
                             "error": f"{type(e).__name__}: {e}"[:200]})
    return None, None, attempts


def save_and_record(relpath, content, url, channel, note, license_):
    fpath = os.path.join(RAW, relpath.replace("/", os.sep))
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as f:
        f.write(content)
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    m["downloaded"][relpath] = {
        "url": url, "license": license_, "note": note,
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
        "downloaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "via_channel": channel,
    }
    m["unavailable"].pop(relpath, None)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    return fpath, m["downloaded"][relpath]["sha256"]


def main():
    session = requests.Session()
    results = []
    for tier, relpath, display, channels in TARGETS:
        print("[%s] %s" % (tier, relpath))
        content, used, attempts = try_channels(session, channels)
        if content is not None:
            license_ = ("CC0 1.0 (Zenodo)" if relpath.startswith("zenodo/")
                        else "CC BY 4.0 (figshare)" if relpath.startswith("figshare/")
                        else "Public Domain (data.cdc.gov)" if relpath.startswith("cdc/")
                        else "CC0 1.0 (Harvard Dataverse)" if relpath.startswith("dataverse/")
                        else "CC BY 4.0 (Dryad)")
            note = display + "（定向审计 VPN 复测轮 2026-09-06 通道解封）"
            fpath, sha = save_and_record(relpath, content,
                                         attempts[-1].get("url", ""),
                                         used, note, license_)
            results.append({
                "tier": tier, "relpath": relpath, "display": display,
                "outcome": "downloaded", "channel": used,
                "path": fpath, "sha256": sha, "size_bytes": len(content),
                "attempts": attempts,
            })
            print("      [OK] via %s（%.0fKB, sha256 %s…）"
                  % (used, len(content) / 1024, sha[:12]))
        else:
            results.append({
                "tier": tier, "relpath": relpath, "display": display,
                "outcome": "still_blocked", "attempts": attempts,
            })
            codes = [str(a.get("http_status") or a.get("error", "?")[:40])
                     for a in attempts]
            print("      [仍封锁] %s" % codes)

    n_ok = sum(1 for r in results if r["outcome"] == "downloaded")
    summary = {
        "date": "2026-09-06",
        "purpose": ("封锁数据集定向审计·VPN 复测轮（用户接入 VPN 后出口"
                    "IP 变化，重测白天仍封锁的 9 个；白天审计见 "
                    "blocked_audit_20260906.json，本文件为独立归档不覆盖）"),
        "vpn_probe_note": ("预探实测：CDC download 200 octet-stream、"
                           "Dataverse S3 200 xlsx（解封）；figshare 403、"
                           "zenodo API 403、Dryad 200 HTML 挑战页"
                           "（仍拒）。socket.getaddrinfo(zenodo) 失败但 "
                           "requests 可达——VPN 系统代理：requests 走代理"
                           "（代理端 DNS），socket 直连不走"),
        "n_target": len(results), "n_downloaded": n_ok,
        "n_still_blocked": len(results) - n_ok,
        "results": results,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== VPN 复测汇总 ===== 目标 %d | 成功 %d | 仍封锁 %d"
          % (len(results), n_ok, len(results) - n_ok))
    print("归档 → %s" % OUT)


if __name__ == "__main__":
    main()

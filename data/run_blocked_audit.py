#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""封锁数据集定向审计（2026-09-06，P5 重排：感染终点家庭前瞻队列优先）

背景：2026-09-05 全球搜寻实测 6 个仓库通道对本机出口 IP 封锁
（figshare CDN 403 / PLOS S1→googleapis 不可达 / Dryad downloads 403
/ zenodo DNS 污染 / data.cdc.gov Socrata 403 / Dataverse S3 域名
间歇可达），11 个高价值数据集入 BLOCKED_MANUAL。当晚 20:07 br760
浏览器通道曾成功（peru_mdr_household_incident.xlsx 已入手并完成
§8.6 分析）。

本审计（距封锁约一天，WAF 速率型封锁可能已冷却）：
- 全部 10 个仍未入手数据集逐通道多策略复测（每数据集 2-3 条
  备选通道，普通请求 + 浏览器 UA 变体）；
- 成功 → 存入 raw/ 对应目录、登记 manifest（downloaded +
  清除 unavailable，与 download_public_datasets.record 同构）、
  计算 sha256；
- 失败 → 精确记录 HTTP 状态码/错误类型，判定封锁形态是否变化；
- 报告按 P5 优先级分层（感染终点家庭接触者 > 双终点大样本 >
  感染终点其他接触者 > 级联/进展 > 非接触者队列）。

P5 分层（用户 2026-09-06 指令：感染终点家庭前瞻队列优先）：
  T1 感染终点·家庭接触者（SOP 机制直接可测——户结构+IGRA）：
     figshare/uganda_household_igra（n=355，户ID+暴露梯度+IGRA）
     dryad/uganda_qftplus_contacts（n=202，QFT-Plus）
  T2 感染+病程双终点·家庭接触者（前瞻）：
     figshare/peru_aibana_contacts（14,044，TST 转阳 1,787 + 继发 TB 406）
     zenodo/spain_qftgit_contacts_5y（n=661，QFT-GIT + 5 年随访——SOP 第四国候选）
  T3 感染终点·接触者（非严格家庭）：
     cdc/tbesc2_partA_tst_igra（~2,121，TST vs IGRA 头对头 + 进展）
     dataverse/korea_hcw_tst_igra（458，串行 TST/IGRA + LTBI 治疗后发病）
  T4 级联/进展·家庭接触者：
     dataverse/carabayllo_contact_evaluation（评估级联）
  T5 非接触者/非感染终点：
     figshare/german_students_dual_igra（134，双平台配对）
     zenodo/lima_treatment_default（1,233，治疗中断）
     cdc/tbesc2_partB_ltbi_cascade（LTBI 级联监测）

用法：
    python data/run_blocked_audit.py
输出：
    data/processed/blocked_audit_20260906.json
    成功文件 → data/raw/{figshare,dryad,zenodo,cdc,dataverse}/...
    manifest 更新（downloaded 追加 + unavailable 清除）
"""
import datetime
import hashlib
import json
import os
import socket
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
PROC = os.path.join(HERE, "processed")
MANIFEST = os.path.join(RAW, "_manifest.json")
OUT = os.path.join(PROC, "blocked_audit_20260906.json")

TIMEOUT = 45
UA_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
UA_PLAIN = {"User-Agent": "python-requests tb-risk-audit/1.0"}

# Dryad 乌干达 QFT-Plus：元数据 API 已知（file_id 1883732，
# sha-256 前缀 da66ca…，57,198 B）
DRYAD_UGANDA_FILE_ID = 1883732
DRYAD_UGANDA_DOI = "10.5061/dryad.k3j9kd5bg"

TARGETS = [
    # (tier, relpath, display, channels)
    ("T1", "figshare/uganda_household_igra.xlsx",
     "乌干达家庭接触者 IGRA（Muchuro 2022, n=355, 户ID+暴露梯度）",
     [("figshare-ndownloader-browser",
       "https://figshare.com/ndownloader/files/36177084", UA_BROWSER),
      ("figshare-ndownloader-cdn",
       "https://ndownloader.figshare.com/files/36177084", UA_BROWSER),
      ("figshare-plain-ua",
       "https://figshare.com/ndownloader/files/36177084", UA_PLAIN)]),
    ("T1", "dryad/uganda_qftplus_contacts.xlsx",
     "乌干达坎帕拉成人接触者 QFT-Plus（Chunda/Mayito 2022, n=202）",
     [("dryad-api-file-download",
       f"https://datadryad.org/api/v2/files/{DRYAD_UGANDA_FILE_ID}/download",
       UA_BROWSER),
      ("dryad-classic-downloads",
       f"https://datadryad.org/downloads/{DRYAD_UGANDA_FILE_ID}",
       UA_BROWSER)]),
    ("T2", "figshare/peru_aibana_contacts.dta",
     "秘鲁利马 14,044 家庭接触者营养-进展队列（Aibana 2016，TST 转阳+继发 TB 双终点）",
     [("plos-s1-doi",
       "https://doi.org/10.1371/journal.pone.0166333.s001", UA_BROWSER),
      ("plos-journals-file",
       "https://journals.plos.org/plosone/article/file"
       "?type=supplementary&id=10.1371/journal.pone.0166333.s001",
       UA_BROWSER)]),
    ("T2", "zenodo/spain_qftgit_contacts_5y.sav",
     "西班牙 BCG 成人接触者 QFT-GIT + 5 年随访（n=661, CC0，SOP 第四国候选）",
     []),  # 通道动态生成（先 DNS 检查）
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
       UA_BROWSER),
      ("dataverse-retry-2",
       "https://dataverse.harvard.edu/api/access/datafile/3210719",
       UA_PLAIN)]),
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
     []),
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


def check_zenodo_dns():
    """zenodo DNS 污染检测：真实 Zenodo 为 CERN 段（188.184.x 等）；
    污染特征 = 199.59.149.x（Twitter/Facebook 段）或解析失败。"""
    try:
        infos = socket.getaddrinfo("zenodo.org", 443, socket.AF_INET)
        ips = sorted({i[4][0] for i in infos})
        polluted = any(ip.startswith("199.59.") or ip.startswith("31.13.")
                       for ip in ips)
        return {"resolved": ips, "dns_polluted": polluted}
    except Exception as e:
        return {"resolved": [], "dns_polluted": True, "error": str(e)[:120]}


def try_channels(session, relpath, channels):
    """逐通道尝试；返回 (content, used_channel, attempts) 或 (None, None, attempts)。"""
    attempts = []
    for name, url, ua in channels:
        try:
            r = session.get(url, headers=ua, timeout=TIMEOUT,
                            allow_redirects=True)
            attempts.append({
                "channel": name, "url": url,
                "http_status": r.status_code,
                "final_url": r.url[:160],
                "content_type": r.headers.get("Content-Type", "")[:60],
                "bytes": len(r.content),
            })
            ctype = r.headers.get("Content-Type", "")
            # 成功判据：200 且非 HTML 挑战页（Dryad/WAF 挑战返回 text/html）
            if (r.status_code == 200 and len(r.content) > 1000
                    and "text/html" not in ctype.lower()):
                return r.content, name, attempts
            if r.status_code == 200 and "text/html" in ctype.lower():
                attempts[-1]["note"] = "HTML 挑战页（Anubis/WAF），非文件"
        except requests.exceptions.HTTPError as e:
            attempts.append({"channel": name, "error": f"HTTPError: {e}"[:200]})
        except Exception as e:
            attempts.append({"channel": name,
                             "error": f"{type(e).__name__}: {e}"[:200]})
    return None, None, attempts


def save_and_record(relpath, content, url, channel, note, license_):
    """存入 raw/ 并更新 manifest（与 download_public_datasets.record 同构）。"""
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
    dns = check_zenodo_dns()

    for tier, relpath, display, channels in TARGETS:
        print(f"\n[{tier}] {relpath}")
        print(f"      {display}")

        # zenodo 数据集：DNS 检查后动态生成通道
        if relpath.startswith("zenodo/") and not channels:
            if dns["dns_polluted"]:
                results.append({
                    "tier": tier, "relpath": relpath, "display": display,
                    "outcome": "dns_polluted_needs_vpn",
                    "dns_check": dns,
                    "attempts": [],
                    "manual_url": ("https://zenodo.org/records/4931404"
                                   if "spain" in relpath
                                   else "https://zenodo.org/records/4992464"),
                })
                print("      [DNS 污染持续] 需用户 VPN，跳过 HTTP 尝试")
                continue
            # DNS 正常：先 API 取文件名再下载
            rec_id = "4931404" if "spain" in relpath else "4992464"
            channels = [("zenodo-api-record",
                         f"https://zenodo.org/api/records/{rec_id}",
                         UA_BROWSER)]

        content, used, attempts = try_channels(session, relpath, channels)

        if content is not None:
            license_ = ("CC0 1.0 (Zenodo)" if relpath.startswith("zenodo/")
                        else "CC BY 4.0 (figshare)" if relpath.startswith("figshare/")
                        else "Public Domain (data.cdc.gov)" if relpath.startswith("cdc/")
                        else "CC0 1.0 (Harvard Dataverse)" if relpath.startswith("dataverse/")
                        else "CC BY 4.0 (Dryad)")
            note = display + "（定向审计 2026-09-06 通道解封）"
            fpath, sha = save_and_record(relpath, content,
                                         attempts and attempts[-1].get("url", ""),
                                         used, note, license_)
            results.append({
                "tier": tier, "relpath": relpath, "display": display,
                "outcome": "downloaded", "channel": used,
                "path": fpath, "sha256": sha, "size_bytes": len(content),
                "attempts": attempts,
            })
            print(f"      [OK] via {used}（{len(content)/1024:.0f}KB, "
                  f"sha256 {sha[:12]}…）")
        else:
            verdict = "still_blocked_manual"
            if relpath.startswith("zenodo/"):
                verdict = "dns_polluted_needs_vpn" if dns["dns_polluted"] \
                    else "still_blocked_manual"
            results.append({
                "tier": tier, "relpath": relpath, "display": display,
                "outcome": verdict, "attempts": attempts,
            })
            codes = [str(a.get("http_status") or a.get("error", "?")[:40])
                     for a in attempts]
            print(f"      [仍封锁] 通道结果：{codes}")

    # ---- 汇总归档 ----
    n_ok = sum(1 for r in results if r["outcome"] == "downloaded")
    summary = {
        "date": "2026-09-06",
        "purpose": ("封锁数据集定向审计（P5：感染终点家庭前瞻队列优先）；"
                    "距 2026-09-05 封锁约一天后通道复测"),
        "priority_note": ("P5 分层：T1 感染终点家庭接触者（SOP 直接可测）> "
                          "T2 双终点家庭前瞻 > T3 感染终点其他接触者 > "
                          "T4 级联/进展 > T5 非接触者队列"),
        "dryad_metadata_recheck": None,
        "n_target": len(results), "n_downloaded": n_ok,
        "n_still_blocked": sum(1 for r in results
                               if r["outcome"] == "still_blocked_manual"),
        "n_needs_vpn": sum(1 for r in results
                           if r["outcome"] == "dns_polluted_needs_vpn"),
        "results": results,
    }

    # Dryad 元数据 API 复检（乌干达 QFT-Plus：验证文件 sha/大小未变）
    try:
        r = session.get(
            "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.k3j9kd5bg",
            headers=UA_BROWSER, timeout=TIMEOUT)
        if r.status_code == 200:
            doc = r.json()
            ver = doc.get("_embedded", {}).get(
                "stash:versions", [{}])[0] if "_embedded" in doc else {}
            summary["dryad_metadata_recheck"] = {
                "http_status": r.status_code,
                "note": "Dryad 元数据 API 仍 200（封锁仅 downloads 层）",
                "title": doc.get("title", "")[:120],
            }
    except Exception as e:
        summary["dryad_metadata_recheck"] = {"error": str(e)[:150]}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n===== 审计汇总 =====")
    print(f"目标 {len(results)} 个 | 成功 {n_ok} | "
          f"仍封锁 {summary['n_still_blocked']} | 需 VPN "
          f"{summary['n_needs_vpn']}")
    if n_ok:
        print("成功清单（已入 raw/ + manifest）：")
        for r in results:
            if r["outcome"] == "downloaded":
                print(f"  [{r['tier']}] {r['relpath']} ({r['channel']})")
    print(f"归档 → {OUT}")


if __name__ == "__main__":
    main()

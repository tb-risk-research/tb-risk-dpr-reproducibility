#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_risk 主入口模块
提供命令行接口、应用程序启动功能和自动依赖安装
"""
import sys
import os
import subprocess

# 确保包路径在 sys.path 中
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

# 可选依赖及其 pip 包名映射
_OPTIONAL_DEP_MAP = {
    "numpy":           {"package": "numpy",            "group": "科学计算", "min_version": "1.20.0"},
    "scipy":           {"package": "scipy",            "group": "科学计算", "min_version": "1.7.0"},
    "matplotlib":      {"package": "matplotlib",       "group": "可视化",   "min_version": "3.4.0"},
    "seaborn":         {"package": "seaborn",          "group": "可视化",   "min_version": "0.11.0"},
    "sklearn":         {"package": "scikit-learn",     "group": "机器学习", "min_version": "1.0.0"},
    "xgboost":         {"package": "xgboost",          "group": "机器学习", "min_version": "1.5.0"},
    "shap":            {"package": "shap",             "group": "机器学习", "min_version": "0.40.0"},
    "joblib":          {"package": "joblib",           "group": "机器学习", "min_version": "1.1.0"},
    "torch":           {"package": "torch",            "group": "图神经网络", "min_version": "1.10.0"},
    "torch_geometric": {"package": "torch-geometric",  "group": "图神经网络", "min_version": "2.0.0"},
    "openpyxl":        {"package": "openpyxl",         "group": "导出",     "min_version": "3.0.0"},
    "reportlab":       {"package": "reportlab",        "group": "导出",     "min_version": "3.6.0"},
    # AI 助手依赖（云端 LLM 接入）
    "httpx":           {"package": "httpx",            "group": "AI",       "min_version": "0.24.0"},
    "tenacity":        {"package": "tenacity",         "group": "AI",       "min_version": "8.0.0"},
    # AI 本地部署依赖（敏感场景：数据不出本机）
    "transformers":    {"package": "transformers",     "group": "AI 本地",  "min_version": "4.30.0"},
}


def _check_dependencies():
    """检查关键依赖是否可用，返回状态字典

    统一使用 _OPTIONAL_DEP_MAP 驱动检查逻辑，避免重复定义。
    tkinter 等标准库也做实际导入检测，避免在最小化环境中误判。
    """
    deps = {}

    # 核心标准库（json/csv/math 始终可用；tkinter 在无图形环境中可能缺失）
    deps["json"] = True
    deps["csv"] = True
    deps["math"] = True
    try:
        import tkinter
        deps["tkinter"] = True
    except ImportError:
        deps["tkinter"] = False

    # 可选依赖：统一从 _OPTIONAL_DEP_MAP 遍历检查
    for mod_name in _OPTIONAL_DEP_MAP:
        try:
            __import__(mod_name)
            deps[mod_name] = True
        except ImportError:
            deps[mod_name] = False

    return deps


def get_missing_deps():
    """获取缺失的可选依赖列表

    Returns:
        list[dict]: 缺失依赖信息，每项包含 name, package, group, min_version
    """
    deps = _check_dependencies()
    missing = []
    for name, info in _OPTIONAL_DEP_MAP.items():
        if not deps.get(name, False):
            missing.append({
                "name": name,
                "package": info["package"],
                "group": info["group"],
                "min_version": info["min_version"],
            })
    return missing


def install_missing_deps(missing=None, confirm=True):
    """安装缺失的可选依赖

    Args:
        missing: 缺失依赖列表，None 则自动检测
        confirm: 是否在安装前确认

    Returns:
        bool: 安装是否成功
    """
    if missing is None:
        missing = get_missing_deps()

    if not missing:
        print("✓ 所有可选依赖已安装")
        return True

    print(f"\n检测到 {len(missing)} 个缺失的可选依赖：\n")
    groups = {}
    for m in missing:
        groups.setdefault(m["group"], []).append(m)

    for group, items in groups.items():
        pkgs = [f"  {m['package']} (>={m['min_version']})" for m in items]
        print(f"  [{group}]")
        for p in pkgs:
            print(p)
        print()

    if confirm:
        try:
            answer = input("是否安装这些依赖？[Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            return False
        if answer and answer not in ("y", "yes", "是"):
            print("已跳过安装。可通过 --install 参数手动安装。")
            return False

    packages = [f"{m['package']}>={m['min_version']}" for m in missing]
    print(f"\n正在安装 {len(packages)} 个包...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade"] + packages,
            capture_output=False,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("\n✓ 所有依赖安装成功")
            return True
        else:
            print(f"\n✗ 安装失败，退出码: {result.returncode}", file=sys.stderr)
            print("  请手动运行: pip install " + " ".join(packages), file=sys.stderr)
            return False
    except Exception as e:
        print(f"\n✗ 安装过程出错: {e}", file=sys.stderr)
        print("  请手动运行: pip install " + " ".join(packages), file=sys.stderr)
        return False


def _auto_ensure_dependencies():
    """启动时自动检测并提示安装缺失依赖

    仅在交互式终端中自动提示；非交互模式下静默跳过。
    """
    if not sys.stdin.isatty():
        return  # 非交互模式，静默跳过

    missing = get_missing_deps()
    if not missing:
        return  # 全部已安装

    print("\n" + "=" * 60)
    print("  tb_risk - 依赖检测")
    print("=" * 60)
    print(f"\n检测到 {len(missing)} 个缺失的可选依赖，部分功能可能受限。")

    groups = {}
    for m in missing:
        groups.setdefault(m["group"], []).append(m["package"])

    for group, pkgs in groups.items():
        print(f"  [{group}]: {', '.join(pkgs)}")

    print("\n运行 'tb-risk --install' 可自动安装所有缺失依赖")
    print("运行 'tb-risk --check' 查看完整依赖状态")

    # 问题三：GNN 环境分级提示与安装指引
    try:
        from tb_risk.ml.gnn import gnn_env_status, gnn_install_guidance
        st = gnn_env_status()
        print("\n  [GNN 环境] 当前级别{} ({})".format(st['level'], st['name']))
        if st['install_command']:
            print("    升级指引: {}".format(st['install_command']))
            print("    或运行: pip install -e '.[gnn]'  （一键安装 torch + torch-geometric）")
    except ImportError:
        pass

    print("=" * 60 + "\n")


def print_dependency_status():
    """打印依赖状态

    动态从 _OPTIONAL_DEP_MAP 生成分组，避免硬编码重复。
    """
    deps = _check_dependencies()
    print("\n" + "=" * 60)
    print("  tb_risk 依赖状态")
    print("=" * 60)

    # 核心库
    core_libs = ["json", "csv", "math", "tkinter"]
    print(f"\n  [核心库]")
    for name in core_libs:
        status = "✓" if deps.get(name, False) else "✗"
        print(f"    {status} {name}")

    # 从 _OPTIONAL_DEP_MAP 动态构建分组
    groups = {}
    for mod_name, info in _OPTIONAL_DEP_MAP.items():
        group = info["group"]
        groups.setdefault(group, []).append(mod_name)

    # 按定义顺序输出分组
    group_order = []
    seen_groups = set()
    for info in _OPTIONAL_DEP_MAP.values():
        g = info["group"]
        if g not in seen_groups:
            seen_groups.add(g)
            group_order.append(g)

    for cat in group_order:
        names = groups.get(cat, [])
        print(f"\n  [{cat}]")
        for name in names:
            status = "✓" if deps.get(name, False) else "✗"
            print(f"    {status} {name}")

    print("\n" + "=" * 60)

    # 功能可用性总结
    can_ml = deps["numpy"] and deps["sklearn"]
    can_gnn = deps["torch"] and deps["torch_geometric"]
    can_viz = deps["matplotlib"]
    can_export = deps["openpyxl"] or deps["reportlab"]
    can_ai_cloud = deps.get("httpx", False) and deps.get("tenacity", False)
    can_ai_local = deps.get("transformers", False)

    print("\n  功能可用性：")
    print("    基础评估: ✓ (始终可用)")
    print(f"    ML风险预测: {'✓' if can_ml else '✗ (需要 numpy + scikit-learn)'}")
    print(f"    GNN分析: {'✓' if can_gnn else '✗ (需要 torch + torch-geometric)'}")
    print(f"    可视化: {'✓' if can_viz else '✗ (需要 matplotlib)'}")
    print(f"    数据导出: {'✓' if can_export else '✗ (需要 openpyxl 或 reportlab)'}")
    print(f"    AI助手(云端): {'✓' if can_ai_cloud else '✗ (需要 httpx + tenacity)'}")
    print(f"    AI助手(本地): {'✓' if can_ai_local else '✗ (需要 transformers)'}")
    print("=" * 60 + "\n")


def launch_gui():
    """启动GUI应用程序"""
    try:
        from tb_risk.assessment import TB_Risk_Assessment

        # TB_Risk_Assessment.__init__ 不接受 root 参数；GUI 初始化由 init_gui() 完成。
        # 修复前：root = tk.Tk(); app = TB_Risk_Assessment(root) 会抛出 TypeError。
        app = TB_Risk_Assessment()
        app.init_gui()
        app.root.mainloop()
    except ImportError as e:
        print(f"错误：无法导入主模块 - {e}", file=sys.stderr)
        print("请确保已安装 tb_risk 包: pip install -e .", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}", file=sys.stderr)
        sys.exit(1)


def run_etl_pipeline(config_path: str = "", input_file: str = "",
                      output_dir: str = "etl_output", template: str = "",
                      source: str = "csv", patient_id_field: str = "patient_id",
                      batch_size: int = 100, verbose: bool = False):
    """运行 ETL 数据映射与特征计算管线

    从原始医院数据（CSV/JSON）到 tb_risk 标准特征的完整转换。

    Args:
        config_path: JSON 配置文件路径（可选）
        input_file: 输入数据文件路径（CSV 或 JSON）
        output_dir: 输出目录
        template: 映射模板文件路径
        source: 数据来源标识
        patient_id_field: 患者ID字段名
        batch_size: 批处理大小
        verbose: 是否输出详细信息
    """
    import json
    import logging
    import os
    import sys
    import time

    from health_interop.etl_pipeline import ETLPipeline

    # ---- 配置日志 ----
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("tb_risk.main.etl")

    # ---- 创建管线 ----
    pipeline = ETLPipeline()
    pipeline.initialize_terminology_mappers()

    # ---- 加载映射模板 ----
    if template:
        count = pipeline.load_mapping_template(template)
        logger.info("从模板 '%s' 加载了 %d 个映射规则", template, count)
    elif config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        mappings = config.get("mappings", [])
        if mappings:
            pipeline.add_mappings(mappings)
            logger.info("从配置文件加载了 %d 个映射规则", len(mappings))
        template_ref = config.get("template")
        if template_ref:
            pipeline.load_mapping_template(template_ref)
    else:
        logger.warning("未指定映射模板或配置文件，仅执行原始数据传递")

    # ---- 设置输出目录 ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- 读取输入数据 ----
    records = []
    if input_file:
        ext = os.path.splitext(input_file)[1].lower()
        if ext == ".json":
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
        elif ext == ".csv":
            _import_csv(input_file, records)
        else:
            logger.error("不支持的文件格式: %s", ext)
            sys.exit(1)
        logger.info("加载了 %d 条患者记录", len(records))
    else:
        # 从标准输入读取 JSON
        try:
            input_str = sys.stdin.read()
            if input_str.strip():
                data = json.loads(input_str)
                records = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, EOFError):
            pass

    # ---- 处理 ----
    if not records:
        logger.info("未提供输入数据，启动交互模式")
        _run_etl_interactive(pipeline, output_dir, source)
        return

    start = time.time()
    results = pipeline.process_batch(
        records, source=source,
        patient_id_field=patient_id_field,
        batch_size=batch_size,
    )

    elapsed = time.time() - start
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # ---- 输出统计 ----
    logger.info(
        "ETL 管线完成: 总 %d 条, 成功 %d, 失败 %d, 耗时 %.2fs",
        len(results), len(success), len(failed), elapsed,
    )

    # ---- 导出特征 ----
    for r in success:
        pid = r.patient_id
        out_path = os.path.join(output_dir, f"{pid}_features.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(r.features, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("导出特征失败 %s: %s", pid, e)

    if verbose:
        for r in success:
            print(f"\n患者 {r.patient_id}:")
            print(f"  特征数: {len(r.features)}")
            if r.warnings:
                for w in r.warnings[:5]:
                    print(f"  警告: {w}")
            print(f"  耗时: {r.processing_time_ms:.0f}ms")

    # ---- 报告 ----
    stats = pipeline.get_pipeline_statistics()
    print(f"\n{'='*50}")
    print(f"ETL 管线运行报告")
    print(f"{'='*50}")
    print(f"  处理患者: {len(results)}")
    print(f"  成功: {len(success)}")
    print(f"  失败: {len(failed)}")
    print(f"  映射规则: {stats['mapping_rules']}")
    print(f"  术语编码: {'已启用' if stats['terminology_initialized'] else '未启用'}")
    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"{'='*50}\n")


def _import_csv(filepath: str, records: list):
    """简易 CSV 导入（避免 pandas 依赖）"""
    import csv
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))


def _run_etl_interactive(pipeline, output_dir: str, source: str):
    """交互式 ETL 处理（单条数据输入）"""
    import json
    print("ETL 交互模式 (输入 JSON 格式数据，空行退出):")
    while True:
        try:
            line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            break
        try:
            raw = json.loads(line)
            pid = raw.get("patient_id", "interactive")
            result = pipeline.process_patient(raw, patient_id=pid, source=source)
            if result.success:
                print(f"  特征: {json.dumps(result.features, ensure_ascii=False, indent=2)}")
                if result.warnings:
                    for w in result.warnings:
                        print(f"  警告: {w}")
            else:
                print(f"  错误: {result.errors}")
        except json.JSONDecodeError as e:
            print(f"  JSON 解析错误: {e}")


def run_cli_assessment(config_path: str):
    """从命令行运行评估（无GUI模式，委托给 click CLI）

    Args:
        config_path: JSON配置文件路径
    """
    import json
    from tb_risk.core import RiskAssessmentService

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    patient_info = config.get('patient_info', {})
    family_members = config.get('family_members', [])
    social_contacts = config.get('social_contacts', [])
    use_ml = config.get('use_ml', False)
    use_seir = config.get('use_seir', False)

    service = RiskAssessmentService()
    result = service.assess(
        patient_info=patient_info,
        family_members=family_members,
        social_contacts=social_contacts,
        use_ml=use_ml,
        use_seir=use_seir,
    )

    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    print(f"\n潜在患者数: family={len(result['potential_patients']['family'])}, "
          f"social={len(result['potential_patients']['social'])}")


def main():
    """主入口函数

    优先使用 click CLI（v5.0+），回退到 argparse（向后兼容）。
    """
    # 尝试使用 click CLI（v5.0+ 新接口）
    try:
        from tb_risk.cli.commands import cli
        # 检查是否传入了子命令参数
        if len(sys.argv) > 1:
            cli(prog_name='tb-risk')
        else:
            # 无参数：启动前自动检测缺失依赖，然后启动GUI
            _auto_ensure_dependencies()
            launch_gui()
        return
    except ImportError:
        pass  # click 未安装，回退到 argparse

    # ====== 向后兼容: argparse 接口（v4.x） ======
    import argparse
    import warnings

    parser = argparse.ArgumentParser(
        description="结核病风险评估工具 - 家庭社会关系网络分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  tb-risk                      # 启动GUI应用程序（自动检测缺失依赖）
  tb-risk assess config.json   # 命令行评估（v5.0+，需安装 click）
  tb-risk check                # 检查依赖状态并退出
  tb-risk --install            # 自动安装所有缺失的可选依赖
  tb-risk --install -y         # 自动安装（无需确认）
  tb-risk --cli config.json    # 命令行模式（已弃用，请使用 'tb-risk assess'）
        """
    )

    parser.add_argument(
        "--check", action="store_true",
        help="检查依赖状态并退出"
    )
    parser.add_argument(
        "--install", action="store_true",
        help="自动检测并安装缺失的可选依赖"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="配合 --install 使用，跳过确认提示"
    )
    parser.add_argument(
        "--cli", metavar="CONFIG",
        help="命令行模式（已弃用，请使用 'tb-risk assess CONFIG'）"
    )
    parser.add_argument(
        "--version", action="store_true",
        help="显示版本信息"
    )
    parser.add_argument(
        "--etl", nargs="?", const="interactive", metavar="CONFIG",
        help="运行 ETL 数据映射管线。可选参数: CONFIG (JSON 配置文件路径)"
    )
    parser.add_argument(
        "--etl-input", metavar="FILE",
        help="ETL 输入数据文件 (CSV 或 JSON)"
    )
    parser.add_argument(
        "--etl-output", metavar="DIR", default="etl_output",
        help="ETL 输出目录 (默认: etl_output)"
    )
    parser.add_argument(
        "--etl-template", metavar="FILE",
        help="ETL 映射模板文件路径"
    )
    parser.add_argument(
        "--etl-source", metavar="NAME", default="csv",
        help="ETL 数据来源标识 (默认: csv)"
    )
    parser.add_argument(
        "--etl-verbose", action="store_true",
        help="ETL 详细输出模式"
    )

    args = parser.parse_args()

    if args.version:
        from tb_risk import __version__, __author__
        print(f"tb_risk v{__version__} by {__author__}")
        return

    if args.etl:
        run_etl_pipeline(
            config_path=args.etl if args.etl != "interactive" else "",
            input_file=args.etl_input,
            output_dir=args.etl_output,
            template=args.etl_template,
            source=args.etl_source,
            verbose=args.etl_verbose,
        )
        return

    if args.check:
        print_dependency_status()
        return

    if args.install:
        install_missing_deps(confirm=not args.yes)
        return

    if args.cli:
        warnings.warn(
            "'--cli' 参数已弃用，请使用 'tb-risk assess CONFIG'（需安装 click）",
            DeprecationWarning
        )
        run_cli_assessment(args.cli)
        return

    # 默认：启动前自动检测缺失依赖
    _auto_ensure_dependencies()

    # 启动GUI
    launch_gui()


if __name__ == "__main__":
    main()
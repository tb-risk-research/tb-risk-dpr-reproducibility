#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tb_risk CLI 子命令定义

基于 click 的命令行评估管线。
子命令体系：
- assess:  单次评估
- batch:   批量评估
- calibrate: 贝叶斯校准
- simulate: SEIR 模拟
- train:   ML 模型训练
- export:  结果导出
- check:   依赖检查
- config:  配置管理
- ai:      AI 助手（文本抽取/报告生成/数据质量诊断）
"""

import json
import os
import sys

import click

from .formatters import output_result


@click.group()
@click.version_option(version='5.0.0', prog_name='tb-risk')
@click.pass_context
def cli(ctx):
    """结核病风险评估工具 - 命令行接口

    基于 SEIR 传播动力学模型和机器学习算法的结核病风险评估系统。
    支持单次评估、批量评估、SEIR模拟、ML模型训练等功能。
    """
    ctx.ensure_object(dict)


# ==================== assess 子命令 ====================

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--format', '-f', 'output_format', type=click.Choice(['json', 'csv', 'table']),
              default='json', help='输出格式')
@click.option('--output', '-o', type=click.Path(), help='输出文件路径')
@click.option('--ml', is_flag=True, help='启用 ML 预测')
@click.option('--seir', is_flag=True, help='启用 SEIR 模拟')
@click.option('--karamay', is_flag=True, help='启用克拉玛依本土化')
@click.pass_context
def assess(ctx, config_file, output_format, output, ml, seir, karamay):
    """单次结核病风险评估

    CONFIG_FILE: JSON/YAML 配置文件路径，包含患者信息和接触者数据。
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"错误: 配置文件 JSON 格式无效 - {e}", err=True)
        sys.exit(1)
    except FileNotFoundError:
        click.echo(f"错误: 配置文件不存在 - {config_file}", err=True)
        sys.exit(1)

    from tb_risk.core import RiskAssessmentService

    # 问题十-3：localizer 创建统一由 service 层负责（create_default_localizer）
    # CLI 不再显式 new KaramayLocalizer()，避免与 GUI 的 get_instance() 单例行为不一致
    service = RiskAssessmentService(enable_karamay=karamay)

    if karamay and service.get_localizer() is not None:
        click.echo("克拉玛依本土化已启用")
    elif karamay:
        click.echo("警告: 克拉玛依本土化模块不可用，使用默认参数", err=True)

    # 提取数据（兼容两种 key 命名）
    patient_info = config.get('patient_info', config.get('patient', {}))
    family_members = config.get('family_members', [])
    social_contacts = config.get('social_contacts', [])

    click.echo(f"评估配置: {config_file}")
    click.echo(f"家庭成员: {len(family_members)} 人, 社会接触者: {len(social_contacts)} 人")

    # 执行评估
    result = service.assess(
        patient_info=patient_info,
        family_members=family_members,
        social_contacts=social_contacts,
        use_ml=ml,
        use_seir=seir,
    )

    # 合成数据警示（CLI 下让用户明确感知演示模式）
    if result.get('synthetic_data_warning'):
        click.echo(
            click.style(
                "[WARNING] 当前ML模型基于合成数据训练，结果仅供演示参考，"
                "不代表真实流行病学预测能力。",
                fg='yellow',
            ),
            err=True,
        )

    # 输出结果
    output_result(result, fmt=output_format, output_file=output)


# ==================== batch 子命令 ====================

@cli.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--format', '-f', 'output_format', type=click.Choice(['json', 'csv', 'table']),
              default='json', help='输出格式')
@click.option('--output', '-o', type=click.Path(), help='输出文件路径')
@click.option('--karamay', is_flag=True, help='启用克拉玛依本土化')
@click.pass_context
def batch(ctx, data_dir, output_format, output, karamay):
    """批量评估目录中的 JSON 配置文件

    DATA_DIR: 包含 JSON 配置文件的目录路径。
    """
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not json_files:
        click.echo(f"错误: 目录 {data_dir} 中没有 JSON 文件", err=True)
        sys.exit(1)

    click.echo(f"找到 {len(json_files)} 个配置文件\n")

    from tb_risk.core import RiskAssessmentService

    # 问题十-3：localizer 创建统一由 service 层负责
    service = RiskAssessmentService(enable_karamay=karamay)
    all_results = []

    for i, json_file in enumerate(sorted(json_files), 1):
        filepath = os.path.join(data_dir, json_file)
        click.echo(f"[{i}/{len(json_files)}] {json_file}...")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)

            result = service.assess(
                patient_info=config.get('patient', {}),
                family_members=config.get('family_members', []),
                social_contacts=config.get('social_contacts', []),
            )

            all_results.append({
                'file': json_file,
                'summary': result['summary'],
            })
        except Exception as e:
            click.echo(f"  错误: {e}", err=True)
            all_results.append({
                'file': json_file,
                'error': str(e),
            })

    # 输出汇总
    summary = {
        'total_files': len(json_files),
        'total_contacts': sum(
            r.get('summary', {}).get('total_contacts', 0)
            for r in all_results if 'summary' in r
        ),
        'results': all_results,
    }

    output_result(summary, fmt=output_format, output_file=output)


# ==================== simulate 子命令 ====================

@cli.command()
@click.option('--population', '-n', type=int, default=1000, help='模拟人口数')
@click.option('--days', '-d', type=int, default=365, help='模拟天数')
@click.option('--beta', type=float, default=0.3, help='传播率')
@click.option('--trajectories', '-t', type=int, default=10, help='轨迹数量')
@click.option('--format', '-f', 'output_format', type=click.Choice(['json', 'csv', 'table']),
              default='json', help='输出格式')
@click.option('--output', '-o', type=click.Path(), help='输出文件路径')
@click.pass_context
def simulate(ctx, population, days, beta, trajectories, output_format, output):
    """SEIR 传播动力学模拟

    运行随机 SEIR 模型模拟结核病在人群中的传播。
    """
    # 问题十-1：通过 RiskAssessmentService 门面调用，避免界面层直接
    # 导入 StochasticSEIRModel 实现类（越层访问）
    from tb_risk.core import RiskAssessmentService

    click.echo(f"SEIR 模拟: 人口={population}, 天数={days}, β={beta}, 轨迹={trajectories}")

    try:
        service = RiskAssessmentService()
        service.simulate_seir(
            population=population,
            days=days,
            beta=beta,
            trajectories=trajectories,
        )
    except ImportError as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    output_result({'simulation_complete': True, 'parameters': {
        'population': population, 'days': days, 'beta': beta,
        'trajectories': trajectories,
    }}, fmt=output_format, output_file=output)


# ==================== train 子命令 ====================

@cli.command()
@click.option('--samples', '-n', type=int, default=2000, help='训练样本数')
@click.option('--hyperopt', is_flag=True, help='启用超参数优化')
@click.option('--output', '-o', type=click.Path(), help='模型保存路径')
@click.pass_context
def train(ctx, samples, hyperopt, output):
    """训练 ML 风险预测模型"""
    # 问题十-1：通过 RiskAssessmentService 门面调用，避免界面层直接
    # 导入 MLRiskPredictor 实现类（越层访问）
    from tb_risk.core import RiskAssessmentService

    click.echo(f"ML 模型训练: 样本数={samples}, 超参数优化={'是' if hyperopt else '否'}")

    try:
        service = RiskAssessmentService()
        result = service.train_ml_models(
            n_samples=samples,
            enable_hyperopt=hyperopt,
            save_path=output,
        )
    except ImportError as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    click.echo(f"训练完成: {result['model_count']} 个模型已训练")

    if output:
        click.echo(f"模型已保存到: {output}")


@cli.command('train-gnn')
@click.option('--samples', '-n', type=int, default=2000, help='合成图样本数')
@click.option('--gnn-hyperopt', 'gnn_hyperopt', is_flag=True,
              help='启用 GNN 超参搜索（受样本量 ≥ 5000 门槛约束，'
                   '门槛下自动跳过）')
@click.option('--epochs', type=int, default=50, help='训练轮数')
@click.option('--output', '-o', type=click.Path(), help='GNN 模型保存路径')
@click.pass_context
def train_gnn(ctx, samples, gnn_hyperopt, epochs, output):
    """训练 GNN 网络风险模型

    M 级审计修复（CLI 暴露缺口）：--gnn-hyperopt 等 GNN 训练开关
    此前仅 GUI 可达，CLI train 子命令只有 --hyperopt（ML 路径）。
    """
    # 问题十-1：通过 RiskAssessmentService 门面调用，避免界面层直接
    # 导入 GNN 实现类（越层访问）
    from tb_risk.core import RiskAssessmentService

    click.echo(f"GNN 模型训练: 样本数={samples}, "
               f"超参搜索={'是' if gnn_hyperopt else '否'}, 轮数={epochs}")

    try:
        service = RiskAssessmentService()
        result = service.train_gnn_models(
            n_samples=samples,
            enable_hyperopt=gnn_hyperopt,
            n_epochs=epochs,
            save_path=output,
        )
    except ImportError as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    if not result.get('gnn_is_trained'):
        click.echo("GNN 训练未完成（依赖缺失或训练失败，详见日志）", err=True)
        sys.exit(1)

    click.echo("GNN 训练完成")

    if output:
        click.echo(f"GNN 模型已保存到: {output}")


@cli.command('ensemble-weights')
@click.option('--ml-auroc', type=float, help='ML 方向验证集 AUROC (0,1]')
@click.option('--seir-auroc', type=float, help='SEIR 方向验证集 AUROC (0,1]')
@click.option('--gnn-auroc', type=float, help='GNN 方向验证集 AUROC (0,1]')
@click.option('--temperature', type=float, default=1.0,
              help='softmax 温度（→∞ 趋于均匀，→0 趋于 winner-take-all）')
@click.pass_context
def ensemble_weights(ctx, ml_auroc, seir_auroc, gnn_auroc, temperature):
    """三方向集成权重优化（基于验证集 AUROC 的 softmax 加权）

    M 级审计修复（CLI 暴露缺口）：三方向权重优化
    （optimize_weights_from_validation）此前仅程序内可达。

    至少提供一个方向的 AUROC；未提供的方向不参与该次加权。
    """
    from tb_risk.core import RiskAssessmentService

    validation_auroc = {}
    for name, val in (('ml', ml_auroc), ('seir', seir_auroc),
                      ('gnn', gnn_auroc)):
        if val is not None:
            validation_auroc[name] = val

    if not validation_auroc:
        click.echo("错误: 至少提供一个方向 AUROC "
                   "(--ml-auroc/--seir-auroc/--gnn-auroc)", err=True)
        sys.exit(1)

    try:
        service = RiskAssessmentService()
        integrator = service.get_integrator()
        weights = integrator.optimize_weights_from_validation(
            validation_auroc, temperature=temperature)
    except (ImportError, ValueError, TypeError) as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    scenario = {frozenset(('ml', 'seir', 'gnn')): 'three'}.get(
        frozenset(validation_auroc.keys()))
    result = {
        'validation_auroc': validation_auroc,
        'temperature': temperature,
        'optimized_weights': {
            k: ({d: round(float(w), 4) for d, w in v.items()}
                if isinstance(v, dict) else v)
            for k, v in weights.items()
        },
        'primary_scenario': scenario,
    }
    output_result(result, fmt='json')

    if scenario:
        click.echo(f"三方向权重: {result['optimized_weights'][scenario]}")


# ==================== check 子命令 ====================

@cli.command()
@click.pass_context
def check(ctx):
    """检查依赖状态"""
    from tb_risk.main import print_dependency_status
    print_dependency_status()


# ==================== config 子命令 ====================

@cli.group()
def config():
    """配置管理"""
    pass


@config.command('show')
@click.option('--path', '-p', default=None, help='配置路径（点号分隔，如 occupation.oilfield_camp_factor）')
@click.pass_context
def config_show(ctx, path):
    """显示当前配置"""
    from tb_risk.config import _EMBEDDED_KARAMAY_CONFIG
    if path:
        keys = path.split('.')
        node = _EMBEDDED_KARAMAY_CONFIG
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                click.echo(f"配置路径不存在: {path}", err=True)
                return
        click.echo(json.dumps(node, ensure_ascii=False, indent=2))
    else:
        click.echo(json.dumps(_EMBEDDED_KARAMAY_CONFIG, ensure_ascii=False, indent=2))


@config.command('validate')
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def config_validate(ctx, config_file):
    """验证外部配置文件"""
    from tb_risk.config import _validate_config_schema
    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    is_valid, errors = _validate_config_schema(cfg)
    if is_valid:
        click.echo("✓ 配置文件验证通过")
    else:
        click.echo("✗ 配置文件验证失败:")
        for err in errors:
            click.echo(f"  - {err}")
        sys.exit(1)


# ==================== gui 子命令 ====================

@cli.command()
@click.pass_context
def gui(ctx):
    """启动 GUI 应用程序"""
    from tb_risk.main import launch_gui
    launch_gui()


# ==================== api 子命令 ====================

@cli.command()
@click.option('--host', default=None, help='监听地址（默认 0.0.0.0）')
@click.option('--port', '-p', type=int, default=None, help='监听端口（默认 8000）')
@click.option('--config', 'config_file', type=click.Path(exists=True),
              default=None, help='REST API 配置文件路径')
@click.option('--workers', type=int, default=None, help='worker 数')
@click.option('--reload', is_flag=True, default=None,
              help='热重载模式（开发用）')
@click.pass_context
def api(ctx, host, port, config_file, workers, reload_flag):
    """启动 REST API 服务（FastAPI）

    提供单患者/批量评估、模型训练与状态、报告生成及 CDS Hooks 接口。
    需安装可选依赖：pip install 'tb_risk[api]'
    """
    try:
        from tb_risk.api import APIConfig, run_server
    except ImportError:
        click.echo(
            "错误: REST API 模块不可用。请执行：pip install 'tb_risk[api]'",
            err=True)
        sys.exit(1)

    config = APIConfig.load(config_file)
    if host:
        config.host = host
    if port:
        config.port = port
    if workers:
        config.workers = workers

    click.echo(f"REST API 启动中: http://{config.host}:{config.port}")
    click.echo("  文档: /docs  (OpenAPI/Swagger)")
    click.echo("  文档: /redoc (ReDoc)")
    try:
        run_server(config, reload=bool(reload_flag))
    except SystemExit as e:
        sys.exit(e.code)


# ==================== ai 子命令组 ====================

def _load_ai_config(config_file=None, local_model_path=None):
    """加载 AI 配置

    优先级：
    1. --config 指定的文件路径（若提供）
    2. 默认 ~/.tb_risk/ai_config.json（由 from_user_config_file 处理）
    3. 环境变量覆盖已设置的字段（由 from_user_config_file 处理）

    --local 覆盖 local_model_path（启用本地模式，数据不出本机）

    Returns:
        AIConfig
    """
    from tb_risk.ai.config import AIConfig
    cfg = AIConfig.from_user_config_file(config_file)
    if local_model_path:
        cfg.local_model_path = local_model_path
    return cfg


def _print_ai_not_configured_error():
    """AI 未配置或调用失败时打印统一的错误提示"""
    click.echo(
        "错误: AI 助手未配置或调用失败。\n"
        "请通过以下任一方式配置：\n"
        "  1. 设置环境变量 TB_AI_API_KEY（云端模式）或 TB_AI_LOCAL_MODEL（本地模式）\n"
        "  2. 创建 ~/.tb_risk/ai_config.json 配置文件\n"
        "  3. 使用 --config <path> 指定配置文件\n"
        "  4. 使用 --local <model_path> 指定本地模型路径",
        err=True,
    )


@cli.group()
def ai():
    """AI 助手功能（文本抽取 / 报告生成 / 数据质量诊断）

    需配置 AI 服务（环境变量、~/.tb_risk/ai_config.json 或 --config 选项）。
    本地模式（--local）数据不出本机，适合敏感场景。
    """
    pass


@ai.command('extract')
@click.argument('text_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='输出文件路径（默认 stdout）')
@click.option('--config', 'config_file', type=click.Path(exists=True),
              default=None, help='AI 配置文件路径（默认 ~/.tb_risk/ai_config.json）')
@click.option('--local', 'local_model_path', type=str, default=None,
              help='本地模型路径或 HuggingFace 模型名（启用本地模式，数据不出本机）')
def ai_extract(text_file, output, config_file, local_model_path):
    """从临床叙述文本中抽取结构化数据

    TEXT_FILE: 包含临床叙述的文本文件路径
    """
    # 读取文本文件
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError as e:
        click.echo(f"错误: 读取文件失败 - {e}", err=True)
        sys.exit(1)

    if not text.strip():
        click.echo("错误: 文本文件为空", err=True)
        sys.exit(1)

    # 加载 AI 配置
    cfg = _load_ai_config(config_file, local_model_path)

    click.echo(f"AI 文本抽取: {text_file}", err=True)
    if cfg.is_local:
        click.echo(f"模式: 本地模型 ({cfg.local_model_path})", err=True)
    else:
        click.echo(f"模式: 云端 ({cfg.provider}/{cfg.model})", err=True)

    # 调用 AI 抽取
    from tb_risk.ai import extract_with_llm
    records = extract_with_llm(text, config=cfg)

    if records is None:
        _print_ai_not_configured_error()
        sys.exit(1)

    # 输出结果
    output_result(records, fmt='json', output_file=output)


@ai.command('report')
@click.argument('result_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='输出文件路径（默认 stdout）')
@click.option('--config', 'config_file', type=click.Path(exists=True),
              default=None, help='AI 配置文件路径')
@click.option('--local', 'local_model_path', type=str, default=None,
              help='本地模型路径或 HuggingFace 模型名（启用本地模式）')
def ai_report(result_file, output, config_file, local_model_path):
    """根据风险评估结果生成 AI 中文临床报告

    RESULT_FILE: JSON 文件路径，包含 assess() 返回的结果（含 patient_score 等字段）
    """
    # 读取 result JSON
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"错误: JSON 格式无效 - {e}", err=True)
        sys.exit(1)
    except OSError as e:
        click.echo(f"错误: 读取文件失败 - {e}", err=True)
        sys.exit(1)

    if not isinstance(result, dict) or not result:
        click.echo("错误: 结果文件应为非空 JSON 对象", err=True)
        sys.exit(1)

    # 加载 AI 配置
    cfg = _load_ai_config(config_file, local_model_path)

    click.echo(f"AI 报告生成: {result_file}", err=True)
    if cfg.is_local:
        click.echo(f"模式: 本地模型 ({cfg.local_model_path})", err=True)
    else:
        click.echo(f"模式: 云端 ({cfg.provider}/{cfg.model})", err=True)

    # 调用 AI 生成报告
    from tb_risk.ai import generate_report
    report = generate_report(result, config=cfg)

    if report is None:
        _print_ai_not_configured_error()
        sys.exit(1)

    # 输出报告（纯文本/markdown，不走 format_output）
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report)
        click.echo(f"报告已保存到: {output}", err=True)
    else:
        click.echo(report)


@ai.command('diagnose')
@click.argument('record_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='输出文件路径（默认 stdout）')
@click.option('--config', 'config_file', type=click.Path(exists=True),
              default=None, help='AI 配置文件路径')
@click.option('--local', 'local_model_path', type=str, default=None,
              help='本地模型路径或 HuggingFace 模型名（启用本地模式）')
def ai_diagnose(record_file, output, config_file, local_model_path):
    """对单条记录进行 AI 数据质量诊断

    RECORD_FILE: JSON 文件路径，包含患者/接触者记录
    """
    # 读取 record JSON
    try:
        with open(record_file, 'r', encoding='utf-8') as f:
            record = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"错误: JSON 格式无效 - {e}", err=True)
        sys.exit(1)
    except OSError as e:
        click.echo(f"错误: 读取文件失败 - {e}", err=True)
        sys.exit(1)

    if not isinstance(record, dict) or not record:
        click.echo("错误: 记录文件应为非空 JSON 对象", err=True)
        sys.exit(1)

    # 加载 AI 配置
    cfg = _load_ai_config(config_file, local_model_path)

    click.echo(f"AI 数据质量诊断: {record_file}", err=True)
    if cfg.is_local:
        click.echo(f"模式: 本地模型 ({cfg.local_model_path})", err=True)
    else:
        click.echo(f"模式: 云端 ({cfg.provider}/{cfg.model})", err=True)

    # 调用 AI 诊断
    from tb_risk.ai import diagnose_record
    diag = diagnose_record(record, config=cfg)

    if diag is None:
        _print_ai_not_configured_error()
        sys.exit(1)

    # 输出结果
    output_result(diag, fmt='json', output_file=output)


if __name__ == '__main__':
    cli()
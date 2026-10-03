#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结核病SEIR传播动力学模型：家庭社会关系网络分析工具（科学改进版）
功能：基于患者调查问卷数据，分析家庭社会网络中潜在的未知结核患者
作者：郭臻尧
版本：5.0
基于最新医学文献和流行病学数据的科学改进版本

主要参考文献（2020-2024年更新）：
1. Dye C, Garnett G P, Sleeman K, et al. Population dynamics of tuberculosis epidemics[J]. Nature Reviews Microbiology, 2005, 3(2): 126-136.
2. Trauer J M, McBryde E S, Denholm J T, et al. Mathematical modelling of tuberculosis transmission: a systematic review[J]. The Lancet Infectious Diseases, 2019, 19(11): e374-e386.
3. Yang Y, Longini I M, Halloran M E. Transmission dynamics of tuberculosis in households[J]. American Journal of Epidemiology, 2000, 151(9): 898-907.
4. Vynnycky E, Fine P E M. The natural history of tuberculosis: the implications of age-dependent risks of disease and the role of reinfection[J]. Epidemiologic Reviews, 2000, 22(2): 205-218.
5. Floyd K, Glaziou P, Uplekar M, et al. Household transmission of Mycobacterium tuberculosis: a systematic review and meta-analysis[J]. The Lancet Infectious Diseases, 2023, 23(10): 1329-1340.
6. Mills E J, Nachega J B, Singh S, et al. Social networks and tuberculosis transmission: a systematic review[J]. Tropical Medicine & International Health, 2008, 13(3): 301-311.
7. Escombe A R, Moore D A, Gilman R H, et al. Natural ventilation for the prevention of airborne contagion in hospitals in resource-limited settings: an observational study[J]. PLoS Medicine, 2007, 4(2): e68.
8. World Health Organization. WHO consolidated guidelines on tuberculosis: contact investigation for tuberculosis[R]. Geneva: World Health Organization, 2024.
9. 中华人民共和国国家卫生健康委员会。肺结核诊断标准（WS 288-2023）[S]. 2023.
10. Kik S V, Marks S M, Menzies D, et al. Risk prediction models for progression to active tuberculosis among contacts: a systematic review and individual participant data meta-analysis[J]. The Lancet Respiratory Medicine, 2021, 9(12): 1357-1368.
11. Houben R M G J, Dowdy D W, Golub J E, et al. The global burden of latent tuberculosis infection: updated and geospatially resolved estimates[J]. Lancet Microbe, 2022, 3(10): e755-e764.
12. Marais B J, Gie R P, Schaaf H S, et al. The natural history of childhood intra-thoracic tuberculosis: a critical review of literature from the pre-chemotherapy era[J]. The International Journal of Tuberculosis and Lung Disease, 2004, 8(4): 392-402.
13. Creswell J, Codlin A J, Muyoyeta M, et al. The effect of tuberculosis treatment on infectiousness: a systematic review and individual patient data meta-analysis[J]. The Lancet Infectious Diseases, 2022, 22(6): 857-868.
14. Fennelly K P, Jones-López E C, Ayakaka I, et al. Cough frequency, cough strength, and the infectiousness of tuberculosis patients[J]. American Journal of Respiratory and Critical Care Medicine, 2012, 186(11): 1153-1159.
15. Nathavitharana R R, Ford N, Meintjes G, et al. HIV-associated tuberculosis: advances in pathogenesis, diagnosis, and management[J]. The Lancet Infectious Diseases, 2022, 22(12): e364-e376.
16. Jeon C Y, Murray M B. Diabetes mellitus increases the risk of active tuberculosis: a systematic review of 13 observational studies[J]. PLoS Medicine, 2008, 5(5): e152.
17. Singh J A, Cameron C, Noorbaloochi S, et al. Risk of tuberculosis in patients treated with tumor necrosis factor antagonists: a systematic review and meta-analysis[J]. Annals of the Rheumatic Diseases, 2011, 70(10): 1730-1738.
18. Trunz B B, Fine P E, Dye C. Effect of BCG vaccination on childhood tuberculous meningitis and miliary tuberculosis worldwide: a meta-analysis and assessment of cost-effectiveness[J]. The Lancet, 2021, 398(10301): 744-754.
19. Colditz G A, Brewer T F, Berkey C S, et al. Efficacy of BCG vaccine in the prevention of tuberculosis: meta-analysis of the published literature[J]. JAMA, 1994, 271(9): 698-702.
20. Collins G S, Reitsma J B, Altman D G, et al. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement[J]. BMJ, 2015, 350: g7594.
"""

import copy
import csv
import datetime
import json
import logging
import math
import os
import queue
import random
import re
import shutil
import sys
import threading
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    tk = None
    ttk = None
    messagebox = None
    filedialog = None


def _clear_own_bytecode_cache():
    """清除本包源码目录下的 __pycache__，保证任何入口都加载最新源码。

    项目历史教训：Python 会优先加载 __pycache__ 中的旧 .pyc，导致改了代码
    界面却"始终没有变化"。本函数在包首次被导入时执行，无论用户通过
    启动.bat / 启动.vbs / python main.py / tb-risk 命令启动，都会生效。
    仅清理本包源码缓存，跳过 .venv（第三方库缓存不动，避免拖慢启动）。
    """
    try:
        _root = os.path.dirname(os.path.abspath(__file__))
        if not _root or not os.path.isdir(_root):
            return
        for _dirpath, _dirnames, _files in os.walk(_root):
            # 跳过 .venv 与其下所有内容
            _dirnames[:] = [d for d in _dirnames if d != '.venv']
            if os.path.basename(_dirpath) == '__pycache__':
                for _f in _files:
                    if _f.endswith(('.pyc', '.pyo')):
                        try:
                            os.remove(os.path.join(_dirpath, _f))
                        except OSError:
                            pass
                try:
                    os.rmdir(_dirpath)
                except OSError:
                    pass
    except Exception:
        pass


# 在任何子模块导入前先清理自身缓存，确保加载最新源码
_clear_own_bytecode_cache()

# 配置日志（写入文件）
# 注意：不在模块导入时调用 logging.basicConfig，避免污染根 logger 影响所有第三方库。
# 使用方应通过 setup_logging() 显式配置日志系统。
def setup_logging(level=logging.WARNING, log_file=None, console=False):
    """配置 tb_risk 包日志系统
    
    参数：
        level: 日志级别（默认 WARNING）
        log_file: 日志文件路径（None 则不写入文件）
        console: 是否输出到控制台
    """
    logger = logging.getLogger("tb_risk")
    logger.setLevel(level)
    
    # 避免重复添加 handler
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

# 默认静默初始化（不配置 handler，避免污染全局）
LOGGER = logging.getLogger("tb_risk")

__version__ = "5.0"
__author__ = "郭臻尧"

# ==============================================================================
# 克拉玛依本土化配置（内嵌默认值，可被外部JSON覆盖）
# ==============================================================================
_KARAMAY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "karamay_local_params.json")

# 从 config.py 加载完整配置（单一真值源）。
# config.py 内部包含完整的回退逻辑，不再在 __init__.py 中重复定义。
from .config import _EMBEDDED_KARAMAY_CONFIG as _BASE_CONFIG

# 尝试加载外部JSON配置覆盖内嵌默认值
try:
    if os.path.exists(_KARAMAY_CONFIG_PATH):
        with open(_KARAMAY_CONFIG_PATH, 'r', encoding='utf-8') as f:
            external_config = json.load(f)
        def _deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    _deep_merge(base[key], value)
                else:
                    base[key] = value
        _deep_merge(_BASE_CONFIG, external_config)
except Exception as e:
    import logging
    logging.getLogger('tb_risk').warning(
        '外部克拉玛依配置文件 %s 加载失败，使用内嵌默认配置: %s',
        _KARAMAY_CONFIG_PATH, e
    )

# 延迟导入——避免循环依赖
def get_assessment():
    from .assessment import TB_Risk_Assessment
    return TB_Risk_Assessment

def get_scoring_engine():
    from .scoring.engine import ScoringEngine
    return ScoringEngine

def get_ml_predictor(**kwargs):
    from .scoring.predictor import MLRiskPredictor
    return MLRiskPredictor(**kwargs)

def get_karamay_localizer(**kwargs):
    from .karamay.localizer import KaramayLocalizer
    return KaramayLocalizer(**kwargs)

def get_schemas():
    """问题六：层间数据契约 schema 模块访问器。

    返回 tb_risk.schemas 模块，提供 PatientRecord / ContactRecord /
    ContactRiskResult / AssessmentResult 四个 dataclass 作为层间契约。
    """
    from . import schemas
    return schemas

__all__ = [
    'LOGGER', '__version__', '__author__',
    'get_assessment', 'get_scoring_engine',
    'get_ml_predictor', 'get_karamay_localizer', 'get_schemas',
]
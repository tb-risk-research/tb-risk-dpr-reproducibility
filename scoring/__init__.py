from .simulator import ScreeningDataSimulator
from .engine import ScoringEngine
from .architecture import (
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_GATE_MIN,
    DEFAULT_GATE_MAX,
    DEFAULT_DECISION_WEIGHTS,
    INTERVENTION_BENEFIT_GAIN,
    compute_individual_base,
    compute_network_increment,
    select_high_risk_set,
    simulate_community_intervention,
    gated_integration,
    ThreeLayerArchitecture,
    run_three_layer_pipeline,
)

__all__ = [
    'ScreeningDataSimulator', 'ScoringEngine',
    # 三层递进架构（个体基础层 → 网络增强层 → 社区干预层）
    'DEFAULT_HIGH_RISK_THRESHOLD',
    'DEFAULT_GATE_MIN',
    'DEFAULT_GATE_MAX',
    'DEFAULT_DECISION_WEIGHTS',
    'INTERVENTION_BENEFIT_GAIN',
    'compute_individual_base',
    'compute_network_increment',
    'select_high_risk_set',
    'simulate_community_intervention',
    'gated_integration',
    'ThreeLayerArchitecture',
    'run_three_layer_pipeline',
]
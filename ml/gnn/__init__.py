"""异质时变图神经网络模块

包含 _GNNFrameworkPlaceholder, HeteroGATLayer, HeteroTimeVaryingGNN, MultimodalGNN

参考文献：
  Zheng Y et al. (2024) HeatGNN; He Q et al. (2025 AAAI) HHAN
  Han S et al. (2025) CSTGNN; Velickovic P et al. (2018) GAT
"""

# SEIR ODE 求解器
from ._seir_ode import (
    seir_ode_step,
    seir_ode_step_v3,
    seir_euler_integrate,
    seir_rk4_integrate,
    seir_v3_euler_integrate,
    seir_v3_rk4_integrate,
    seir_ode_gating,
    seir_solver_error_baseline,
    _IDX_S, _IDX_LF, _IDX_LS, _IDX_M,
    _IDX_ISUB, _IDX_ISP, _IDX_ISN, _IDX_R, _IDX_C,
    _N_COMPARTMENTS_V3,
)

# 异质图注意力层
from ._layers import HeteroGATLayer

# 异质时变图神经网络
from .hetero_gnn import HeteroTimeVaryingGNN

# 多模态特征融合 GNN
from .multimodal import MultimodalGNN

# 导入守卫与基类（向后兼容：原 ml/gnn.py 模块级常量）
from ._base import (
    PYTORCH_AVAILABLE,
    PYG_AVAILABLE,
    _GNN_BASE_CLASS,
)

# 深度学习环境分级检测（问题三：无 torch/PyG 时的降级指引）
from ._env import (
    NUMPY_AVAILABLE,
    gnn_degradation_level,
    gnn_degradation_name,
    gnn_tier_description,
    gnn_install_guidance,
    gnn_env_status,
    gnn_env_status_text,
)

# 纯 NumPy 轻量图卷积回退（问题三：无深度学习环境时保证 GNN 分支有输出）
from ._numpy_gnn import (
    NumPyGraphRiskPredictor,
    feature_vector as numpy_feature_vector,
    predict_risk_numpy,
)

__all__ = [
    # SEIR ODE
    'seir_ode_step',
    'seir_ode_step_v3',
    'seir_euler_integrate',
    'seir_rk4_integrate',
    'seir_v3_euler_integrate',
    'seir_v3_rk4_integrate',
    'seir_ode_gating',
    'seir_solver_error_baseline',
    '_IDX_S', '_IDX_LF', '_IDX_LS', '_IDX_M',
    '_IDX_ISUB', '_IDX_ISP', '_IDX_ISN', '_IDX_R', '_IDX_C',
    '_N_COMPARTMENTS_V3',
    # GNN 层与模型
    'HeteroGATLayer',
    'HeteroTimeVaryingGNN',
    'MultimodalGNN',
    # 模块级常量
    'PYTORCH_AVAILABLE',
    'PYG_AVAILABLE',
    '_GNN_BASE_CLASS',
    # GNN 环境分级（问题三）
    'NUMPY_AVAILABLE',
    'gnn_degradation_level',
    'gnn_degradation_name',
    'gnn_tier_description',
    'gnn_install_guidance',
    'gnn_env_status',
    'gnn_env_status_text',
    # NumPy 轻量图卷积回退（问题三）
    'NumPyGraphRiskPredictor',
    'numpy_feature_vector',
    'predict_risk_numpy',
]

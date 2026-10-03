#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section X 全局体验层 — 独立帮助窗口

提供 ``HelpWindow`` 类与内置 ``HELP_CONTENT`` 字典：
- 左侧目录树导航 + 右侧 ScrolledText 显示内容
- 全文搜索（搜索框 + 高亮匹配）
- 防止重复打开（已打开则聚焦）

帮助文本直接内置为 Python 字典，不依赖外部 Markdown 文件。
所有方法均使用 try/except 防护，避免子模块异常影响主窗口。
"""

from ._imports import *  # noqa: F401,F403 — 共享导入与常量


# ==================== 帮助内容（内置字典） ====================
# 12 个章节，按用户工作流顺序组织
HELP_CONTENT = {
    '快速入门': """【快速入门】

欢迎使用「结核病传播风险评估系统」。本章节将带您在 5 分钟内完成首次评估。

1. 启动系统
   - 双击运行主程序，或通过命令行执行 ``tb-risk gui``
   - 默认窗口尺寸 1100x750，可自由缩放

2. 录入患者信息
   - 切换到「患者基本信息」标签页
   - 填写年龄、痰涂片、空洞、治疗情况等核心字段
   - 选择家庭居住条件与潜伏感染比例

3. 添加接触者
   - 按 Ctrl+N 在「家庭成员」与「社会接触者」标签页之间自动判断并添加
   - 或点击对应标签页的「添加」按钮
   - 填写姓名、年龄、关系、接触频率、通风条件等

4. 执行评估
   - 按 F5 或点击「开始风险评估」按钮
   - 等待 SEIR 模型与 ML 评分完成
   - 在「风险评估结果」标签页查看综合结论

5. 导出报告
   - 按 Ctrl+E 或通过「文件」菜单导出评估结果
   - 支持 CSV / Excel / JSON / PDF 多种格式

提示：所有数据仅保存在本地，不会上传网络。
""",

    '患者信息录入': """【患者信息录入】

「患者基本信息」标签页用于录入指示病例（index case）的核心临床与流行病学字段。

1. 必填字段
   - 年龄（岁）
   - 痰涂片结果（阴性 / 阳性 1+ / 2+ / 3+）
   - 是否有空洞（是 / 否）
   - 治疗情况（未治疗 / 初治 / 复治 / 治疗完成）

2. 风险修饰字段
   - 症状严重程度：轻度 / 中度 / 重度
   - 病程阶段：潜伏期 / 活动期 / 恢复期
   - 家庭居住条件：1-5 级（数字越大居住越拥挤）
   - 潜伏感染比例（FLP%）：0-100
   - 高风险亚人群比例（HRSP%）：0-100

3. 场景预设
   - 顶部「场景选择」下拉框提供 4 种预设：
     * 农村家庭、城市职场、学校、克拉玛依油田
   - 选择预设后会自动填充环境参数默认值
   - 自定义模式允许手动输入全部参数

4. 校验提示
   - 字段右侧出现红色文字表示校验失败
   - 鼠标悬停在输入框上可查看详细说明
   - 底部进度条显示总填写完整度

5. 自动保存
   - 每次修改后系统会标记为「脏数据」
   - 定时自动保存到本地缓存目录
   - 退出时若仍有未保存修改，将弹出确认对话框
""",

    '接触者管理': """【接触者管理】

系统支持两类接触者：家庭成员与社会接触者，分别位于独立标签页。

1. 添加接触者
   - 点击「添加家庭成员」/「添加社会接触者」按钮
   - 或按 Ctrl+N（系统根据当前标签页自动判断类型）
   - 在弹出的 ContactEditDialog 中填写字段

2. 编辑接触者
   - 选中表格行后点击「编辑选中」按钮
   - 或直接按 Enter 键编辑当前选中行
   - 修改后自动应用场景默认值（如启用）

3. 删除接触者
   - 选中表格行后点击「删除选中」按钮
   - 或按 Delete 键删除当前选中行
   - 删除前会弹出确认对话框，避免误操作

4. 双击快捷操作
   - 双击表格行可切换「是否确诊」状态
   - 适用于快速标记已确诊的继发病例

5. 撤销 / 重做
   - Ctrl+Z 撤销最近一次操作
   - Ctrl+Y 重做最近一次撤销
   - 操作历史由 undo_manager 管理，支持多步撤销

6. 批量操作
   - 「粘贴数据导入」按钮支持从剪贴板批量导入
   - 适合从 Excel 复制多行数据后一次性录入
   - 导入时自动校验字段类型与取值范围

7. 字段说明
   - 关系：配偶 / 子女 / 父母 / 同事 / 同学 / 朋友 等
   - 单次时长（分钟）：每次接触的持续时间
   - 频次密度（次/周）：每周接触次数
   - 持续周期（周）：接触关系总持续时间
   - 通风条件：1-5 级（数字越大通风越差）
   - 接触距离：极近 / 近 / 中 / 远
   - 暴露场景：家庭 / 工作场所 / 学校 / 公共场所 / 医疗机构
""",

    '数据导入导出': """【数据导入导出】

「文件」菜单提供完整的数据交换能力，支持多种格式与来源。

1. CSV 模板导出
   - 「文件」→「导出 CSV 模板」生成标准模板
   - 包含所有字段列与示例行，便于填写

2. 数据导入（6 种来源）
   - 从 CSV 导入：标准逗号分隔文件
   - 从 Excel 导入：.xlsx / .xls 文件
   - 从 JSON 导入：结构化 JSON 文件
   - 从 API 导入：HTTP 接口拉取数据
   - 从数据库导入：直接 SQL 查询
   - 粘贴数据导入：从剪贴板批量录入

   导入流程：
   a) 选择来源并加载文件
   b) 预览面板显示前 N 行与字段映射
   c) 系统自动校验字段类型与取值
   d) 错误行以红色高亮，可点击查看详情
   e) 确认后写入家庭/社会接触者列表

3. 数据导出
   - 保存数据到 CSV：导出当前接触者数据
   - 导出评估结果：保存风险评估结论
   - 导出 Excel 报告：含图表的完整报告
   - 导出 JSON 报告：结构化数据交换
   - 导出 PDF 报告：可打印的最终报告
   - 导出模型训练报告：ML/GNN 训练详情
   - 导出到数据库：写入关系型数据库

4. 快捷键
   - Ctrl+O：从 CSV 导入
   - Ctrl+S：保存到 CSV
   - Ctrl+E：导出评估结果

5. 数据安全
   - 所有导出文件均保存在用户指定路径
   - 不上传任何数据到网络
   - 数据库连接信息不写入日志
""",

    '风险评估': """【风险评估】

「风险评估结果」标签页展示综合评估结论与可视化。

1. 启动评估
   - 按 F5 或点击「开始风险评估」按钮
   - 评估流程：特征初始化 → SEIR 仿真 → 风险评分 → ML 增强
   - 进度条显示当前阶段与百分比

2. 综合风险等级
   - 极低 / 低 / 中 / 高 / 极高 五级
   - 基于接触者累计暴露剂量与传染概率计算
   - 颜色编码：绿 / 黄 / 橙 / 红 / 深红

3. 风险柱状图
   - 横轴：接触者姓名
   - 纵轴：风险评分（0-1）
   - 颜色：风险等级
   - 鼠标悬停显示详细数值
   - 点击柱条可选中对应接触者

4. SEIR 曲线
   - 显示易感者(S) / 暴露者(E) / 感染者(I) / 康复者(R) 随时间演化
   - 支持参数敏感性分析（拖动滑块调整 β、γ、σ）
   - 双击曲线可标记关键时间点

5. 风险热力图
   - 行：接触者
   - 列：风险维度（接触频率、距离、通风、症状等）
   - 颜色：单维度风险贡献
   - 点击单元格查看该维度的详细解释

6. 潜在患者识别
   - 系统自动标记高风险接触者为「潜在患者」
   - 提供干预建议（隔离、检测、预防性治疗）
   - 建议文本可一键导出为 PDF

7. 评估结果刷新
   - 修改接触者数据后需重新评估
   - 系统会自动清除旧结果并提示
   - ML 结果在后台线程更新，不阻塞 UI
""",

    'ML/SHAP 分析': """【ML/SHAP 分析】

机器学习增强模块提供基于 XGBoost / LightGBM 的风险预测与 SHAP 可解释性分析。

1. ML 训练配置
   - 在「训练面板」→「ML 配置」标签页设置参数
   - 关键参数：
     * n_samples：蒙特卡洛样本数（默认 2000）
     * random_state：随机种子（默认 42）
     * ml_cv_folds：交叉验证折数（默认 5）
     * ml_enable_hyperopt：是否启用超参搜索
   - 配置可序列化为 JSON 便于复现

2. 训练流程
   - 点击「开始训练」按钮启动后台线程
   - 训练日志实时显示在底部日志面板
   - 训练完成后自动更新结果图表

3. ML 评分对比
   - 传统评分 vs ML 评分散点图
   - 显示 C-index、AUC(52w) 等指标
   - 校准曲线展示预测概率与实际概率一致性

4. SHAP 特征重要性
   - 全局 SHAP 值条形图：特征对模型输出的平均贡献
   - 个体 SHAP 解释：选中接触者查看其风险归因
   - SHAP 依赖图：特征值与 SHAP 值的关系

5. 模型仓库
   - 训练完成的模型自动保存到本地仓库
   - 支持版本号管理（语义化版本）
   - 可按类型筛选、加载、删除模型
   - 原子写入保证仓库索引完整性

6. 训练日志
   - 每次训练生成独立日志文件
   - 支持倒序读取、limit / 类型筛选
   - 损坏单行自动跳过，不影响整体读取

7. 依赖说明
   - ML 模块依赖 scikit-learn、xgboost、shap、joblib
   - 若未安装，相关功能会自动禁用并提示
""",

    'MCMC 诊断': """【MCMC 诊断】

马尔可夫链蒙特卡洛（MCMC）用于 SEIR 参数的后验估计与不确定性量化。

1. 贝叶斯校准
   - 在「克拉玛依」菜单或训练面板中启动
   - 基于历史数据进行参数后验采样
   - 默认采样 5000 次，可配置

2. 诊断指标
   - R-hat（Gelman-Rubin）：链间收敛性，<1.01 为收敛
   - ESS（有效样本量）：>400 为良好
   - 自相关函数：衰减速度反映采样效率
   - Trace plot：链的轨迹应类似「毛毛虫」

3. 后验预测检验
   - 采样后生成后验预测分布
   - 与观测数据对比验证模型拟合
   - 提供 p-value 与贝叶斯因子

4. 不确定性量化
   - 95% 可信区间（Credible Interval）
   - 后验均值 / 中位数 / 标准差
   - 参数相关性热力图

5. 输出文件
   - 后验样本保存为 .npz
   - 诊断图保存为 .png
   - 摘要报告保存为 .txt

6. 性能提示
   - MCMC 计算密集，建议在多核 CPU 上运行
   - 可通过 n_chains 参数并行化
   - 长时间任务可最小化窗口，后台继续运行

7. 依赖说明
   - 可选依赖 scipy、numpy
   - 高级功能依赖 PyMC3 / ArviZ（如已安装）
""",

    '反事实干预分析': """【反事实干预分析】

反事实分析（Counterfactual Analysis）回答「如果……会怎样」的因果问题。

1. 反事实场景
   - 假设某接触者被提前隔离
   - 假设通风条件改善至 X 级
   - 假设预防性治疗覆盖率提升至 Y%

2. 可微优化器
   - 使用 DifferentiableCounterfactualOptimizer
   - 基于 PyTorch 自动微分求解最优干预
   - 目标：在预算约束下最小化总感染数

3. 干预规划
   - DynamicInterventionPlanner 生成动态干预策略
   - 考虑时间维度：何时对何人采取何种干预
   - 输出干预时间表与预期效果

4. 干预验证
   - InterventionValidator 验证干预的因果效应
   - 基于潜在结果框架
   - 提供 ATE（平均处理效应）与 CATE（条件平均处理效应）

5. 偏好条件策略
   - PreferenceConditionedPolicy 支持用户偏好
   - 例如：偏好「最小化经济成本」vs「最大化健康收益」
   - 通过偏好向量条件化策略网络

6. 结果可视化
   - 反事实 vs 事实对比曲线
   - 干预效果按接触者排序
   - 敏感性分析（参数扰动下的鲁棒性）

7. 依赖说明
   - 核心依赖 PyTorch
   - 可选依赖 Pyro（概率编程）
   - 若未安装，相关菜单项会自动禁用
""",

    '因果 DAG': """【因果 DAG】

有向无环图（DAG）用于表示变量之间的因果关系结构。

1. DAG 构建
   - 在「训练面板」→「因果分析」标签页编辑
   - 节点：变量（年龄、通风、接触频率、感染结局 等）
   - 边：因果方向（A → B 表示 A 导致 B）

2. 因果图约束
   - CausalGraphConstraint 类校验图的无环性
   - 检测潜在的后门路径与混淆变量
   - 提示需要控制哪些变量以识别因果效应

3. DAG 可视化
   - 自动布局（spring / hierarchical / circular）
   - 节点大小反映变量重要性
   - 边粗细反映因果强度
   - 支持缩放与拖拽

4. 干预的图操作
   - do-operator：在 DAG 上模拟干预
   - 删除指向干预节点的所有边
   - 计算干预后的边缘分布

5. 与 ML 集成
   - DAG 指导特征选择（避免后门变量）
   - DAG 指导 SHAP 解释的因果归因
   - 支持基于 DAG 的反事实推理

6. 验证与敏感性
   - 通过条件独立性检验验证边
   - 通过结构学习算法（PC、GES）发现 DAG
   - 提供多种 DAG 候选的对比

7. 注意事项
   - DAG 是因果假设的编码，不是数据驱动发现
   - 错误的 DAG 会导致错误的因果结论
   - 建议结合领域知识与数据检验共同构建
""",

    'GNN 训练': """【GNN 训练】

图神经网络（GNN）用于在接触者网络上进行风险传播建模。

1. 图构建
   - EnhancedTemporalGraphBuilder 构建时空图
   - 节点：接触者（含患者）
   - 边：接触关系（含时间权重与距离权重）
   - 多模态特征：人口学 + 临床 + 行为

2. 模型架构
   - HeteroTimeVaryingGNN：异构时变 GNN
     * 支持不同节点类型（患者 / 家庭接触 / 社会接触）
     * 支持时间演化的边权重
   - MultimodalGNN：多模态融合 GNN
     * 融合表格、文本、图像特征
   - Physics-Informed GNN（PIGNN）：
     * 结合 SEIR ODE 物理约束

3. 三阶段训练
   - ThreePhaseTrainer 实现：
     a) 阶段 1：节点特征预训练（自监督）
     b) 阶段 2：边权重构（对比学习）
     c) 阶段 3：下游风险预测（监督学习）
   - 每阶段独立 epoch 与学习率

4. 训练配置
   - gnn_n_epochs：默认 50
   - gnn_learning_rate：默认 0.001
   - gnn_enable_hyperopt：是否启用超参搜索
   - 支持早停、学习率调度、梯度裁剪

5. 不确定性量化
   - DeepEnsemblePredictor：深度集成
   - MCDropoutGNNUncertainty：MC Dropout
   - SWAGEstimator：SWA 近似贝叶斯
   - ConformalPredictor：保形预测（有限样本保证）
   - UncertaintyFusionEngine：融合多种不确定性

6. 评估指标
   - C-index（一致性指数）
   - AUC@52w（52 周生存 AUC）
   - Brier Score
   - 校准曲线

7. 依赖说明
   - 必需：PyTorch
   - 必需：PyTorch Geometric
   - 可选：PyTorch Geometric Temporal（时空 GNN）
   - 若未安装，GNN 标签页会显示禁用提示
""",

    '快捷键列表': """【快捷键列表】

系统提供完整键盘快捷键支持，提升高频操作效率。

┌──────────────────┬─────────────────────────────────┐
│ 快捷键           │ 功能                            │
├──────────────────┼─────────────────────────────────┤
│ Ctrl + N         │ 添加接触者（按当前标签页判断）  │
│ Ctrl + O         │ 从 CSV 导入数据                 │
│ Ctrl + S         │ 保存数据到 CSV                  │
│ Ctrl + E         │ 导出评估结果                    │
│ F1               │ 显示帮助窗口                    │
│ F5               │ 执行风险评估                    │
│ Ctrl + Q         │ 退出程序                        │
│ Ctrl + W         │ 关闭生存分析对话框              │
│ Ctrl + Tab       │ 切换到下一个标签页              │
│ Ctrl + Shift+Tab │ 切换到上一个标签页              │
│ Ctrl + 加号      │ 图表放大                        │
│ Ctrl + 减号      │ 图表缩小                        │
│ Ctrl + Z         │ 撤销                            │
│ Ctrl + Y         │ 重做                            │
│ Delete           │ 删除选中接触者                  │
│ Enter            │ 编辑选中行                      │
│ Escape           │ 关闭顶层弹窗                    │
└──────────────────┴─────────────────────────────────┘

提示：
- 快捷键全局生效，无论当前焦点在哪个控件
- 标签页切换会自动更新 Ctrl+N / Delete / Enter 的行为
- 图表缩放仅对 matplotlib 图表生效
- 在文本输入框中按 Ctrl+Z 优先触发文本框内置撤销
""",

    'FAQ': """【常见问题（FAQ）】

Q1: 启动时提示「matplotlib 不可用」怎么办？
A1: 安装可视化依赖：``pip install matplotlib seaborn``。
    系统会自动降级为纯文本模式，但图表功能不可用。

Q2: ML 模块显示「已禁用」？
A2: 安装 ML 依赖：``pip install scikit-learn xgboost shap joblib``。
    或安装完整依赖：``pip install tb_risk[ml]``。

Q3: GNN 训练按钮灰色不可点击？
A3: GNN 依赖 PyTorch 与 PyTorch Geometric：
    ``pip install torch torch-geometric``
    安装后重启程序即可。

Q4: 数据丢失了怎么办？
A4: 系统每 60 秒自动保存到本地缓存目录。
    查看：用户目录下的 .tb_risk_cache 文件夹。
    也可通过「文件」→「从 JSON 导入」加载最近自动保存。

Q5: 评估结果为空？
A5: 检查以下几点：
    1) 至少添加 1 个接触者
    2) 患者基本信息必填字段已填写
    3) 查看底部日志面板是否有错误信息
    4) 尝试按 F5 重新评估

Q6: MCMC 采样很慢？
A6: MCMC 是计算密集型任务。建议：
    1) 减少采样次数（n_samples）
    2) 使用多核 CPU（n_chains > 1）
    3) 关闭其他 CPU 密集型程序
    4) 考虑使用变分推断（VI）作为快速近似

Q7: SHAP 图表报错？
A7: SHAP 与 numpy 版本兼容性问题常见。尝试：
    1) 升级 shap：``pip install -U shap``
    2) 升级 numpy：``pip install -U numpy``
    3) 若仍失败，禁用 SHAP 后 ML 评分仍可用

Q8: 克拉玛依本土化模块如何启用？
A8: 程序自动检测本地化模块。若未启用：
    1) 检查 karamay/ 目录是否完整
    2) 查看「克拉玛依」菜单是否出现
    3) 通过「克拉玛依」→「显示本土化模块信息」查看状态

Q9: 导出 PDF 报告失败？
A9: PDF 导出依赖 reportlab：
    ``pip install reportlab``
    若字体缺失，系统会回退到默认字体。

Q10: 如何反馈 bug 或建议？
A10: 请通过项目仓库的 Issue 系统提交：
     1) 描述复现步骤
     2) 附上日志文件（位于 .tb_risk_cache/logs/）
     3) 注明操作系统与 Python 版本
""",
}


class HelpWindow:
    """Section X: 独立帮助窗口

    左侧目录树导航 + 右侧 ScrolledText 显示内容，支持全文搜索与高亮。
    防止重复打开：若已存在实例，则聚焦已有窗口。

    使用方式：
        HelpWindow(parent_root)  # 创建并显示
    """

    # 类级单例引用，防止重复打开
    _instance = None

    def __init__(self, parent_root):
        """初始化帮助窗口

        Args:
            parent_root: 父窗口（通常是 self.root）
        """
        try:
            # 单例检查：若已存在且未关闭，则聚焦并返回
            if HelpWindow._instance is not None:
                inst = HelpWindow._instance
                try:
                    if inst._window.winfo_exists():
                        inst._window.lift()
                        inst._window.focus_force()
                        return
                except (tk.TclError, AttributeError, RuntimeError) as e:
                    LOGGER.debug("HelpWindow 单例窗口已失效: %s", e)
                # 引用已失效，清理后重建
                HelpWindow._instance = None

            self._parent = parent_root
            self._window = tk.Toplevel(parent_root)
            self._window.title("使用帮助 — 结核病传播风险评估系统")
            self._window.geometry("900x640")
            self._window.minsize(700, 480)

            # 从父窗口获取主题颜色（若可用）
            self._colors = self._get_colors_from_parent()

            # 窗口关闭时清理单例引用
            self._window.protocol("WM_DELETE_WINDOW", self._on_close)

            # 当前选中章节（用于搜索时维持上下文）
            self._current_section = None
            # 搜索高亮 tag 计数（避免 tag 冲突）
            self._highlight_tags = []

            self._build_ui()
            # 默认显示第一个章节
            first_section = next(iter(HELP_CONTENT.keys()), None)
            if first_section:
                self._show_section(first_section)

            HelpWindow._instance = self
        except Exception as e:
            LOGGER.warning("HelpWindow 初始化失败: %s", e)
            HelpWindow._instance = None

    def _get_colors_from_parent(self):
        """从父窗口获取当前主题颜色字典，失败则返回浅色默认值"""
        default = {
            'bg_card': '#ffffff', 'text': '#2c3e50', 'accent': '#3498db',
            'warning': '#f39c12', 'bg_light': '#f8f9fa',
        }
        try:
            if hasattr(self._parent, 'master') and hasattr(self._parent.master, 'COLORS'):
                colors = self._parent.master.COLORS
                return {k: colors.get(k, v) for k, v in default.items()}
            elif hasattr(self._parent, 'COLORS'):
                colors = self._parent.COLORS
                return {k: colors.get(k, v) for k, v in default.items()}
        except Exception as e:
            LOGGER.debug("从父窗口获取主题颜色失败，使用默认值: %s", e)
        return default

    def apply_theme(self, colors=None):
        """外部调用：应用新的主题颜色到窗口控件

        Args:
            colors: 主题颜色字典，若为None则从父窗口重新读取
        """
        try:
            if colors is not None:
                self._colors = {k: colors.get(k, v) for k, v in self._colors.items()}
            else:
                self._colors = self._get_colors_from_parent()
            bg = self._colors['bg_card']
            fg = self._colors['text']
            insert_fg = fg
            sel_bg = self._colors['accent']
            # 更新Text控件
            if hasattr(self, '_text') and self._text.winfo_exists():
                self._text.configure(bg=bg, fg=fg, insertbackground=insert_fg,
                                    selectbackground=sel_bg, selectforeground='white')
                # 更新tag颜色
                self._text.tag_configure('highlight',
                                         background='#fff3a3' if fg == '#2c3e50' else '#665500',
                                         foreground=fg)
                self._text.tag_configure('section_title',
                                         font=('Microsoft YaHei', 14, 'bold'),
                                         foreground=fg,
                                         spacing3=8)
        except Exception as e:
            LOGGER.debug("apply_theme 更新控件失败: %s", e)

    def _build_ui(self):
        """构建左侧目录树 + 右侧内容区 + 顶部搜索框"""
        try:
            # 顶部搜索栏
            search_frame = ttk.Frame(self._window, padding=(8, 6))
            search_frame.pack(side='top', fill='x')

            ttk.Label(search_frame, text="搜索：").pack(side='left')
            self._search_var = tk.StringVar()
            self._search_var.trace_add('write', self._on_search_change)
            search_entry = ttk.Entry(search_frame, textvariable=self._search_var,
                                     width=40)
            search_entry.pack(side='left', fill='x', expand=True, padx=(4, 8))
            ttk.Label(search_frame, text="（输入关键词自动高亮匹配）",
                      font=('Microsoft YaHei', 8)).pack(side='left')

            # 中部 PanedWindow：左目录树 + 右内容
            paned = ttk.PanedWindow(self._window, orient='horizontal')
            paned.pack(fill='both', expand=True, padx=8, pady=(0, 8))

            # 左侧目录树
            tree_frame = ttk.Frame(paned)
            paned.add(tree_frame, weight=1)
            ttk.Label(tree_frame, text="目录", font=('Microsoft YaHei', 10, 'bold')
                      ).pack(anchor='w', padx=4, pady=(4, 2))
            self._tree = ttk.Treeview(tree_frame, show='tree', selectmode='browse')
            self._tree.pack(fill='both', expand=True, padx=4, pady=4)
            tree_scroll = ttk.Scrollbar(tree_frame, orient='vertical',
                                        command=self._tree.yview)
            tree_scroll.pack(side='right', fill='y')
            self._tree.configure(yscrollcommand=tree_scroll.set)
            self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)

            # 填充目录项
            for section in HELP_CONTENT.keys():
                self._tree.insert('', 'end', iid=section, text=section)

            # 右侧内容区
            content_frame = ttk.Frame(paned)
            paned.add(content_frame, weight=3)
            ttk.Label(content_frame, text="内容", font=('Microsoft YaHei', 10, 'bold')
                      ).pack(anchor='w', padx=4, pady=(4, 2))

            # 使用 ScrolledText 显示内容（兼容标准库）
            bg = self._colors['bg_card']
            fg = self._colors['text']
            sel_bg = self._colors['accent']
            highlight_bg = '#fff3a3' if fg == '#2c3e50' else '#665500'
            try:
                from tkinter.scrolledtext import ScrolledText
                self._text = ScrolledText(content_frame, wrap='word',
                                          font=('Microsoft YaHei', 10),
                                          padx=10, pady=8,
                                          bg=bg, fg=fg,
                                          insertbackground=fg,
                                          selectbackground=sel_bg,
                                          selectforeground='white')
            except ImportError:
                # 回退：手动构造 Text + Scrollbar
                text_frame = ttk.Frame(content_frame)
                text_frame.pack(fill='both', expand=True)
                self._text = tk.Text(text_frame, wrap='word',
                                     font=('Microsoft YaHei', 10),
                                     padx=10, pady=8,
                                     bg=bg, fg=fg,
                                     insertbackground=fg,
                                     selectbackground=sel_bg,
                                     selectforeground='white')
                self._text.pack(side='left', fill='both', expand=True)
                sb = ttk.Scrollbar(text_frame, orient='vertical',
                                   command=self._text.yview)
                sb.pack(side='right', fill='y')
                self._text.configure(yscrollcommand=sb.set)
            else:
                self._text.pack(fill='both', expand=True, padx=4, pady=4)

            # 配置文本 tag（颜色跟随主题）
            self._text.tag_configure('highlight',
                                     background=highlight_bg,
                                     foreground=fg)
            self._text.tag_configure('section_title',
                                     font=('Microsoft YaHei', 14, 'bold'),
                                     foreground=fg,
                                     spacing3=8)
            self._text.configure(state='disabled')
        except Exception as e:
            LOGGER.warning("HelpWindow._build_ui 构建 UI 失败: %s", e)

    def _on_tree_select(self, event=None):
        """目录树选中事件：显示对应章节内容"""
        try:
            selection = self._tree.selection()
            if selection:
                self._show_section(selection[0])
        except Exception as e:
            LOGGER.debug("_on_tree_select 事件处理失败: %s", e)

    def _show_section(self, section_name):
        """显示指定章节内容

        Args:
            section_name: HELP_CONTENT 中的章节键名
        """
        try:
            content = HELP_CONTENT.get(section_name, '')
            if not content:
                return
            self._current_section = section_name
            self._text.configure(state='normal')
            self._text.delete('1.0', 'end')
            self._text.insert('1.0', content)
            self._text.configure(state='disabled')
            # 应用当前搜索关键词的高亮
            self._apply_highlight()
        except Exception as e:
            LOGGER.debug("_show_section 显示章节 %r 失败: %s", section_name, e)

    def _on_search_change(self, *args):
        """搜索框内容变化时刷新高亮"""
        try:
            self._apply_highlight()
        except Exception as e:
            LOGGER.debug("_on_search_change 高亮刷新失败: %s", e)

    def _apply_highlight(self):
        """在当前显示的内容中高亮搜索关键词

        实现策略：
        1) 清除所有旧高亮 tag
        2) 若搜索框为空则直接返回
        3) 否则在全文中查找所有匹配位置并添加 'highlight' tag
        4) 滚动到第一个匹配处
        """
        try:
            # 清除旧高亮
            self._text.tag_remove('highlight', '1.0', 'end')

            keyword = self._search_var.get().strip()
            if not keyword:
                return

            # 大小写不敏感搜索
            start = '1.0'
            first_match = None
            while True:
                pos = self._text.search(keyword, start, 'end',
                                        nocase=True, count=tk.IntVar())
                if not pos:
                    break
                end_pos = f"{pos}+{len(keyword)}c"
                self._text.tag_add('highlight', pos, end_pos)
                if first_match is None:
                    first_match = pos
                start = end_pos

            # 滚动到第一个匹配
            if first_match:
                self._text.see(first_match)
        except Exception as e:
            LOGGER.debug("_apply_highlight 搜索高亮失败: %s", e)

    def _on_close(self):
        """窗口关闭：清理单例引用并销毁窗口"""
        try:
            HelpWindow._instance = None
            self._window.destroy()
        except Exception as e:
            LOGGER.debug("_on_close 销毁帮助窗口失败: %s", e)

    @classmethod
    def is_open(cls):
        """检查帮助窗口是否已打开"""
        try:
            if cls._instance is None:
                return False
            return cls._instance._window.winfo_exists()
        except Exception as e:
            LOGGER.debug("is_open 检查窗口状态失败: %s", e)
            cls._instance = None
            return False

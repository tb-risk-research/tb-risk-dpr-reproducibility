#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""有向无环图（DAG）— 因果推断的结构基础

本模块提供因果图数据结构，显式建模变量间的因果关系。
SHAP 衡量的是相关性，而因果推断要求先有结构假设（DAG）才能将关联转化为因果效应。

核心能力：
- 节点角色标注：暴露/结局/混杂/中介/效应修饰/碰撞/工具
- 环检测（保证无环性）
- d-分离判定（Pearl, Causality 2009, §1.2.3）
- 后门路径枚举与充分调整集求解（后门准则，Pearl 1995）

文献支撑：
- Pearl J. Causality: Models, Reasoning, and Inference. Cambridge UP, 2009.
- Spirtes P et al. Causation, Prediction, and Search. MIT Press, 2000.
- Koller D, Friedman N. Probabilistic Graphical Models. MIT Press, 2009. §3.3.

依赖策略：networkx 可选。不可用时降级为内部邻接表实现，
与项目现有 conditional import 模式一致。
"""
import enum
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger("tb_risk.validation.causal.dag")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class NodeType(enum.Enum):
    """变量在因果图中的角色（Pearl 因果阶梯中的语义分类）。"""
    EXPOSURE = 'exposure'             # 暴露/处理变量（干预对象）
    OUTCOME = 'outcome'               # 结局变量（效应测量对象）
    CONFOUNDER = 'confounder'         # 混杂因素（同时影响暴露与结局）
    MEDIATOR = 'mediator'             # 中介变量（暴露→中介→结局路径上）
    EFFECT_MODIFIER = 'effect_modifier'  # 效应修饰因子（改变效应大小）
    COLLIDER = 'collider'             # 碰撞节点（暴露与结局共同后果）
    INSTRUMENT = 'instrument'         # 工具变量（影响暴露但不直接影响结局）
    OBSERVED = 'observed'             # 一般观测变量（无特定角色）


class CausalDAG:
    """有向无环图（DAG）— 因果推断的结构假设载体。

    DAG 是因果推断的前提：没有结构假设，观察数据中的关联无法转化为因果效应。
    本类显式声明变量间的因果方向，并基于图结构自动识别混杂路径与调整集。

    示例：
        dag = CausalDAG()
        dag.add_node('age', NodeType.CONFOUNDER, '年龄')
        dag.add_node('bcg', NodeType.CONFOUNDER, '卡介苗接种')
        dag.add_node('exposure', NodeType.EXPOSURE, '累积暴露')
        dag.add_node('risk', NodeType.OUTCOME, '发病风险')
        dag.add_edge('age', 'bcg')
        dag.add_edge('age', 'risk')
        dag.add_edge('bcg', 'risk')
        dag.add_edge('exposure', 'risk')
        assert dag.is_acyclic()
        adj = dag.find_adjustment_set('exposure', 'risk')  # {'age', 'bcg'}
    """

    def __init__(self):
        # 节点：name -> (NodeType, description)
        self._nodes: Dict[str, Tuple[NodeType, str]] = {}
        # 邻接表：name -> set(child names)
        self._children: Dict[str, Set[str]] = {}
        # 反向邻接表：name -> set(parent names)
        self._parents: Dict[str, Set[str]] = {}
        # 边描述：(cause, effect) -> description
        self._edge_desc: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def add_node(self, name: str, node_type: NodeType = NodeType.OBSERVED,
                 description: str = "") -> "CausalDAG":
        """添加节点。重复添加同名节点将更新其角色与描述。"""
        if not name:
            raise ValueError("节点名不能为空")
        self._nodes[name] = (node_type, description)
        self._children.setdefault(name, set())
        self._parents.setdefault(name, set())
        return self

    def add_edge(self, cause: str, effect: str,
                 description: str = "") -> "CausalDAG":
        """添加有向边 cause → effect。若引入环则抛出 ValueError。"""
        if cause == effect:
            raise ValueError(f"自环不被允许: {cause}")
        for n in (cause, effect):
            if n not in self._nodes:
                self.add_node(n)
        # 临时加入边后做环检测，环则回滚
        self._children[cause].add(effect)
        self._parents[effect].add(cause)
        self._edge_desc[(cause, effect)] = description
        if not self.is_acyclic():
            self._children[cause].discard(effect)
            self._parents[effect].discard(cause)
            self._edge_desc.pop((cause, effect), None)
            raise ValueError(
                f"添加边 {cause}→{effect} 会引入环，违反 DAG 无环性约束")
        return self

    # ------------------------------------------------------------------
    # 基本查询
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def node_info(self, name: str) -> Tuple[NodeType, str]:
        if name not in self._nodes:
            raise KeyError(f"未知节点: {name}")
        return self._nodes[name]

    def node_type(self, name: str) -> NodeType:
        return self.node_info(name)[0]

    def edges(self) -> List[Tuple[str, str]]:
        return [(c, e) for c, children in self._children.items() for e in children]

    def parents(self, name: str) -> Set[str]:
        return set(self._parents.get(name, set()))

    def children(self, name: str) -> Set[str]:
        return set(self._children.get(name, set()))

    def neighbors(self, name: str) -> Set[str]:
        """所有相邻节点（不论方向）。"""
        return self.parents(name) | self.children(name)

    def ancestors(self, name: str) -> Set[str]:
        """所有祖先节点（直接或间接指向 name 的节点）。"""
        result: Set[str] = set()
        stack = list(self._parents.get(name, set()))
        while stack:
            n = stack.pop()
            if n not in result:
                result.add(n)
                stack.extend(self._parents.get(n, set()))
        return result

    def descendants(self, name: str) -> Set[str]:
        """所有后代节点（name 直接或间接指向的节点）。"""
        result: Set[str] = set()
        stack = list(self._children.get(name, set()))
        while stack:
            n = stack.pop()
            if n not in result:
                result.add(n)
                stack.extend(self._children.get(n, set()))
        return result

    def nodes_by_type(self, node_type: NodeType) -> List[str]:
        """按角色筛选节点。"""
        return [n for n, (t, _) in self._nodes.items() if t == node_type]

    # ------------------------------------------------------------------
    # 环检测
    # ------------------------------------------------------------------
    def is_acyclic(self) -> bool:
        """验证图的无环性（Kahn 拓扑排序）。"""
        in_degree = {n: len(self._parents.get(n, set())) for n in self._nodes}
        queue = [n for n, d in in_degree.items() if d == 0]
        visited = 0
        # 操作副本，避免修改原结构
        remaining = dict(in_degree)
        while queue:
            n = queue.pop()
            visited += 1
            for child in self._children.get(n, set()):
                remaining[child] -= 1
                if remaining[child] == 0:
                    queue.append(child)
        return visited == len(self._nodes)

    # ------------------------------------------------------------------
    # d-分离
    # ------------------------------------------------------------------
    def is_d_separated(self, x: str, y: str,
                       z: Optional[Iterable[str]] = None) -> bool:
        """判定 x 与 y 在给定条件集 z 下是否 d-分离。

        d-分离（Pearl 2009, Def 1.2.3）：路径被 z 阻断当且仅当：
        - 链 A→M→B 或分叉 A←M→B：M ∈ z
        - 碰撞 A→M←B：M ∉ z 且 M 的所有后代 ∉ z

        若 networkx 可用且版本支持，委托 nx.d_separated；否则使用
        Bayes-Ball 可达性算法（Koller & Friedman 2009, Algorithm 3.1）。
        """
        z_set = set(z) if z else set()
        for n in (x, y):
            if n not in self._nodes:
                raise KeyError(f"未知节点: {n}")
        z_set = {n for n in z_set if n in self._nodes}

        if NETWORKX_AVAILABLE:
            try:
                return bool(nx.algorithms.d_separated(
                    self._to_networkx(), {x}, {y}, z_set))
            except (AttributeError, Exception):  # pragma: no cover
                # 旧版 networkx 无 d_separated，降级到内部实现
                pass

        return not self._reachable(x, y, z_set)

    def _reachable(self, x: str, y: str, z: Set[str]) -> bool:
        """Bayes-Ball 可达性：判断 y 是否能从 x 经活跃路径到达。

        实现 Koller & Friedman (2009) Algorithm 3.1 "Reachable"。
        """
        # Phase I：找 z 中所有节点及其祖先（碰撞节点需"激活"）
        z_ancestors: Set[str] = set(z)
        for node in list(z):
            z_ancestors |= self.ancestors(node)

        # Phase II：BFS，状态 (node, direction)
        # direction='up' 表示从子节点到达（沿父方向走）
        # direction='down' 表示从父节点到达（沿子方向走）
        visited: Set[Tuple[str, str]] = set()
        to_visit: List[Tuple[str, str]] = [(x, 'up')]
        reachable_nodes: Set[str] = set()

        while to_visit:
            node, direction = to_visit.pop()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            reachable_nodes.add(node)

            if direction == 'up':
                # 从子节点到达（即沿 node→child 走到这里，或起点）
                # 可继续向上（父节点）：若 node ∉ z
                if node not in z:
                    for parent in self._parents.get(node, set()):
                        to_visit.append((parent, 'up'))
                # 可向下（子节点）：若 node ∉ z（链/分叉）
                if node not in z:
                    for child in self._children.get(node, set()):
                        to_visit.append((child, 'down'))
            else:  # direction == 'down'
                # 从父节点到达（即沿 parent→node 走到这里）
                # 可继续向下（子节点）：若 node ∉ z（链）
                if node not in z:
                    for child in self._children.get(node, set()):
                        to_visit.append((child, 'down'))
                # 可向上（父节点）：碰撞点——若 node 或其后代 ∈ z_ancestors
                if node in z_ancestors:
                    for parent in self._parents.get(node, set()):
                        to_visit.append((parent, 'up'))

        return y in reachable_nodes

    # ------------------------------------------------------------------
    # 后门路径与调整集
    # ------------------------------------------------------------------
    def find_backdoor_paths(self, exposure: str,
                            outcome: str) -> List[List[str]]:
        """枚举从 exposure 到 outcome 的所有后门路径。

        后门路径：以指向 exposure 的边开头的路径（即 exposure←...→outcome），
        这些路径若不被阻断，会引入混杂偏倚。
        """
        if exposure not in self._nodes or outcome not in self._nodes:
            raise KeyError("暴露或结局节点不存在")
        paths: List[List[str]] = []
        # 从 exposure 的每个父节点出发，找所有到 outcome 的无向路径
        # （不重复经过节点，避免环路径）
        for parent in self._parents.get(exposure, set()):
            self._dfs_undirected_paths(parent, outcome, [exposure, parent],
                                       {exposure, parent}, paths)
        return paths

    def _dfs_undirected_paths(self, current: str, target: str,
                              path: List[str], visited: Set[str],
                              results: List[List[str]]) -> None:
        """沿无向边深度优先搜索路径。"""
        if current == target:
            results.append(list(path))
            return
        for nb in self.neighbors(current):
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                self._dfs_undirected_paths(nb, target, path, visited, results)
                path.pop()
                visited.discard(nb)

    def is_valid_adjustment_set(self, exposure: str, outcome: str,
                                z: Iterable[str]) -> bool:
        """验证 z 是否满足后门准则（Pearl 1995, Def 3.3.1）。

        后门准则：(1) z 中无 exposure 的后代；(2) z 阻断 exposure→outcome
        的所有后门路径。
        """
        z_set = set(z)
        desc = self.descendants(exposure)
        # 条件1：z 不得包含 exposure 的后代（也不得含 exposure 本身）
        if z_set & (desc | {exposure}):
            return False
        # 条件2：z 必须阻断所有后门路径
        for path in self.find_backdoor_paths(exposure, outcome):
            if not self._path_blocked(path, z_set):
                return False
        return True

    def _path_blocked(self, path: List[str], z: Set[str]) -> bool:
        """判断单条路径是否被 z 阻断（d-分离路径级判定）。

        路径被阻断当且仅当存在某一中间节点阻断它：
        - 链 A→M→B 或分叉 A←M→B：M ∈ z 阻断
        - 碰撞 A→M←B：M ∉ z 且 M 的所有后代 ∉ z 时阻断
        """
        if len(path) < 3:
            # 长度 2 的后门路径形如 [exposure, outcome]，即 outcome→exposure 直接边。
            # 仅当 outcome ∈ z 才能阻断，但结局变量不应进入调整集，故恒为未阻断。
            return path[-1] in z
        for i in range(1, len(path) - 1):
            prev_n, curr_n, next_n = path[i - 1], path[i], path[i + 1]
            prev_to_curr = curr_n in self._children.get(prev_n, set())
            next_to_curr = curr_n in self._children.get(next_n, set())
            is_collider = prev_to_curr and next_to_curr
            if is_collider:
                # 碰撞点：阻断当且仅当 curr 及其后代均不在 z 中
                if curr_n not in z and not (self.descendants(curr_n) & z):
                    return True
            else:
                # 链/分叉：curr ∈ z 即阻断
                if curr_n in z:
                    return True
        return False

    def find_adjustment_set(self, exposure: str,
                            outcome: str) -> Optional[Set[str]]:
        """求解满足后门准则的充分调整集。

        启发式：取 exposure 与 outcome 的共同祖先，剔除 exposure 的后代与
        中介节点。若该集合满足后门准则则返回，否则返回 None（需人工介入）。

        这是 Pearl 后门准则的一个充分实现，覆盖大多数实际 DAG。
        对于复杂图（含多个碰撞后门路径），可能需要更精细的算法。
        """
        if exposure not in self._nodes or outcome not in self._nodes:
            raise KeyError("暴露或结局节点不存在")

        desc_exposure = self.descendants(exposure)
        anc_exposure = self.ancestors(exposure)
        anc_outcome = self.ancestors(outcome)
        # 候选：共同祖先，剔除 exposure 后代与 exposure/outcome 本身
        candidates = (anc_exposure & anc_outcome) - desc_exposure - {exposure, outcome}
        # 剔除中介节点（exposure→mediator→outcome 路径上的节点），
        # 否则会"过度调整"阻断因果效应本身
        mediators = self._find_mediators(exposure, outcome)
        candidates -= mediators

        if self.is_valid_adjustment_set(exposure, outcome, candidates):
            return candidates
        # 回退：尝试逐步扩展候选集（加入更多祖先）
        # 注意：集合运算符 - 优先级高于 |，此处用显式括号避免歧义
        extended = candidates | (
            (anc_exposure | anc_outcome) - desc_exposure
            - {exposure, outcome} - mediators
        )
        if self.is_valid_adjustment_set(exposure, outcome, extended):
            return extended
        LOGGER.warning(
            "无法自动求解充分调整集 (exposure=%s, outcome=%s)；"
            "建议人工审查后门路径并指定 adjustment_set", exposure, outcome)
        return None

    def _find_mediators(self, exposure: str, outcome: str) -> Set[str]:
        """识别 exposure→...→outcome 有向路径上的中介节点。"""
        mediators: Set[str] = set()
        for child in self._children.get(exposure, set()):
            if child == outcome:
                continue
            if outcome in self.descendants(child):
                mediators.add(child)
                # 递归找下游中介
                stack = [child]
                while stack:
                    n = stack.pop()
                    for c in self._children.get(n, set()):
                        if c == outcome:
                            continue
                        if outcome in self.descendants(c):
                            mediators.add(c)
                            stack.append(c)
        return mediators

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """序列化为可 JSON 持久化的字典。"""
        return {
            'nodes': [
                {'name': n, 'type': t.value, 'description': d}
                for n, (t, d) in self._nodes.items()
            ],
            'edges': [
                {'cause': c, 'effect': e, 'description': d}
                for (c, e), d in self._edge_desc.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CausalDAG":
        dag = cls()
        for node in data.get('nodes', []):
            dag.add_node(node['name'],
                         NodeType(node.get('type', 'observed')),
                         node.get('description', ''))
        for edge in data.get('edges', []):
            dag.add_edge(edge['cause'], edge['effect'],
                         edge.get('description', ''))
        return dag

    def _to_networkx(self):
        """转换为 networkx.DiGraph（用于 d_separated 委托）。"""
        if not NETWORKX_AVAILABLE:
            return None
        g = nx.DiGraph()
        for n, (t, d) in self._nodes.items():
            g.add_node(n, type=t.value, description=d)
        for (c, e), d in self._edge_desc.items():
            g.add_edge(c, e, description=d)
        return g

    def summary(self) -> str:
        """人类可读的图摘要。"""
        lines = [f"CausalDAG (nodes={len(self._nodes)}, "
                 f"edges={len(self._edge_desc)}, acyclic={self.is_acyclic()})"]
        type_counts: Dict[str, int] = {}
        for t, _ in self._nodes.values():
            type_counts[t.value] = type_counts.get(t.value, 0) + 1
        for k, v in sorted(type_counts.items()):
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

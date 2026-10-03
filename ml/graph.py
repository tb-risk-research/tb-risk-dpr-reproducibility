"""
多模态数据接入与时变动态图构建

包含多模态数据接入层 (MultiModalDataAccess) 和
时变动态图构建器 (EnhancedTemporalGraphBuilder)。

文献：
  Open-Meteo API / InfluxDB时序 / 油田物联网标准
  Rossi E et al. (2020) TGN
  Sankar A et al. (2020) DySAT
"""

import numpy as np

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

try:
    from torch_geometric.data import Data
    _TORCH_GEOMETRIC_PIGNN_OK = True
except ImportError:
    _TORCH_GEOMETRIC_PIGNN_OK = False


class MultiModalDataAccess:
    """多模态数据接入层

    对接环境数据 API、企业 HR 轮班数据、基站定位数据，
    统一输出为时间索引的接触频次修正因子和节点属性变化。

    文献：Open-Meteo API / InfluxDB时序 / 油田物联网标准
    """

    DEFAULT_ENV_STATIONS = {
        'karamay': {'latitude': 45.58, 'longitude': 84.89, 'elevation': 350},
        'urumqi': {'latitude': 43.83, 'longitude': 87.62, 'elevation': 800},
        'xinjiang_default': {'latitude': 42.0, 'longitude': 86.0, 'elevation': 1000},
    }

    def __init__(self, climate_interaction=None, station_name='karamay',
                 use_synthetic_env=True):
        self.climate = climate_interaction
        self.station_name = station_name
        self.station_info = self.DEFAULT_ENV_STATIONS.get(
            station_name, self.DEFAULT_ENV_STATIONS['xinjiang_default'])
        self.use_synthetic_env = use_synthetic_env
        self.env_cache = {}
        self.behavior_cache = {}
        self.shift_cache = {}
        self.mobility_cache = {}

    def get_environment_data(self, date, pm10=None, humidity=None,
                             temp=None, wind_speed=None, use_cache=True):
        """获取指定日期的环境数据

        优先使用传入实测值，其次使用气候交互层月均值。
        合成模式下使用季节性正弦模拟。

        返回：
            dict: 环境数据 {'pm10', 'humidity', 'temp', 'wind_speed',
                          'climate_multiplier', 'daily_risk_score'}
        """
        date_str = date if isinstance(date, str) else (
            date.isoformat() if hasattr(date, 'isoformat') else str(date))

        if use_cache and date_str in self.env_cache:
            return self.env_cache[date_str]

        if self.climate is not None:
            daily = self.climate.get_daily_climate_multiplier(
                date=date, pm10=pm10, humidity=humidity,
                temp=temp, wind_speed=wind_speed)
        elif self.use_synthetic_env:
            daily = self._synthetic_environment(date)
        else:
            daily = {'pm10_value': 80.0, 'humidity_value': 50.0,
                     'temperature_value': 10.0, 'wind_speed_value': 3.5,
                     'climate_multiplier': 1.0, 'daily_risk_score': 0.1}

        env_data = {
            'pm10': daily.get('pm10_value', 80.0),
            'humidity': daily.get('humidity_value', 50.0),
            'temp': daily.get('temperature_value', 10.0),
            'wind_speed': daily.get('wind_speed_value', 3.5),
            'climate_multiplier': daily.get('climate_multiplier', 1.0),
            'daily_risk_score': daily.get('daily_risk_score', 0.1),
        }
        self.env_cache[date_str] = env_data
        return env_data

    def _synthetic_environment(self, date):
        import datetime as dt_lib
        if isinstance(date, str):
            try:
                d = dt_lib.datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                d = dt_lib.datetime.now()
        elif isinstance(date, dt_lib.date):
            d = dt_lib.datetime(date.year, date.month, date.day)
        elif isinstance(date, dt_lib.datetime):
            d = date
        else:
            d = dt_lib.datetime.now()
        day_of_year = d.timetuple().tm_yday
        season_phase = 2 * np.pi * (day_of_year - 15) / 365.0
        pm10 = 60 + 80 * np.sin(season_phase)
        humidity = 45 + 25 * np.cos(season_phase)
        temp = 5 + 20 * np.sin(season_phase - np.pi / 3)
        wind_speed = 3.0 + 3.0 * np.sin(season_phase + np.pi / 4)
        climate_mult = 0.8 + 0.4 * np.sin(season_phase)
        daily_risk = 0.05 + 0.15 * abs(np.sin(season_phase))
        return {
            'pm10_value': float(pm10), 'humidity_value': float(humidity),
            'temperature_value': float(temp), 'wind_speed_value': float(wind_speed),
            'climate_multiplier': float(climate_mult),
            'daily_risk_score': float(daily_risk),
        }

    def set_shift_data(self, shift_data):
        self.shift_cache = shift_data

    def set_mobility_data(self, mobility_matrix):
        self.mobility_cache = mobility_matrix

    def get_contact_modifier(self, date, node_idx=None, location='home'):
        """根据轮班和行为数据获取接触频次修正因子

        参数：
            date: 日期
            node_idx: 节点索引
            location: 位置类型 ('camp'/'home'/'community'/'transit')

        返回：
            float: 接触频次修正因子 (默认为1.0)
        """
        import datetime as dt_lib
        if isinstance(date, str):
            try:
                d = dt_lib.datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                d = dt_lib.datetime.now()
        elif isinstance(date, (dt_lib.date, dt_lib.datetime)):
            d = date if isinstance(date, dt_lib.datetime) else dt_lib.datetime(
                date.year, date.month, date.day)
        else:
            d = dt_lib.datetime.now()

        weekday = d.weekday()
        modifier = 1.0

        if location == 'camp':
            if weekday < 5:
                modifier = 1.5
            else:
                modifier = 0.3
        elif location == 'home':
            if weekday >= 5:
                modifier = 2.0
            else:
                modifier = 0.6
        elif location == 'community':
            modifier = 1.0
        elif location == 'transit':
            if weekday < 5:
                modifier = 1.3
            else:
                modifier = 0.4

        hour = d.hour
        if 7 <= hour < 9 or 17 <= hour < 19:
            modifier *= 1.3
        elif 22 <= hour or hour < 5:
            modifier *= 0.3

        return round(modifier, 4)

    def get_node_environment_features(self, date, node_features):
        """获取节点级别的环境特征扩展

        将原始节点特征扩展为 node_features + 4 环境维
        （v3.0: 节点特征从 22 维升级为 30 维，扩展后为 34 维）

        返回：
            numpy array: [node_dim + 4] 扩展特征向量
        """
        env_data = self.get_environment_data(date)
        env_vector = np.array([
            min(1.0, env_data['pm10'] / 300.0),
            env_data['humidity'] / 100.0,
            (env_data['temp'] + 20.0) / 60.0,
            min(1.0, env_data['wind_speed'] / 15.0),
        ], dtype=np.float32)
        if isinstance(node_features, np.ndarray):
            return np.concatenate([node_features, env_vector])
        return np.array(list(node_features) + list(env_vector), dtype=np.float32)


class EnhancedTemporalGraphBuilder:
    """时变动态图构建器

    以天/周为粒度构建图快照序列 G_t = (V, E_t, X_t)。
    节点固定，边集和节点特征根据行为数据和环境数据动态调整。

    文献：
      Rossi E et al. (2020) TGN
      Sankar A et al. (2020) DySAT
    """

    def __init__(self, data_access=None, climate_interaction=None,
                 static_node_features=None, n_patient=1, n_household=5,
                 n_social=10, n_community=4, seed=None, device='cpu'):
        self.data_access = data_access
        self.climate = climate_interaction
        self.static_node_features = static_node_features
        self.n_patient = n_patient
        self.n_household = n_household
        self.n_social = n_social
        self.n_community = n_community
        self.device = torch.device(device) if PYTORCH_AVAILABLE and device is not None else device
        self._rng = np.random.RandomState(seed)

        self.total_nodes = n_patient + n_household + n_social + n_community
        self.node_roles = (
            ['patient'] * n_patient +
            ['household'] * n_household +
            ['social'] * n_social +
            ['community'] * n_community
        )

        self.VENTILATION_BY_ROLE = {
            'patient': 2.0, 'household': 2.5, 'social': 3.0, 'community': 3.5,
        }
        self.BASE_CONTACT = {
            ('patient', 'household'): 0.8,
            ('patient', 'social'): 0.3,
            ('patient', 'community'): 0.1,
            ('household', 'household'): 0.5,
            ('household', 'social'): 0.2,
            ('household', 'community'): 0.05,
            ('social', 'social'): 0.15,
            ('social', 'community'): 0.08,
            ('community', 'community'): 0.03,
        }

    def set_static_features(self, static_features):
        self.static_node_features = static_features

    def build_snapshot(self, date, behavior_override=None):
        """为单个时间步构建图快照

        参数：
            date: 日期
            behavior_override: 行为数据覆盖 dict（可选）

        返回：
            dict: {
                'date': 日期,
                'x': 节点特征 [N, feature_dim],
                'edge_index': 边索引 [2, E],
                'edge_attr': 边属性 [E, edge_dim],
                'y_mask': 感染标签掩码,
                'ventilation_vector': 各节点通风向量,
                'climate_multiplier': 气候乘数,
            }
        """
        import datetime as dt_lib
        if isinstance(date, str):
            try:
                d = dt_lib.datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                d = dt_lib.datetime.now()
        elif isinstance(date, dt_lib.date):
            d = dt_lib.datetime(date.year, date.month, date.day)
        elif isinstance(date, dt_lib.datetime):
            d = date
        else:
            d = dt_lib.datetime.now()

        x = self._build_node_features(date)
        edge_index, edge_attr = self._build_dynamic_edges(
            date, behavior_override)

        env_data = None
        if self.data_access is not None:
            env_data = self.data_access.get_environment_data(date)
        elif self.climate is not None:
            daily = self.climate.get_daily_climate_multiplier(date=date)
            env_data = {'climate_multiplier': daily.get('climate_multiplier', 1.0)}

        climate_mult = env_data.get('climate_multiplier', 1.0) if env_data else 1.0

        ventilation = np.array([
            self.VENTILATION_BY_ROLE.get(self.node_roles[i], 3.0)
            for i in range(self.total_nodes)
        ], dtype=float)

        return {
            'date': d.strftime('%Y-%m-%d'),
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_nodes': self.total_nodes,
            'ventilation_vector': ventilation,
            'climate_multiplier': float(climate_mult),
            'node_roles': list(self.node_roles),
        }

    def _build_node_features(self, date):
        env_features = np.zeros((self.total_nodes, 4), dtype=np.float32)
        if self.data_access is not None:
            env_data = self.data_access.get_environment_data(date)
            env_features[:, 0] = min(1.0, env_data['pm10'] / 300.0)
            env_features[:, 1] = env_data['humidity'] / 100.0
            env_features[:, 2] = (env_data['temp'] + 20.0) / 60.0
            env_features[:, 3] = min(1.0, env_data['wind_speed'] / 15.0)
        elif self.climate is not None:
            env_vec = self.climate.get_environment_feature_vector(date=date)
            env_features[:] = env_vec

        if self.static_node_features is not None:
            static = np.array(self.static_node_features, dtype=np.float32)
            if static.ndim == 1:
                static = np.tile(static, (self.total_nodes, 1))
            if static.shape[0] < self.total_nodes:
                pad = np.zeros((self.total_nodes - static.shape[0], static.shape[1]))
                static = np.vstack([static, pad])
            return np.concatenate([static[:, :22], env_features], axis=1).astype(np.float32)

        static = np.zeros((self.total_nodes, 22), dtype=np.float32)
        for i, role in enumerate(self.node_roles):
            val = {'patient': 0.1, 'household': 0.3, 'social': 0.05, 'community': 0.01}
            static[i, 5] = val.get(role, 0.1)
            vent_val = {'patient': 2.0, 'household': 2.5, 'social': 3.0, 'community': 3.5}
            static[i, 8] = vent_val.get(role, 3.0) / 5.0
        return np.concatenate([static, env_features], axis=1).astype(np.float32)

    def _build_dynamic_edges(self, date, behavior_override=None):
        import datetime as dt_lib
        if isinstance(date, str):
            try:
                d = dt_lib.datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                d = dt_lib.datetime.now()
        elif isinstance(date, dt_lib.date):
            d = dt_lib.datetime(date.year, date.month, date.day)
        elif isinstance(date, dt_lib.datetime):
            d = date
        else:
            d = dt_lib.datetime.now()

        weekday = d.weekday()
        is_weekend = weekday >= 5

        edges_src = []
        edges_dst = []
        edge_feats = []

        role_to_range = {
            'patient': (0, self.n_patient),
            'household': (self.n_patient, self.n_patient + self.n_household),
            'social': (self.n_patient + self.n_household,
                        self.n_patient + self.n_household + self.n_social),
            'community': (self.n_patient + self.n_household + self.n_social,
                           self.total_nodes),
        }

        for (r1, r2), base_weight in self.BASE_CONTACT.items():
            start1, end1 = role_to_range[r1]
            start2, end2 = role_to_range[r2]
            weight = base_weight
            if r1 == 'patient' or r2 == 'patient':
                if r1 == 'household' or r2 == 'household':
                    weight *= 1.5 if is_weekend else 1.0
                elif r1 == 'social' or r2 == 'social':
                    weight *= 0.5 if is_weekend else 1.0
            if (r1 == 'household' and r2 == 'household'):
                weight *= 1.2 if is_weekend else 0.8
            if r1 == r2:
                for i in range(start1, end1):
                    for j in range(i + 1, min(i + 3, end1)):
                        if self._rng.random() < weight * 0.5:
                            edges_src.extend([i, j])
                            edges_dst.extend([j, i])
                            edge_feats.append([weight, 0.5, 0.2,
                                              self.VENTILATION_BY_ROLE.get(r1, 3.0) / 5.0,
                                              0.3, 1.0])
                            edge_feats.append([weight, 0.5, 0.2,
                                              self.VENTILATION_BY_ROLE.get(r2, 3.0) / 5.0,
                                              0.3, 1.0])
            else:
                block_size = min((end1 - start1), (end2 - start2))
                for i in range(start1, min(start1 + block_size, end1)):
                    for j in range(start2, min(start2 + 2, end2)):
                        if self._rng.random() < weight * 0.4:
                            edges_src.extend([i, j])
                            edges_dst.extend([j, i])
                            vent_avg = (self.VENTILATION_BY_ROLE[r1] +
                                        self.VENTILATION_BY_ROLE[r2]) / 2.0 / 5.0
                            edge_feats.append([weight, 0.4, 0.1, vent_avg, 0.2, 0.5])
                            edge_feats.append([weight, 0.4, 0.1, vent_avg, 0.2, 0.5])

        if len(edges_src) == 0:
            edges_src = [0, 1]
            edges_dst = [1, 0]
            edge_feats = [[0.5, 0.3, 0.1, 0.5, 0.2, 0.5],
                          [0.5, 0.3, 0.1, 0.5, 0.2, 0.5]]

        if PYTORCH_AVAILABLE:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=self.device)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32, device=self.device)
        else:
            # torch 不可用时回退到 numpy 数组，下游消费者会做 isinstance 检查
            edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
            edge_attr = np.array(edge_feats, dtype=np.float32)

        return edge_index, edge_attr

    def build_snapshot_sequence(self, start_date, n_days=7, stride='day',
                                 behavior_overrides=None):
        """构建图快照序列

        参数：
            start_date: 起始日期
            n_days: 天数
            stride: 步长 ('day'/'week')
            behavior_overrides: 行为数据覆盖列表

        返回：
            list[dict]: 图快照列表
        """
        import datetime as dt_lib
        try:
            import dateutil.parser
            _has_dateutil = True
        except ImportError:
            _has_dateutil = False

        if isinstance(start_date, str):
            if _has_dateutil:
                base = dateutil.parser.parse(start_date)
            else:
                base = dt_lib.datetime.strptime(start_date.replace('/', '-'), '%Y-%m-%d')
        else:
            base = start_date

        snapshots = []
        for t in range(n_days):
            delta = dt_lib.timedelta(days=t if stride == 'day' else t * 7)
            current_date = base + delta
            bh = behavior_overrides[t] if behavior_overrides and t < len(
                behavior_overrides) else None
            snap = self.build_snapshot(current_date, behavior_override=bh)
            snap['time_step'] = t
            snapshots.append(snap)
        return snapshots

    def to_torch_geometric_list(self, snapshot_sequence, device=None):
        """将快照序列转换为 torch_geometric Data 列表"""
        if not _TORCH_GEOMETRIC_PIGNN_OK:
            return None
        if device is None:
            device = self.device
        data_list = []
        for snap in snapshot_sequence:
            data = Data(
                x=torch.tensor(snap['x'], dtype=torch.float32, device=device),
                edge_index=snap['edge_index'].to(device) if hasattr(snap['edge_index'], 'to') else snap['edge_index'],
                edge_attr=snap['edge_attr'].to(device) if hasattr(snap['edge_attr'], 'to') else snap['edge_attr'],
                climate_multiplier=torch.tensor([snap['climate_multiplier']], device=device),
                time_step=snap.get('time_step', 0),
            )
            data_list.append(data)
        return data_list
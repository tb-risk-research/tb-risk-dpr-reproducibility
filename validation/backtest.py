import logging

from ..scoring.engine import ScoringEngine

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False
    np = None


LOGGER = logging.getLogger("tb_risk.validation.backtest")


def _compute_auroc(y_true, y_score):
    """计算 AUROC（ROC 曲线下面积）

    使用 Mann-Whitney U 统计量实现，无需 sklearn。
    AUROC = P(score(positive) > score(negative))
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # 无法区分时返回 0.5
    # 按分数降序排列，计算正样本排在负样本前面的比例
    order = np.argsort(-y_score, kind='stable')
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # 处理并列：使用平均秩（用 np.isclose 避免浮点精度问题）
    unique_scores, counts = np.unique(y_score, return_counts=True)
    if (counts > 1).any():
        sorted_scores = np.sort(y_score)
        for val, cnt in zip(unique_scores[counts > 1], counts[counts > 1]):
            mask = np.isclose(y_score, val, rtol=1e-12, atol=1e-12)
            min_rank = np.searchsorted(sorted_scores, val, side='left') + 1
            avg_rank = min_rank + (cnt - 1) / 2.0
            ranks[mask] = avg_rank
    sum_ranks_pos = ranks[y_true].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _compute_auprc(y_true, y_score):
    """计算 AUPRC（Precision-Recall 曲线下面积）

    使用阶梯法逐点计算并求平均，无需 sklearn。
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind='stable')
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for idx in order:
        if y_true[idx]:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return float(area)


def _compute_brier_score(y_true, y_prob):
    """计算 Brier Score（概率预测均方误差）

    BS = mean((prob - label)^2)，越低越好。
    项目中概率用 0-100 量表，此处转为 0-1 后计算。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64) / 100.0
    return float(np.mean((y_prob - y_true) ** 2))


class BacktestEngine:
    """本地数据回测引擎

    功能：
    - 运行数据集通过评分引擎
    - 计算检出率提升、早期发现率
    - 比较预测高风险排序与实际确诊病例的吻合度
    - 生成累积增益曲线数据
    """

    def __init__(self, localizer=None, use_localization=True):
        self.engine = ScoringEngine(localizer, use_localization)
        self.localizer = localizer
        self.use_localization = use_localization

    def run_backtest(self, records):
        """运行回测

        参数：
            records (list[dict]): 筛查记录列表

        返回：
            dict: 回测报告
        """
        scored = []
        for rec in records:
            result = self.engine.compute_risk_score(rec)
            scored.append({
                **rec,
                **result,
            })

        sorted_by_risk = sorted(
            scored,
            key=lambda x: x['disease_probability'],
            reverse=True,
        )

        total = len(scored)
        total_confirmed = sum(1 for r in scored if r.get('is_confirmed', 0))
        detected_at_top = {}

        thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        for thresh in thresholds:
            top_n = max(1, int(total * thresh))
            top_set = sorted_by_risk[:top_n]
            found = sum(1 for r in top_set if r.get('is_confirmed', 0))
            detected_at_top[f'top_{int(thresh*100)}pct'] = {
                'screened': top_n,
                'detected': found,
                'total_confirmed': total_confirmed,
                'detection_rate': found / max(total_confirmed, 1),
                'screening_fraction': thresh,
                'nns': top_n / max(found, 1),  # Number Needed to Screen
            }

        top100 = sorted_by_risk[:100]
        top500 = sorted_by_risk[:500]
        top1000 = sorted_by_risk[:1000]

        report = {
            'dataset_size': total,
            'total_confirmed': total_confirmed,
            'positivity_rate': total_confirmed / max(total, 1),
            'detection_at_thresholds': detected_at_top,
            'top100_detected': sum(1 for r in top100 if r.get('is_confirmed', 0)),
            'top500_detected': sum(1 for r in top500 if r.get('is_confirmed', 0)),
            'top1000_detected': sum(1 for r in top1000 if r.get('is_confirmed', 0)),
            'mean_risk_confirmed': self._mean_risk(scored, True),
            'mean_risk_non_confirmed': self._mean_risk(scored, False),
            'risk_ratio': 0.0,
            'cumulative_gain': self._compute_cumulative_gain(sorted_by_risk),
            'scored_records': scored,
            'localization_enabled': self.use_localization,
        }

        # 标准分类指标（AUROC、AUPRC、Brier Score）便于与其他模型横向比较
        if _NUMPY_OK and total > 0:
            y_true = [int(r.get('is_confirmed', 0)) for r in scored]
            y_prob = [r.get('disease_probability', 0.0) for r in scored]
            report['classification_metrics'] = {
                'auroc': _compute_auroc(y_true, y_prob),
                'auprc': _compute_auprc(y_true, y_prob),
                'brier_score': _compute_brier_score(y_true, y_prob),
            }

        if report['mean_risk_non_confirmed'] > 0:
            report['risk_ratio'] = (
                report['mean_risk_confirmed'] /
                report['mean_risk_non_confirmed']
            )

        return report

    @staticmethod
    def _mean_risk(records, confirmed_only):
        subset = [r for r in records if r.get('is_confirmed', 0) == (1 if confirmed_only else 0)]
        if not subset:
            return 0.0
        return sum(r['disease_probability'] for r in subset) / len(subset)

    @staticmethod
    def _compute_cumulative_gain(sorted_records):
        total_confirmed = sum(1 for r in sorted_records if r.get('is_confirmed', 0))
        if total_confirmed == 0:
            return []

        cumulative = 0
        gain = []
        for i, r in enumerate(sorted_records):
            if r.get('is_confirmed', 0):
                cumulative += 1
            if (i + 1) % max(1, len(sorted_records) // 50) == 0:
                gain.append({
                    'screened': i + 1,
                    'found': cumulative,
                    'found_rate': cumulative / total_confirmed,
                    'screening_fraction': (i + 1) / len(sorted_records),
                })
        return gain

    def compare_localization_gain(self, records):
        """比较本土化 vs 非本土化的增益"""
        localized = BacktestEngine(self.localizer, use_localization=True)
        baseline = BacktestEngine(self.localizer, use_localization=False)

        loc_report = localized.run_backtest(records)
        base_report = baseline.run_backtest(records)

        gains = {}
        for key in ['top100_detected', 'top500_detected', 'top1000_detected']:
            gains[key] = loc_report[key] - base_report[key]

        detection_gains = {}
        for thresh in ['top_1pct', 'top_2pct', 'top_5pct']:
            if thresh in loc_report['detection_at_thresholds']:
                loc_rate = loc_report['detection_at_thresholds'][thresh]['detection_rate']
                base_rate = base_report['detection_at_thresholds'][thresh]['detection_rate']
                detection_gains[thresh] = loc_rate - base_rate

        return {
            'localized': loc_report,
            'baseline': base_report,
            'absolute_gains': gains,
            'detection_rate_gains': detection_gains,
            'risk_ratio_localized': loc_report['risk_ratio'],
            'risk_ratio_baseline': base_report['risk_ratio'],
        }

    def run_backtest_with_time_split(self, records, time_field='date',
                                      train_ratio=0.7, gap_days=0):
        """时间分割回测（避免前瞻偏差）

        按时间顺序将数据分为训练集和测试集，在训练集上不做任何训练
        （评分引擎为无参数规则引擎），但确保测试集的时间严格晚于训练集，
        以验证模型在真实时间序列场景下的泛化能力。

        参数：
            records (list[dict]): 筛查记录，每条记录需包含 time_field 字段
            time_field (str): 时间字段名（默认 'date'）
            train_ratio (float): 训练集比例（默认 0.7）
            gap_days (int): 训练/测试集之间的间隔天数（默认 0）

        返回：
            dict: 时间分割回测报告
        """
        if not records:
            return {
                'dataset_size': 0,
                'error': '空记录集',
            }

        # 按时间排序
        try:
            sorted_records = sorted(
                records,
                key=lambda r: self._parse_time(r.get(time_field, ''))
            )
        except Exception as e:
            LOGGER.debug("时间字段解析失败，使用原始顺序: %s", e)
            # 无法解析时间字段时，使用原始顺序
            sorted_records = list(records)

        n = len(sorted_records)
        split_idx = int(n * train_ratio)

        train_records = sorted_records[:split_idx]
        test_records = sorted_records[split_idx:]

        # 如果指定了间隔天数，进一步过滤测试集
        if gap_days > 0 and train_records and test_records:
            try:
                last_train_time = self._parse_time(
                    train_records[-1].get(time_field, ''))
                if last_train_time is not None:
                    from datetime import timedelta
                    cutoff = last_train_time + timedelta(days=gap_days)
                    test_records = [
                        r for r in test_records
                        if self._parse_time(r.get(time_field, '')) is None
                        or self._parse_time(r.get(time_field, '')) >= cutoff
                    ]
            except Exception as e:
                LOGGER.warning("应用 gap_days=%d 失败: %s", gap_days, e)

        # 在各子集上运行回测
        train_report = self.run_backtest(train_records) if train_records else {}
        test_report = self.run_backtest(test_records) if test_records else {}

        # 计算时间泛化指标
        generalization_metrics = {}
        if train_report and test_report:
            train_auroc = train_report.get('classification_metrics', {}).get('auroc')
            test_auroc = test_report.get('classification_metrics', {}).get('auroc')
            if train_auroc is not None and test_auroc is not None:
                generalization_metrics['auroc_drop'] = train_auroc - test_auroc
                generalization_metrics['auroc_ratio'] = (
                    test_auroc / train_auroc if train_auroc > 0 else 0
                )

            train_dr = train_report.get('detection_at_thresholds', {}).get(
                'top_5pct', {}).get('detection_rate', 0)
            test_dr = test_report.get('detection_at_thresholds', {}).get(
                'top_5pct', {}).get('detection_rate', 0)
            generalization_metrics['detection_rate_drop'] = train_dr - test_dr

        return {
            'dataset_size': n,
            'train_size': len(train_records),
            'test_size': len(test_records),
            'train_ratio': train_ratio,
            'gap_days': gap_days,
            'time_field': time_field,
            'train_report': train_report,
            'test_report': test_report,
            'generalization_metrics': generalization_metrics,
            'temporal_validation': True,
        }

    @staticmethod
    def _parse_time(time_str):
        """解析时间字符串为 datetime 对象"""
        if not time_str:
            return None
        from datetime import datetime
        formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S', '%Y%m%d',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(str(time_str)[:19], fmt)
            except (ValueError, TypeError):
                continue
        return None

    def run_backtest_kfold(self, records, n_folds=5, stratified=True,
                            time_field=None, shuffle=False, random_seed=42):
        """k-fold 交叉验证回测。

        将数据分为 k 折，每折轮流作为验证集，其余作为训练集。
        输出各折和汇总指标，评估模型在未见数据上的泛化性能。

        参数：
            records (list[dict]): 筛查记录
            n_folds (int): 折数（默认 5）
            stratified (bool): 是否分层抽样（保持确诊比例）
            time_field (str|None): 时间字段名。若提供，按时间顺序分割
                                   （不 shuffle），确保验证集时间晚于训练集
            shuffle (bool): 是否打乱数据（time_field 为 None 时生效）
            random_seed (int): 随机种子

        返回：
            dict: k-fold 交叉验证报告
        """
        import numpy as np
        n = len(records)
        if n < n_folds:
            return {
                'error': f'样本量 ({n}) 小于折数 ({n_folds})',
                'dataset_size': n,
            }

        labels = np.array([r.get('is_confirmed', 0) for r in records])

        pos_map = None
        if time_field is not None:
            from datetime import datetime
            parsed = [(i, self._parse_time(records[i].get(time_field, '')))
                      for i in range(n)]
            sorted_indices = [i for i, t in sorted(
                parsed, key=lambda x: (x[1] is None, x[1] or datetime.min))]
            pos_map = {idx: pos for pos, idx in enumerate(sorted_indices)}
            fold_indices = []
            for k in range(n_folds):
                start = int(n * k / n_folds)
                end = int(n * (k + 1) / n_folds)
                fold_indices.append(sorted_indices[start:end])
        elif stratified and np.sum(labels) >= n_folds:
            try:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(
                    n_splits=n_folds, shuffle=shuffle,
                    random_state=random_seed)
                fold_indices = []
                indices = np.arange(n)
                for _, val_idx in skf.split(indices, labels):
                    fold_indices.append(val_idx)
            except ImportError:
                fold_indices = self._manual_stratified_split(
                    labels, n_folds, random_seed)
        elif shuffle:
            rng = np.random.RandomState(random_seed)
            indices = rng.permutation(n)
            fold_indices = np.array_split(indices, n_folds)
        else:
            fold_indices = np.array_split(np.arange(n), n_folds)

        fold_results = []
        all_aurocs = []
        all_detection_rates = []

        for fold_idx, val_indices in enumerate(fold_indices):
            val_indices = list(val_indices)
            if time_field is not None:
                val_start_pos = min(pos_map[i] for i in val_indices)
                train_indices = sorted_indices[:val_start_pos]
            else:
                train_indices = [i for i in range(n) if i not in val_indices]
            val_records = [records[i] for i in val_indices]
            val_report = self.run_backtest(val_records) if val_records else {}

            auroc = val_report.get('classification_metrics', {}).get('auroc')
            if auroc is not None:
                all_aurocs.append(auroc)

            top5_dr = val_report.get('detection_at_thresholds', {}).get(
                'top_5pct', {}).get('detection_rate', 0)
            all_detection_rates.append(top5_dr)

            fold_results.append({
                'fold': fold_idx + 1,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'val_confirmed': int(sum(labels[val_indices])),
                'auroc': auroc,
                'detection_rate_top5pct': top5_dr,
            })

        summary = {
            'n_folds': n_folds,
            'dataset_size': n,
            'stratified': stratified,
            'time_based': time_field is not None,
        }

        if all_aurocs:
            auroc_arr = np.array(all_aurocs)
            summary['auroc_mean'] = float(np.mean(auroc_arr))
            summary['auroc_std'] = float(np.std(auroc_arr, ddof=1))
            summary['auroc_cv'] = float(
                np.std(auroc_arr, ddof=1) / max(np.mean(auroc_arr), 0.001))

        if all_detection_rates:
            dr_arr = np.array(all_detection_rates)
            summary['detection_rate_mean'] = float(np.mean(dr_arr))
            summary['detection_rate_std'] = float(np.std(dr_arr, ddof=1))

        return {
            'summary': summary,
            'folds': fold_results,
            'kfold_validation': True,
        }

    @staticmethod
    def _manual_stratified_split(labels, n_folds, random_seed):
        """手动分层分割（无 sklearn 时的回退方案）。"""
        import numpy as np
        rng = np.random.RandomState(random_seed)
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)
        pos_folds = np.array_split(pos_indices, n_folds)
        neg_folds = np.array_split(neg_indices, n_folds)
        folds = []
        for i in range(n_folds):
            folds.append(np.concatenate([pos_folds[i], neg_folds[i]]))
        return folds
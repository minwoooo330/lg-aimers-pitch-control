"""Step 2-1: 고카디널리티 ID 컬럼(pitcher_id, batter_id) 인코딩 실험용 유틸.

주의(누수 방지):
- frequency encoding: 타깃을 쓰지 않으므로 train 전체 통계를 그대로 valid/test에 적용해도 무해.
- target encoding: train 자기 자신에 대해서는 K-fold OOF로 계산해야 한다(자기 행의 정답이
  자기 인코딩 값에 들어가면 누수). valid/test에는 train 전체로 만든 매핑을 그대로 적용한다
  (valid/test는 애초에 그 통계 계산에 쓰이지 않으므로 추가 조치 불필요).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def frequency_encode(train_col: pd.Series, apply_col: pd.Series) -> pd.Series:
    counts = train_col.value_counts()
    return apply_col.map(counts).fillna(0).astype("float64")


def _smoothed_mean(group_sum: pd.Series, group_count: pd.Series, global_mean: float, smoothing: float) -> pd.Series:
    return (group_sum + smoothing * global_mean) / (group_count + smoothing)


def oof_target_encode(
    train_col: pd.Series,
    train_target: pd.Series,
    n_splits: int = 5,
    smoothing: float = 20.0,
    seed: int = 42,
) -> pd.Series:
    """train 파티션 내부 OOF target encoding (K-fold, 무작위 fold)."""
    oof = pd.Series(index=train_col.index, dtype="float64")
    global_mean = train_target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fit_idx, hold_idx in kf.split(train_col):
        fit_col = train_col.iloc[fit_idx]
        fit_target = train_target.iloc[fit_idx]
        stats = fit_target.groupby(fit_col).agg(["sum", "count"])
        mapping = _smoothed_mean(stats["sum"], stats["count"], global_mean, smoothing)

        hold_col = train_col.iloc[hold_idx]
        oof.iloc[hold_idx] = hold_col.map(mapping).fillna(global_mean).to_numpy()

    return oof


def fit_target_encode_map(train_col: pd.Series, train_target: pd.Series, smoothing: float = 20.0) -> tuple[pd.Series, float]:
    """train 전체로 만든 매핑(valid/test 적용용)과 global_mean을 반환."""
    global_mean = train_target.mean()
    stats = train_target.groupby(train_col).agg(["sum", "count"])
    mapping = _smoothed_mean(stats["sum"], stats["count"], global_mean, smoothing)
    return mapping, global_mean


def apply_target_encode_map(apply_col: pd.Series, mapping: pd.Series, global_mean: float) -> pd.Series:
    return apply_col.map(mapping).fillna(global_mean).astype("float64")

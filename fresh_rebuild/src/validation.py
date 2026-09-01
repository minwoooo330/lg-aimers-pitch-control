"""
검증 하네스 (Step 0)

설계 근거 (fresh/BRIEF.md + 자체 EDA):
- train.csv season x game_type 성공률을 직접 집계한 결과, 퓨처스(F)는 2023년에
  0.70대 -> 0.47대로 급락한다. KBO는 ABS(자동 볼판정)를 퓨처스 2023년, 1군 2024년부터
  도입했으므로 이 급락은 심판 판정 -> 기계 판정 전환에 의한 라벨 체제 변화로 해석한다.
- 2025 평가 데이터는 두 리그 모두 ABS 체제이므로, 검증 fold도 "그 시점에 ABS 체제였는가"를
  기준으로 대표성을 판단한다.

fold 구성:
- main   : train season<=2023 (전체 리그) -> valid season==2024
           2024는 R/F 모두 ABS 체제라 2025를 가장 잘 대표. 모델/피처 선택의 주 판정 기준.
- ref    : train season<=2021 -> valid season==2022
           둘 다 ABS 이전 체제라 2025 대표성은 없음. 안정성(붕괴 여부) 점검용 보조 지표일 뿐,
           이 결과로 피처/모델을 선택하지 않는다.

비교 원칙:
- 모델/피처 비교는 mean-aligned Brier(예측 평균을 실제 평균에 맞춘 뒤 채점)를 기본으로 쓴다.
  평균 보정(calibration) 효과와 모델 자체의 개선을 분리하기 위함이다.
- 최종 제출용 보정(shift)은 이 모듈이 아니라 별도 단계에서 전체 학습 데이터로 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FOLD_DEFS = {
    "main": {"train_end_season": 2023, "valid_season": 2024},
    "ref": {"train_end_season": 2021, "valid_season": 2022},
}


@dataclass
class FoldSplit:
    name: str
    train_end_season: int
    valid_season: int
    train_idx: np.ndarray
    valid_idx: np.ndarray


def make_fold(df: pd.DataFrame, name: str) -> FoldSplit:
    """df는 최소 'season' 컬럼을 가진 train.csv 로드 결과."""
    spec = FOLD_DEFS[name]
    train_end = spec["train_end_season"]
    valid_season = spec["valid_season"]

    train_mask = df["season"] <= train_end
    valid_mask = df["season"] == valid_season

    return FoldSplit(
        name=name,
        train_end_season=train_end,
        valid_season=valid_season,
        train_idx=df.index[train_mask].to_numpy(),
        valid_idx=df.index[valid_mask].to_numpy(),
    )


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def mean_aligned_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """예측 평균을 실제 평균에 맞춰 상수 이동한 뒤 Brier를 계산한다.
    (검증 fold의 실제 평균만 사용 — 대회 규정상 test 정보 미사용, 여기선 train 파생 valid라 무해)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    shift = y_true.mean() - y_pred.mean()
    y_pred_aligned = np.clip(y_pred + shift, 0.0, 1.0)
    return brier_score(y_true, y_pred_aligned)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "n": int(len(y_true)),
        "true_mean": float(y_true.mean()),
        "pred_mean": float(y_pred.mean()),
        "brier_raw": brier_score(y_true, y_pred),
        "brier_mean_aligned": mean_aligned_brier(y_true, y_pred),
    }


def summarize_fold(df: pd.DataFrame, fold: FoldSplit) -> dict:
    train_df = df.loc[fold.train_idx]
    valid_df = df.loc[fold.valid_idx]
    return {
        "fold": fold.name,
        "train_seasons": f"<={fold.train_end_season}",
        "valid_season": fold.valid_season,
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "valid_success_rate": float(valid_df["control_success"].mean()),
        "valid_game_type_counts": valid_df["game_type"].value_counts().to_dict(),
    }

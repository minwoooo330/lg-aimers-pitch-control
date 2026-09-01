"""Step 4-4: 최종 제출 모델 정의.

확정 사항 요약 (Step2~4 실험 근거):
- 모델: CatBoost (Step3-1: HGB 대비 main +277e-5, ref +169e-5 압도적 우세)
- ID 처리: pitcher_id/batter_id는 raw 숫자 (Step2-1, Step3-2에서 두 번 확인: native categorical/
  freq/target encoding 전부 raw보다 나쁨)
- 하이퍼파라미터: 기본값 (Step3-3: 5개 후보 탐색, 두 fold에서 일관되게 이기는 후보 없음)
- 앙상블: 5시드 평균 (Step4-1: main +2.9e-5, ref +3.5e-5 개선, 공짜 이득이라 채택)
- 다양성 블렌딩: LightGBM 등 추가 안 함 (Step4-2: 오차 상관 0.90~0.98로 높아 이득 없음)
- 피처: features.py의 47원본 + trend_gap5_success 1개 파생 (Step2-4 유일 채택)
- 확률 보정: shift = -0.006689 (Step4-3/4-3b: 4개 walk-forward 지점 편향의 단순평균,
  leave-one-out 검증으로 방법론 확정. 표준편차 0.0118로 불확실성 큼 - 인지하고 채택)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import CATEGORICAL_COLS, build_categories, prepare_features

CATBOOST_PARAMS = dict(iterations=200, learning_rate=0.06, depth=6, l2_leaf_reg=3.0)
SEEDS = [42, 7, 2024, 13, 77]
FINAL_SHIFT = -0.006689


def train_ensemble(train_df: pd.DataFrame, categories: dict, model_dir: Path | None = None) -> list[CatBoostClassifier]:
    X_train = prepare_features(train_df, categories)
    y_train = train_df["control_success"].to_numpy()

    models = []
    for seed in SEEDS:
        model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed, verbose=0, cat_features=CATEGORICAL_COLS)
        model.fit(X_train, y_train)
        models.append(model)
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_dir / f"catboost_seed{seed}.cbm"))
    return models


def load_ensemble(model_dir: Path) -> list[CatBoostClassifier]:
    models = []
    for seed in SEEDS:
        model = CatBoostClassifier()
        model.load_model(str(model_dir / f"catboost_seed{seed}.cbm"))
        models.append(model)
    return models


def predict_ensemble(models: list[CatBoostClassifier], X: pd.DataFrame) -> np.ndarray:
    preds = np.zeros(len(X), dtype=np.float64)
    for model in models:
        preds += model.predict_proba(X)[:, 1]
    preds /= len(models)
    return preds


def predict_final(models: list[CatBoostClassifier], df: pd.DataFrame, categories: dict) -> np.ndarray:
    """평균 앙상블 예측 + shift 적용 + [0,1] clip. 최종 제출값 산출 함수."""
    X = prepare_features(df, categories)
    raw_pred = predict_ensemble(models, X)
    shifted = np.clip(raw_pred + FINAL_SHIFT, 0.0, 1.0)
    return shifted

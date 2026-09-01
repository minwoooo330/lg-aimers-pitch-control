"""Step 2-1: pitcher_id / batter_id 인코딩 방식 비교.

가설: asof_pitcher_success_rate 등 결과 기반 이력 피처가 이미 있는 상태에서,
ID 자체를 어떻게 넣느냐(raw 숫자 / 제거 / 빈도 / OOF target encoding)가
성능에 유의미한 차이를 만드는가?

비교 변형:
  v0_raw    : 현재 베이스라인 (원시 ID 숫자 그대로) - Step1과 동일
  v1_drop   : pitcher_id / batter_id 완전 제거
  v2_freq   : 빈도 인코딩 (train 파티션 등장 횟수)
  v3_target : OOF smoothed target encoding (K=5, smoothing=20)

판정 기준: main fold(2024)를 주 판정, ref fold(2022)는 참고용. mean-aligned Brier 사용.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from encoders import (
    apply_target_encode_map,
    fit_target_encode_map,
    frequency_encode,
    oof_target_encode,
)
from features import CATEGORICAL_COLS, NUMERIC_COLS, build_categories
from validation import FOLD_DEFS, evaluate, make_fold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

ID_COLS = ["pitcher_id", "batter_id"]
BASE_NUMERIC_COLS = [c for c in NUMERIC_COLS if c not in ID_COLS]

HGB_PARAMS = dict(
    max_iter=200,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    random_state=42,
)


def build_base_X(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    out = df[CATEGORICAL_COLS + BASE_NUMERIC_COLS].copy()
    for c in CATEGORICAL_COLS:
        out[c] = out[c].astype(str).astype(categories[c])
    for c in BASE_NUMERIC_COLS:
        out[c] = out[c].astype("float64")
    return out


def build_variant(variant: str, train_df: pd.DataFrame, valid_df: pd.DataFrame, base_cat_mask: list[bool]):
    X_train = build_base_X(train_df, CATS)
    X_valid = build_base_X(valid_df, CATS)
    cat_mask = list(base_cat_mask)

    if variant == "v1_drop":
        pass  # ID 컬럼 추가 안 함

    elif variant == "v0_raw":
        for c in ID_COLS:
            X_train[c] = train_df[c].astype("float64").to_numpy()
            X_valid[c] = valid_df[c].astype("float64").to_numpy()
            cat_mask.append(False)

    elif variant == "v2_freq":
        for c in ID_COLS:
            X_train[c] = frequency_encode(train_df[c], train_df[c]).to_numpy()
            X_valid[c] = frequency_encode(train_df[c], valid_df[c]).to_numpy()
            cat_mask.append(False)

    elif variant == "v3_target":
        y_train = train_df["control_success"]
        for c in ID_COLS:
            X_train[c] = oof_target_encode(train_df[c], y_train, n_splits=5, smoothing=20.0, seed=42).to_numpy()
            mapping, global_mean = fit_target_encode_map(train_df[c], y_train, smoothing=20.0)
            X_valid[c] = apply_target_encode_map(valid_df[c], mapping, global_mean).to_numpy()
            cat_mask.append(False)

    else:
        raise ValueError(variant)

    return X_train, X_valid, cat_mask


def main():
    global CATS
    print("데이터 로드...", flush=True)
    df = pd.read_csv(DATA_DIR / "train.csv")
    CATS = build_categories(df)
    base_cat_mask = [True] * len(CATEGORICAL_COLS) + [False] * len(BASE_NUMERIC_COLS)

    variants = ["v0_raw", "v1_drop", "v2_freq", "v3_target"]
    rows = []

    for fold_name in FOLD_DEFS:
        fold = make_fold(df, fold_name)
        train_df = df.loc[fold.train_idx]
        valid_df = df.loc[fold.valid_idx]
        y_valid = valid_df["control_success"].to_numpy()

        for variant in variants:
            t0 = time.time()
            X_train, X_valid, cat_mask = build_variant(variant, train_df, valid_df, base_cat_mask)
            y_train = train_df["control_success"].to_numpy()

            model = HistGradientBoostingClassifier(categorical_features=cat_mask, **HGB_PARAMS)
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_valid)[:, 1]
            result = evaluate(y_valid, pred)
            dt = time.time() - t0

            rows.append({
                "fold": fold_name,
                "variant": variant,
                "n_features": X_train.shape[1],
                "brier_raw": result["brier_raw"],
                "brier_mean_aligned": result["brier_mean_aligned"],
                "pred_mean": result["pred_mean"],
                "time_s": dt,
            })
            print(f"[{fold_name}] {variant}: brier_mean_aligned={result['brier_mean_aligned']:.8f}  ({dt:.1f}s)", flush=True)

    res_df = pd.DataFrame(rows)
    print()
    print(res_df.to_string(index=False))

    out_path = ROOT / "results"
    out_path.mkdir(exist_ok=True)
    res_df.to_csv(out_path / "02_encoder_experiment.csv", index=False)
    print(f"\n저장: {out_path / '02_encoder_experiment.csv'}")


if __name__ == "__main__":
    main()

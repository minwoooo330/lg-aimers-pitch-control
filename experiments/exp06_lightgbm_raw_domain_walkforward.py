# -*- coding: utf-8 -*-
"""실험 06: LightGBM 원본/도메인 피처 공통 시간 검증."""
from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from features import add_features

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "train.csv"
RESULT_DIR = HERE / "results"
RESULT_PATH = RESULT_DIR / "exp06_lightgbm_raw_domain_walkforward.csv"
ID, TARGET = "row_id", "control_success"
FOLDS = [("2022", 2022), ("2023", 2023), ("2024", 2024)]
CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
]
CONFIGS = [
    ("raw_47", False, RESULT_DIR / "exp06_lightgbm_raw_oof.csv.gz"),
    ("raw_47_plus_domain_43", True, RESULT_DIR / "exp06_lightgbm_domain_oof.csv.gz"),
]


def build_train_valid(train, valid, base_cols, use_domain):
    """범주 대응표는 각 시험 연도를 보지 않고 학습 구간에서만 만든다."""
    x_train = train[base_cols].copy()
    x_valid = valid[base_cols].copy()
    for col in CAT_COLS:
        values = sorted(train[col].dropna().astype(str).unique())
        mapping = {value: code for code, value in enumerate(values)}
        x_train[col] = train[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
        x_valid[col] = valid[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
    if use_domain:
        x_train = pd.concat([x_train, add_features(train)], axis=1)
        x_valid = pd.concat([x_valid, add_features(valid)], axis=1)
    return x_train, x_valid


def metric_row(features, fold, train_end, y_train, y_valid, pred, seconds, trees):
    return {
        "model": "lightgbm", "features": features, "fold": fold,
        "train_end_year": train_end, "n_train": len(y_train), "n_valid": len(y_valid),
        "train_target_mean": float(np.mean(y_train)),
        "valid_target_mean": float(np.mean(y_valid)), "pred_mean": float(np.mean(pred)),
        "brier": brier_score_loss(y_valid, pred),
        "logloss": log_loss(y_valid, pred, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_valid, pred),
        "seconds": seconds, "trees": trees,
    }


def main():
    started = time.time()
    print(f"데이터 로드: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    base_cols = [c for c in df.columns if c not in (ID, TARGET)]
    rows = []
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for feature_name, use_domain, oof_path in CONFIGS:
        print("\n" + "=" * 72)
        print(f"LightGBM 설정: {feature_name}")
        print("=" * 72)
        oof_parts = []
        for fold_name, valid_year in FOLDS:
            fold_started = time.time()
            train = df[df["season"] < valid_year]
            valid = df[df["season"] == valid_year]
            y_train = train[TARGET].to_numpy(np.int8)
            y_valid = valid[TARGET].to_numpy(np.int8)
            x_train, x_valid = build_train_valid(train, valid, base_cols, use_domain)
            print(f"[{feature_name}/{fold_name}] 학습 {len(train):,} -> 시험 {len(valid):,}, 피처 {x_train.shape[1]}")

            model = LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=200,
                reg_lambda=1.0,
                max_bin=255,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
            fit_started = time.time()
            model.fit(x_train, y_train, categorical_feature=CAT_COLS)
            pred = model.predict_proba(x_valid)[:, 1]
            seconds = time.time() - fit_started
            row = metric_row(feature_name, fold_name, valid_year - 1,
                             y_train, y_valid, pred, seconds, model.n_estimators_)
            rows.append(row)
            oof_parts.append(pd.DataFrame({
                ID: valid[ID].to_numpy(), "season": valid_year,
                TARGET: y_valid, "prediction": pred,
            }))
            print(f"Brier={row['brier']:.6f}, LogLoss={row['logloss']:.6f}, "
                  f"AUC={row['roc_auc']:.6f}, 시간={seconds:.1f}초")
            del train, valid, x_train, x_valid, model, pred
            gc.collect()
            print(f"fold 전체={time.time()-fold_started:.1f}초")

        pd.concat(oof_parts, ignore_index=True).to_csv(
            oof_path, index=False, encoding="utf-8", compression="gzip"
        )
        del oof_parts
        gc.collect()

    results = pd.DataFrame(rows)
    summary = (results.groupby(["model", "features"], as_index=False)
               .agg(brier=("brier", "mean"), logloss=("logloss", "mean"),
                    roc_auc=("roc_auc", "mean"), seconds=("seconds", "sum")))
    summary["fold"] = "mean_2022_2024"
    for col in results.columns:
        if col not in summary.columns:
            summary[col] = np.nan
    results = pd.concat([results, summary[results.columns]], ignore_index=True)
    results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print("\n=== 최종 결과 ===")
    print(results[["model", "features", "fold", "brier", "logloss", "roc_auc", "seconds"]].to_string(index=False))
    print(f"결과: {RESULT_PATH}")
    print(f"총 소요: {time.time()-started:.1f}초")


if __name__ == "__main__":
    main()

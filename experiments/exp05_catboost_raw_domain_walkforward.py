# -*- coding: utf-8 -*-
"""실험 05: CatBoost 원본/도메인 피처 공통 시간 검증.

익명 선수·팀 ID와 작은 코드형 변수를 연속 숫자가 아닌 서로 다른 종류로 처리한다.
각 회차는 과거 연도만 학습하고 바로 다음 연도를 시험한다.
"""
from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from features import add_features

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "train.csv"
RESULT_DIR = HERE / "results"
RESULT_PATH = RESULT_DIR / "exp05_catboost_raw_domain_walkforward.csv"
ID, TARGET = "row_id", "control_success"
FOLDS = [("2022", 2022), ("2023", 2023), ("2024", 2024)]
# 숫자 크기에 의미가 없는 식별자/종류 코드
CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
]
CONFIGS = [
    ("raw_47", False, RESULT_DIR / "exp05_catboost_raw_oof.csv.gz"),
    ("raw_47_plus_domain_43", True, RESULT_DIR / "exp05_catboost_domain_oof.csv.gz"),
]


def build_x(frame, base_cols, use_domain):
    x = frame[base_cols].copy()
    # CatBoost 범주형 값은 결측 없는 문자열 이름표로 전달한다.
    for col in CAT_COLS:
        x[col] = frame[col].where(frame[col].notna(), "__MISSING__").astype(str)
    if use_domain:
        x = pd.concat([x, add_features(frame)], axis=1)
    return x


def metrics(model_name, features, fold, train_end, y_train, y_valid, pred, seconds, iterations):
    return {
        "model": model_name, "features": features, "fold": fold,
        "train_end_year": train_end, "n_train": len(y_train), "n_valid": len(y_valid),
        "train_target_mean": float(np.mean(y_train)),
        "valid_target_mean": float(np.mean(y_valid)), "pred_mean": float(np.mean(pred)),
        "brier": brier_score_loss(y_valid, pred),
        "logloss": log_loss(y_valid, pred, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_valid, pred),
        "seconds": seconds, "iterations": iterations,
    }


def main():
    total_started = time.time()
    print(f"데이터 로드: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    base_cols = [c for c in df.columns if c not in (ID, TARGET)]
    rows = []
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for feature_name, use_domain, oof_path in CONFIGS:
        print("\n" + "=" * 72)
        print(f"CatBoost 설정: {feature_name}")
        print("=" * 72)
        oof_parts = []
        for fold_name, valid_year in FOLDS:
            fold_started = time.time()
            train = df[df["season"] < valid_year]
            valid = df[df["season"] == valid_year]
            y_train = train[TARGET].to_numpy(np.int8)
            y_valid = valid[TARGET].to_numpy(np.int8)
            x_train = build_x(train, base_cols, use_domain)
            x_valid = build_x(valid, base_cols, use_domain)
            cat_indices = [x_train.columns.get_loc(c) for c in CAT_COLS]
            print(f"[{feature_name}/{fold_name}] 학습 {len(train):,} -> 시험 {len(valid):,}, 피처 {x_train.shape[1]}")

            constant = np.full(len(valid), y_train.mean(), dtype=np.float64)
            rows.append(metrics("constant", feature_name, fold_name, valid_year - 1,
                                y_train, y_valid, constant, 0.0, 0))

            model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.06,
                depth=8,
                loss_function="Logloss",
                l2_leaf_reg=3.0,
                random_seed=42,
                thread_count=-1,
                allow_writing_files=False,
                verbose=50,
            )
            fit_started = time.time()
            model.fit(x_train, y_train, cat_features=cat_indices)
            pred = model.predict_proba(x_valid)[:, 1]
            seconds = time.time() - fit_started
            row = metrics("catboost", feature_name, fold_name, valid_year - 1,
                          y_train, y_valid, pred, seconds, model.tree_count_)
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
    print(f"총 소요: {time.time()-total_started:.1f}초")


if __name__ == "__main__":
    main()

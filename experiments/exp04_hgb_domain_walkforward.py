# -*- coding: utf-8 -*-
"""실험 04: 원본 47개 + 야구 도메인 가공 피처 HGB 공통 시간 검증.

각 회차(fold)에서 과거 연도로만 학습하고 바로 다음 연도를 시험한다.
features.py의 행별 가공 피처를 사용하며 test/validation 다른 행은 참조하지 않는다.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from features import add_features

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "train.csv"
RESULT_DIR = HERE / "results"
RESULT_PATH = RESULT_DIR / "exp04_hgb_domain_walkforward.csv"
OOF_PATH = RESULT_DIR / "exp04_hgb_domain_oof.csv.gz"

ID = "row_id"
TARGET = "control_success"
# 종류를 나타내는 작은 범주형 컬럼. 익명 선수/팀 ID는 이번 '원본 기준선'에서는
# 기존 코드와의 비교를 위해 숫자 그대로 두며, 다음 실험에서 별도로 개선한다.
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
FOLDS = [
    ("2022", 2022),  # 2019~2021 학습 -> 2022 시험
    ("2023", 2023),  # 2019~2022 학습 -> 2023 시험
    ("2024", 2024),  # 2019~2023 학습 -> 2024 시험
]


def encode_train_valid(train, valid, feature_cols):
    """범주값 대응표는 시험 연도를 보지 않고 학습 데이터에서만 만든다."""
    x_train = train[feature_cols].copy()
    x_valid = valid[feature_cols].copy()
    for col in CAT_COLS:
        values = sorted(train[col].dropna().astype(str).unique())
        mapping = {value: code for code, value in enumerate(values)}
        x_train[col] = train[col].astype(str).map(mapping).fillna(-1).astype(np.int16)
        x_valid[col] = valid[col].astype(str).map(mapping).fillna(-1).astype(np.int16)
    # 공식 원본에 현재 행만으로 계산한 야구 가공 피처를 추가한다.
    x_train = pd.concat([x_train, add_features(train)], axis=1)
    x_valid = pd.concat([x_valid, add_features(valid)], axis=1)
    return x_train, x_valid


def metric_row(model_name, fold_name, train_end, y_train, y_valid, pred, seconds, n_iter):
    """모든 팀원이 같은 열 이름으로 결과를 저장할 수 있게 한 행을 만든다."""
    return {
        "model": model_name,
        "features": "raw_47_plus_domain_43",
        "fold": fold_name,
        "train_end_year": train_end,
        "n_train": len(y_train),
        "n_valid": len(y_valid),
        "train_target_mean": float(np.mean(y_train)),
        "valid_target_mean": float(np.mean(y_valid)),
        "pred_mean": float(np.mean(pred)),
        "brier": brier_score_loss(y_valid, pred),
        "logloss": log_loss(y_valid, pred, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_valid, pred),
        "seconds": seconds,
        "n_iter": n_iter,
    }


def main():
    started = time.time()
    print(f"데이터 로드: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    feature_cols = [c for c in df.columns if c not in (ID, TARGET)]
    print(f"전체 {len(df):,}행, 원본 입력 {len(feature_cols)}개 + 가공 피처 43개")

    rows = []
    oof_parts = []
    for fold_name, valid_year in FOLDS:
        fold_started = time.time()
        train = df[df["season"] < valid_year]
        valid = df[df["season"] == valid_year]
        y_train = train[TARGET].to_numpy(np.int8)
        y_valid = valid[TARGET].to_numpy(np.int8)
        print(f"\n[{fold_name}] 학습 {len(train):,}행 -> 시험 {len(valid):,}행")

        x_train, x_valid = encode_train_valid(train, valid, feature_cols)
        categorical_mask = [c in CAT_COLS for c in x_train.columns]

        # 가장 단순한 비교 기준: 학습 기간 평균을 모든 시험 행에 똑같이 예측
        constant_pred = np.full(len(valid), y_train.mean(), dtype=np.float64)
        rows.append(metric_row(
            "constant", fold_name, valid_year - 1, y_train, y_valid,
            constant_pred, 0.0, 0,
        ))

        # 시간 순서를 흐리는 무작위 조기 종료를 피하기 위해 반복 횟수를 200으로 고정한다.
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=False,
            categorical_features=categorical_mask,
            random_state=42,
        )
        fit_started = time.time()
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_valid)[:, 1]
        fit_seconds = time.time() - fit_started
        rows.append(metric_row(
            "hgb", fold_name, valid_year - 1, y_train, y_valid,
            pred, fit_seconds, model.n_iter_,
        ))
        oof_parts.append(pd.DataFrame({
            ID: valid[ID].to_numpy(),
            "season": valid_year,
            TARGET: y_valid,
            "prediction": pred,
        }))
        print(
            f"HGB Brier={rows[-1]['brier']:.6f}, "
            f"LogLoss={rows[-1]['logloss']:.6f}, AUC={rows[-1]['roc_auc']:.6f}, "
            f"학습+예측={fit_seconds:.1f}초"
        )
        print(f"fold 전체 소요={time.time()-fold_started:.1f}초")
        del train, valid, x_train, x_valid, model

    results = pd.DataFrame(rows)
    # 모델별 3개 시험 연도의 단순 평균 행도 추가한다.
    summary = (results.groupby(["model", "features"], as_index=False)
               .agg(brier=("brier", "mean"),
                    logloss=("logloss", "mean"),
                    roc_auc=("roc_auc", "mean"),
                    seconds=("seconds", "sum")))
    summary["fold"] = "mean_2022_2024"
    for col in results.columns:
        if col not in summary.columns:
            summary[col] = np.nan
    results = pd.concat([results, summary[results.columns]], ignore_index=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    pd.concat(oof_parts, ignore_index=True).to_csv(
        OOF_PATH, index=False, encoding="utf-8", compression="gzip"
    )
    print("\n=== 결과표 ===")
    print(results[["model", "features", "fold", "brier", "logloss", "roc_auc", "seconds"]].to_string(index=False))
    print(f"\n결과: {RESULT_PATH}")
    print(f"OOF 예측: {OOF_PATH}")
    print(f"총 소요: {time.time()-started:.1f}초")


if __name__ == "__main__":
    main()

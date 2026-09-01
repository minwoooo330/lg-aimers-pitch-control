"""Step 3-2: CatBoost 승리 원인 진단 + 재현성 확인.

가설: CatBoost의 승리가 (a) 알고리즘 자체(ordered boosting)의 우위인지,
(b) pitcher_id/batter_id를 raw 숫자가 아니라 native categorical(ordered target stat)로
처리해서인지 분리한다. 3시드로 우연이 아닌지도 확인한다.

variant native_id : pitcher_id/batter_id를 CATEGORICAL_COLS에 포함 (진짜 범주형)
variant raw_id    : 현재 HGB와 동일하게 raw 숫자로 (기존 08 스크립트와 동일 설정)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import CATEGORICAL_COLS, build_categories, prepare_features
from validation import FOLD_DEFS, evaluate, make_fold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEEDS = [42, 7, 2024]
ID_COLS = ["pitcher_id", "batter_id"]


def fit_predict_catboost(X_train, y_train, X_valid, cat_cols, seed):
    Xt = X_train.copy()
    Xv = X_valid.copy()
    for c in cat_cols:
        Xt[c] = Xt[c].astype(str)
        Xv[c] = Xv[c].astype(str)
    model = CatBoostClassifier(
        iterations=200, learning_rate=0.06, depth=6, l2_leaf_reg=3.0,
        random_seed=seed, verbose=0, cat_features=cat_cols,
    )
    model.fit(Xt, y_train)
    return model.predict_proba(Xv)[:, 1]


def main():
    print("데이터 로드...", flush=True)
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)

    rows = []
    for fold_name in FOLD_DEFS:
        fold = make_fold(df, fold_name)
        train_df = df.loc[fold.train_idx]
        valid_df = df.loc[fold.valid_idx]
        y_train = train_df["control_success"].to_numpy()
        y_valid = valid_df["control_success"].to_numpy()

        X_train = prepare_features(train_df, categories)
        X_valid = prepare_features(valid_df, categories)

        for variant, cat_cols in [("raw_id", CATEGORICAL_COLS), ("native_id", CATEGORICAL_COLS + ID_COLS)]:
            diffs = []
            for seed in SEEDS:
                t0 = time.time()
                pred = fit_predict_catboost(X_train, y_train, X_valid, cat_cols, seed)
                result = evaluate(y_valid, pred)
                dt = time.time() - t0
                rows.append({"fold": fold_name, "variant": variant, "seed": seed,
                             "brier_mean_aligned": result["brier_mean_aligned"], "time_s": dt})
                print(f"[{fold_name}] {variant} seed={seed}: brier_mean_aligned={result['brier_mean_aligned']:.8f} ({dt:.1f}s)", flush=True)

    res_df = pd.DataFrame(rows)
    print()
    summary = res_df.groupby(["fold", "variant"])["brier_mean_aligned"].agg(["mean", "std"])
    print(summary)
    out_path = ROOT / "results"
    out_path.mkdir(exist_ok=True)
    res_df.to_csv(out_path / "09_catboost_diagnosis.csv", index=False)


if __name__ == "__main__":
    main()

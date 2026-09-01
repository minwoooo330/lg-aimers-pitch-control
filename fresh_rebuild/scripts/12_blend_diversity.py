"""Step 4-2: 다양성 모델(LightGBM) 블렌딩 검증.

CatBoost 5시드 평균(11_seed_ensemble.py 결과 재사용)에 LightGBM을 소폭 섞어
오차 상관이 낮아 앙상블 이득이 있는지 확인한다. 블렌딩 비중은 0%~30% 격자로
두 fold 모두에서 개선되는 비중만 인정한다(단일 fold 특이 개선 배제).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb

from features import CATEGORICAL_COLS, build_categories, prepare_features
from validation import FOLD_DEFS, evaluate, make_fold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 42


def fit_predict_lgbm(X_train, y_train, X_valid, seed):
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.06, num_leaves=31, min_child_samples=200,
        random_state=seed, verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)
    return model.predict_proba(X_valid)[:, 1]


def main():
    print("데이터 로드...", flush=True)
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)

    for fold_name in FOLD_DEFS:
        fold = make_fold(df, fold_name)
        train_df = df.loc[fold.train_idx]
        valid_df = df.loc[fold.valid_idx]
        y_train = train_df["control_success"].to_numpy()

        seed_pred_df = pd.read_csv(RESULTS_DIR / f"11_seed_preds_{fold_name}.csv.gz")
        seed_cols = [c for c in seed_pred_df.columns if c.startswith("seed_")]
        cat_avg_pred = seed_pred_df[seed_cols].mean(axis=1).to_numpy()
        y_valid = seed_pred_df["y_true"].to_numpy()

        X_train = prepare_features(train_df, categories)
        X_valid = prepare_features(valid_df, categories)
        lgbm_pred = fit_predict_lgbm(X_train, y_train, X_valid, SEED)

        corr = np.corrcoef(cat_avg_pred, lgbm_pred)[0, 1]
        b_cat = evaluate(y_valid, cat_avg_pred)["brier_mean_aligned"]
        b_lgbm = evaluate(y_valid, lgbm_pred)["brier_mean_aligned"]
        print(f"[{fold_name}] catboost5seed={b_cat:.8f}  lightgbm={b_lgbm:.8f}  corr={corr:.4f}")

        for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
            blend = (1 - w) * cat_avg_pred + w * lgbm_pred
            b = evaluate(y_valid, blend)["brier_mean_aligned"]
            print(f"  w_lgbm={w:.2f}: brier_mean_aligned={b:.8f}")
        print()


if __name__ == "__main__":
    main()

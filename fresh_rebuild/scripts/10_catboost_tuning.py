"""Step 3-3: CatBoost 경량 하이퍼파라미터 탐색.

Step2 외부 인사이트(무리한 튜닝은 독, 견고함 우선)를 따라 대규모 그리드서치 대신
후보 5개만 두 fold(main 주판정, ref 참고) x 2시드로 비교한다.
raw_id(Step3-2에서 확정) 고정.
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
SEEDS = [42, 7]

CANDIDATES = {
    "default": dict(iterations=200, learning_rate=0.06, depth=6, l2_leaf_reg=3.0),
    "deeper": dict(iterations=200, learning_rate=0.06, depth=8, l2_leaf_reg=3.0),
    "more_iter_lower_lr": dict(iterations=400, learning_rate=0.03, depth=6, l2_leaf_reg=3.0),
    "more_reg": dict(iterations=200, learning_rate=0.06, depth=6, l2_leaf_reg=10.0),
    "less_reg": dict(iterations=200, learning_rate=0.06, depth=6, l2_leaf_reg=1.0),
}


def fit_predict_catboost(X_train, y_train, X_valid, params, seed):
    model = CatBoostClassifier(
        **params, random_seed=seed, verbose=0, cat_features=CATEGORICAL_COLS,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]


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

        for name, params in CANDIDATES.items():
            for seed in SEEDS:
                t0 = time.time()
                pred = fit_predict_catboost(X_train, y_train, X_valid, params, seed)
                result = evaluate(y_valid, pred)
                dt = time.time() - t0
                rows.append({"fold": fold_name, "candidate": name, "seed": seed,
                             "brier_mean_aligned": result["brier_mean_aligned"], "time_s": dt})
                print(f"[{fold_name}] {name} seed={seed}: brier_mean_aligned={result['brier_mean_aligned']:.8f} ({dt:.1f}s)", flush=True)

    res_df = pd.DataFrame(rows)
    print()
    summary = res_df.groupby(["fold", "candidate"])["brier_mean_aligned"].agg(["mean", "std"]).sort_values(["fold", "mean"])
    print(summary)
    out_path = ROOT / "results"
    out_path.mkdir(exist_ok=True)
    res_df.to_csv(out_path / "10_catboost_tuning.csv", index=False)


if __name__ == "__main__":
    main()

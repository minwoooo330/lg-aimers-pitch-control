"""판정 문턱 확정: '같은 시드, 피처만 다름'인 paired 비교의 실제 잡음 폭을 잰다.
아무 정보도 없는 순수 랜덤 컬럼을 추가했을 때 base 대비 brier가 얼마나 움직이는지가
곧 앞으로의 '유의미한 차이' 판단 기준선이 된다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import CATEGORICAL_COLS, FEATURE_COLS, build_categories, prepare_features
from validation import FOLD_DEFS, evaluate, make_fold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

HGB_PARAMS = dict(
    max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
    min_samples_leaf=200, early_stopping=False,
)


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)
    base_cat_mask = [c in CATEGORICAL_COLS for c in FEATURE_COLS]

    for fold_name in FOLD_DEFS:
        fold = make_fold(df, fold_name)
        train_df = df.loc[fold.train_idx]
        valid_df = df.loc[fold.valid_idx]
        y_train = train_df["control_success"].to_numpy()
        y_valid = valid_df["control_success"].to_numpy()

        diffs = []
        for seed in [42, 7, 2024]:
            rng = np.random.default_rng(seed)
            X_train = prepare_features(train_df, categories)
            X_valid = prepare_features(valid_df, categories)

            m_base = HistGradientBoostingClassifier(categorical_features=base_cat_mask, random_state=seed, **HGB_PARAMS)
            m_base.fit(X_train, y_train)
            b_base = evaluate(y_valid, m_base.predict_proba(X_valid)[:, 1])["brier_mean_aligned"]

            X_train_p = X_train.copy()
            X_train_p["noise"] = rng.normal(size=len(X_train_p))
            X_valid_p = X_valid.copy()
            X_valid_p["noise"] = rng.normal(size=len(X_valid_p))
            m_pl = HistGradientBoostingClassifier(categorical_features=base_cat_mask + [False], random_state=seed, **HGB_PARAMS)
            m_pl.fit(X_train_p, y_train)
            b_pl = evaluate(y_valid, m_pl.predict_proba(X_valid_p)[:, 1])["brier_mean_aligned"]

            diff = b_pl - b_base
            diffs.append(diff)
            print(f"[{fold_name}] seed={seed}: base={b_base:.8f} +noise={b_pl:.8f} diff={diff:+.8e}", flush=True)

        diffs = np.array(diffs)
        print(f"=== {fold_name}: placebo diff mean={diffs.mean():+.4e}  |diff| max={np.abs(diffs).max():.4e} ===")
        print()


if __name__ == "__main__":
    main()

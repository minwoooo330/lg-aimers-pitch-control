"""판정 기준 확립: 같은 설정에서 random_state만 바꿔 재학습했을 때 Brier가 얼마나 흔들리는지 측정.
이후 모든 피처/모델 비교는 이 잡음 수준보다 큰 차이만 '신호'로 인정한다.

참고: HistGradientBoostingClassifier는 n_samples>10000이면 early_stopping='auto'가 기본값이라
내부적으로 검증용 분할을 random_state로 무작위 수행한다. 이게 시드 간 변동의 주요 원인일 수 있다.
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

SEEDS = [0, 1, 2, 3, 4, 5]


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)
    cat_mask = [c in CATEGORICAL_COLS for c in FEATURE_COLS]

    for fold_name in FOLD_DEFS:
        fold = make_fold(df, fold_name)
        train_df = df.loc[fold.train_idx]
        valid_df = df.loc[fold.valid_idx]
        X_train = prepare_features(train_df, categories)
        X_valid = prepare_features(valid_df, categories)
        y_train = train_df["control_success"].to_numpy()
        y_valid = valid_df["control_success"].to_numpy()

        scores = []
        for seed in SEEDS:
            model = HistGradientBoostingClassifier(
                categorical_features=cat_mask,
                max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                min_samples_leaf=200, random_state=seed,
                early_stopping=False,
            )
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_valid)[:, 1]
            b = evaluate(y_valid, pred)["brier_mean_aligned"]
            scores.append(b)
            print(f"[{fold_name}] seed={seed}: brier_mean_aligned={b:.8f}", flush=True)

        scores = np.array(scores)
        print(f"=== {fold_name}: mean={scores.mean():.8f}  sd={scores.std(ddof=1):.8e}  "
              f"range=[{scores.min():.8f}, {scores.max():.8f}]  1sigma_paired={np.sqrt(2)*scores.std(ddof=1):.8e} ===")
        print()


if __name__ == "__main__":
    main()

"""Step 4-1: CatBoost 시드 앙상블 검증.

가설(외부 레포 인사이트): 튜닝 대신 시드 앙상블로 robustness를 확보한다.
5개 시드의 확률 예측을 각각 저장하고, 평균이 단일 시드보다 두 fold 모두에서
개선되는지 직접 확인한다 (단순히 개별 시드 Brier의 평균이 아니라, 실제로
예측 확률을 평균한 뒤 채점 - 앙상블 효과는 예측 자체를 섞어야 나타난다).
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
SEEDS = [42, 7, 2024, 13, 77]

PARAMS = dict(iterations=200, learning_rate=0.06, depth=6, l2_leaf_reg=3.0)


def fit_predict(X_train, y_train, X_valid, seed):
    model = CatBoostClassifier(**PARAMS, random_seed=seed, verbose=0, cat_features=CATEGORICAL_COLS)
    model.fit(X_train, y_train)
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
        y_valid = valid_df["control_success"].to_numpy()

        X_train = prepare_features(train_df, categories)
        X_valid = prepare_features(valid_df, categories)

        preds = {}
        for seed in SEEDS:
            t0 = time.time()
            pred = fit_predict(X_train, y_train, X_valid, seed)
            preds[seed] = pred
            b = evaluate(y_valid, pred)["brier_mean_aligned"]
            print(f"[{fold_name}] seed={seed}: brier_mean_aligned={b:.8f} ({time.time()-t0:.1f}s)", flush=True)

        individual_briers = [evaluate(y_valid, preds[s])["brier_mean_aligned"] for s in SEEDS]
        print(f"  개별 시드 평균 brier: {np.mean(individual_briers):.8f}")

        # 시드 수를 1개씩 늘려가며 누적 평균 앙상블 효과 확인
        cum_pred = np.zeros_like(y_valid, dtype=np.float64)
        for i, seed in enumerate(SEEDS, start=1):
            cum_pred += preds[seed]
            avg_pred = cum_pred / i
            b_ens = evaluate(y_valid, avg_pred)["brier_mean_aligned"]
            print(f"  [{fold_name}] 누적 {i}개 시드 평균: brier_mean_aligned={b_ens:.8f}")
        print()

        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)
        pred_df = pd.DataFrame({"row_id": valid_df["row_id"].values, "y_true": y_valid})
        for seed in SEEDS:
            pred_df[f"seed_{seed}"] = preds[seed]
        pred_df.to_csv(out_dir / f"11_seed_preds_{fold_name}.csv.gz", index=False)


if __name__ == "__main__":
    main()

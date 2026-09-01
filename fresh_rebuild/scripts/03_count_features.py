"""Step 2-2: 카운트 조합 파생 피처가 GBDT에서 실제로 값어치가 있는가.

관찰(직접 EDA): balls_before=3에서 성공률이 뚜렷이 낮고(0.499~0.507),
strikes_before는 0~1볼에서는 1스트라이크가 오히려 최고점인 비단조 패턴.
스프레드는 3.5%p로 크지 않고 조합이 12가지뿐이라, balls_before/strikes_before/
outs_before가 이미 raw 숫자로 있는 상태에서 트리가 스스로 이 상호작용을 학습할
가능성이 높다는 가설을 세운다.

비교:
  base        : features.py 기본 피처셋 (Step2-1 결론: ID는 raw 숫자)
  +count_bundle: base + count_code(범주형 12종) + is_three_ball + is_full_count
                 + count_leverage(strikes-balls) + is_hitters_count
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import CATEGORICAL_COLS, FEATURE_COLS, build_categories, prepare_features
from validation import FOLD_DEFS, evaluate, make_fold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

HGB_PARAMS = dict(
    max_iter=200,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    random_state=42,
)


def add_count_bundle(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["count_code"] = (df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)).astype("category")
    X["is_three_ball"] = (df["balls_before"] == 3).astype("float64")
    X["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype("float64")
    X["count_leverage"] = (df["strikes_before"] - df["balls_before"]).astype("float64")
    X["is_hitters_count"] = (df["balls_before"] > df["strikes_before"]).astype("float64")
    return X


def main():
    print("데이터 로드...", flush=True)
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)
    base_cat_mask = [c in CATEGORICAL_COLS for c in FEATURE_COLS]

    rows = []
    for fold_name in FOLD_DEFS:
        fold = make_fold(df, fold_name)
        train_df = df.loc[fold.train_idx]
        valid_df = df.loc[fold.valid_idx]
        y_train = train_df["control_success"].to_numpy()
        y_valid = valid_df["control_success"].to_numpy()

        for variant in ["base", "count_bundle"]:
            t0 = time.time()
            X_train = prepare_features(train_df, categories)
            X_valid = prepare_features(valid_df, categories)
            cat_mask = list(base_cat_mask)

            if variant == "count_bundle":
                X_train = add_count_bundle(train_df, X_train)
                X_valid = add_count_bundle(valid_df, X_valid)
                cat_mask = cat_mask + [True, False, False, False, False]

            model = HistGradientBoostingClassifier(categorical_features=cat_mask, **HGB_PARAMS)
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_valid)[:, 1]
            result = evaluate(y_valid, pred)
            dt = time.time() - t0

            rows.append({
                "fold": fold_name, "variant": variant, "n_features": X_train.shape[1],
                "brier_raw": result["brier_raw"], "brier_mean_aligned": result["brier_mean_aligned"],
                "time_s": dt,
            })
            print(f"[{fold_name}] {variant}: brier_mean_aligned={result['brier_mean_aligned']:.8f} ({dt:.1f}s)", flush=True)

    res_df = pd.DataFrame(rows)
    print()
    print(res_df.to_string(index=False))
    out_path = ROOT / "results"
    out_path.mkdir(exist_ok=True)
    res_df.to_csv(out_path / "03_count_features.csv", index=False)


if __name__ == "__main__":
    main()

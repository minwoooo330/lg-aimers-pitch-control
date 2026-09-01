"""Step 2-4: 투수 이력 추세(trend) 피처 검증.

EDA 결론: 고표본(n>=1000)에서 career asof_pitcher_success_rate는 실제 대비 과대(0.534 vs 0.519,
누적 이력이 최신 흐름에 뒤처짐 = stale), prev5는 거의 정확(0.520 vs 0.519).
trend_gap=career-prev5의 10분위별 actual은 0.553->0.497로 단조 하락하는데 career_rate는
전 구간 평평(0.532~0.537) -> career_rate에 없는 독립 신호. 트리가 축 분할만으로 두 raw 컬럼의
차이를 재현하기 어려우므로 explicit 차분 피처가 이번엔 값어치가 있을 것으로 예상.
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
    max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
    min_samples_leaf=200, early_stopping=False, random_state=42,
)


def add_trend_bundle(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["trend_gap5_success"] = df["asof_pitcher_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    X["trend_gap1_success"] = df["asof_pitcher_success_rate"] - df["asof_pitcher_prev1_game_success_rate"]
    X["trend_gap5_middle"] = df["asof_pitcher_middle_rate"] - df["asof_pitcher_prev5_game_middle_rate"]
    X["trend_gap1_middle"] = df["asof_pitcher_middle_rate"] - df["asof_pitcher_prev1_game_middle_rate"]
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

        for variant in ["base", "trend_bundle"]:
            t0 = time.time()
            X_train = prepare_features(train_df, categories)
            X_valid = prepare_features(valid_df, categories)
            cat_mask = list(base_cat_mask)

            if variant == "trend_bundle":
                X_train = add_trend_bundle(train_df, X_train)
                X_valid = add_trend_bundle(valid_df, X_valid)
                cat_mask = cat_mask + [False, False, False, False]

            model = HistGradientBoostingClassifier(categorical_features=cat_mask, **HGB_PARAMS)
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_valid)[:, 1]
            result = evaluate(y_valid, pred)
            dt = time.time() - t0

            rows.append({
                "fold": fold_name, "variant": variant, "n_features": X_train.shape[1],
                "brier_mean_aligned": result["brier_mean_aligned"], "time_s": dt,
            })
            print(f"[{fold_name}] {variant}: brier_mean_aligned={result['brier_mean_aligned']:.8f} ({dt:.1f}s)", flush=True)

    res_df = pd.DataFrame(rows)
    print()
    print(res_df.to_string(index=False))
    out_path = ROOT / "results"
    out_path.mkdir(exist_ok=True)
    res_df.to_csv(out_path / "06_trend_features.csv", index=False)


if __name__ == "__main__":
    main()

"""Step 2-3: 압박 상황 파생 피처 검증.

EDA 결론: '압박이 제구를 흔든다'는 통념과 달리 고LI/접전에서 오히려 성공률이 높다
(선수 배치 선별 편향으로 추정). 득점권은 사실상 무차이, 만루만 약한(-1%p) 음의 신호.
li, runner_on_*, score_diff_pitcher_team이 이미 raw로 있어 redundant일 것으로 예상.
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


def add_pressure_bundle(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    scoring_pos = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype("float64")
    X["close_game"] = (df["score_diff_pitcher_team"].abs() <= 1).astype("float64")
    X["bases_loaded"] = (df["num_runners_on"] == 3).astype("float64")
    X["li_x_scoring_pos"] = df["li"].astype("float64") * scoring_pos
    X["score_diff_abs"] = df["score_diff_pitcher_team"].abs().astype("float64")
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

        for variant in ["base", "pressure_bundle"]:
            t0 = time.time()
            X_train = prepare_features(train_df, categories)
            X_valid = prepare_features(valid_df, categories)
            cat_mask = list(base_cat_mask)

            if variant == "pressure_bundle":
                X_train = add_pressure_bundle(train_df, X_train)
                X_valid = add_pressure_bundle(valid_df, X_valid)
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
    res_df.to_csv(out_path / "05_pressure_features.csv", index=False)


if __name__ == "__main__":
    main()

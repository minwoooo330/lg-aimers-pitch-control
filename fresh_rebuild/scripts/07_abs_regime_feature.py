"""Step 2-5: ABS(자동 볼판정) 레짐 플래그 검증.

Step0 EDA + 사용자 확인: KBO는 ABS를 퓨처스 2023년, 1군 2024년부터 도입.
is_abs = (game_type=='F' & season>=2023) | (game_type=='R' & season>=2024)

주의: 2025 test는 두 리그 모두 ABS 체제이므로 is_abs는 test에서 상수(=1)다.
그래도 유용할 수 있는 이유: season/game_type 조합을 raw로 두면 트리가
'game_type==F AND season>=2023'을 배우려면 2번의 순차 분할이 필요하고,
게다가 레짐 차이가 단순 main effect가 아니라 다른 피처들과의 관계 자체를
바꾸는 것이라면(상호작용), 명시적 플래그 하나가 각 트리에서 최상단 분할로
싸게 레짐을 가른 뒤 하위 분할 예산을 절약해줄 수 있다.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import CATEGORICAL_COLS, FEATURE_COLS, build_categories, prepare_features
from model_config import DEFAULT_HGB_PARAMS
from validation import FOLD_DEFS, evaluate, make_fold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEEDS = [42, 7, 2024]


def add_abs_flag(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    is_f = df["game_type"] == "F"
    X["is_abs"] = (((is_f) & (df["season"] >= 2023)) | ((~is_f) & (df["season"] >= 2024))).astype("float64")
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

        print(f"  [{fold_name}] train is_abs 비율 확인용 분포:")
        is_f_t = train_df["game_type"] == "F"
        abs_flag_t = ((is_f_t & (train_df["season"] >= 2023)) | (~is_f_t & (train_df["season"] >= 2024)))
        print(f"    train is_abs=1 비율: {abs_flag_t.mean():.4f} (n={abs_flag_t.sum():,})")
        is_f_v = valid_df["game_type"] == "F"
        abs_flag_v = ((is_f_v & (valid_df["season"] >= 2023)) | (~is_f_v & (valid_df["season"] >= 2024)))
        print(f"    valid is_abs=1 비율: {abs_flag_v.mean():.4f} (n={abs_flag_v.sum():,})")

        diffs = []
        for seed in SEEDS:
            params = dict(DEFAULT_HGB_PARAMS)
            params["random_state"] = seed
            results = {}
            for variant in ["base", "abs_flag"]:
                X_train = prepare_features(train_df, categories)
                X_valid = prepare_features(valid_df, categories)
                cat_mask = list(base_cat_mask)
                if variant == "abs_flag":
                    X_train = add_abs_flag(train_df, X_train)
                    X_valid = add_abs_flag(valid_df, X_valid)
                    cat_mask = cat_mask + [False]
                model = HistGradientBoostingClassifier(categorical_features=cat_mask, **params)
                model.fit(X_train, y_train)
                pred = model.predict_proba(X_valid)[:, 1]
                results[variant] = evaluate(y_valid, pred)["brier_mean_aligned"]
            diff = results["base"] - results["abs_flag"]
            diffs.append(diff)
            rows.append({"fold": fold_name, "seed": seed, "base": results["base"],
                         "abs_flag": results["abs_flag"], "diff": diff})
            print(f"[{fold_name}] seed={seed}: base={results['base']:.8f} abs_flag={results['abs_flag']:.8f} diff={diff:+.4e}", flush=True)

        diffs = np.array(diffs)
        print(f"=== {fold_name}: mean_diff={diffs.mean():+.4e} (양수=개선) ===\n")

    res_df = pd.DataFrame(rows)
    out_path = ROOT / "results"
    out_path.mkdir(exist_ok=True)
    res_df.to_csv(out_path / "07_abs_regime_feature.csv", index=False)


if __name__ == "__main__":
    main()

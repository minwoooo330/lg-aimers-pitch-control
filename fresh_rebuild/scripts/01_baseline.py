"""Step 1: 베이스라인 모델.

- 원본 47개 입력 그대로 (파생 피처 없음).
- sklearn HistGradientBoostingClassifier (환경에 lightgbm/xgboost/catboost 미설치).
- main/ref fold에서 mean-aligned Brier로 평가.
- test.csv(5행 샘플) + sample_submission.csv 포맷으로 end-to-end 제출 파이프라인 확인.
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


def main():
    print("데이터 로드...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    categories = build_categories(df)
    cat_mask = [c in CATEGORICAL_COLS for c in FEATURE_COLS]

    for name in FOLD_DEFS:
        fold = make_fold(df, name)
        train_df = df.loc[fold.train_idx]
        valid_df = df.loc[fold.valid_idx]

        X_train = prepare_features(train_df, categories)
        y_train = train_df["control_success"].to_numpy()
        X_valid = prepare_features(valid_df, categories)
        y_valid = valid_df["control_success"].to_numpy()

        model = HistGradientBoostingClassifier(categorical_features=cat_mask, **DEFAULT_HGB_PARAMS)

        t0 = time.time()
        model.fit(X_train, y_train)
        fit_s = time.time() - t0

        pred = model.predict_proba(X_valid)[:, 1]
        result = evaluate(y_valid, pred)

        print(f"--- fold={name} (train n={len(X_train):,}, valid n={len(X_valid):,}, fit {fit_s:.1f}s) ---")
        print(f"  pred_mean={result['pred_mean']:.6f}  true_mean={result['true_mean']:.6f}")
        print(f"  brier_raw={result['brier_raw']:.8f}  brier_mean_aligned={result['brier_mean_aligned']:.8f}")
        print()

    # end-to-end 제출 포맷 점검: 전체 train으로 학습 -> test.csv(5행 샘플) 추론
    print("제출 포맷 점검 (test.csv 5행 샘플)...")
    X_full = prepare_features(df, categories)
    y_full = df["control_success"].to_numpy()
    final_model = HistGradientBoostingClassifier(categorical_features=cat_mask, **DEFAULT_HGB_PARAMS)
    final_model.fit(X_full, y_full)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_test = prepare_features(test_df, categories)
    test_pred = final_model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({"row_id": test_df["row_id"], "control_success": test_pred})
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    print(submission)
    print("row_id 일치:", (submission["row_id"].values == sample_sub["row_id"].values).all())
    print("컬럼 일치:", list(submission.columns) == list(sample_sub.columns))

    # 행 독립성 점검: 단일 행 예측 vs 배치 예측이 같아야 함
    single_preds = []
    for i in range(len(X_test)):
        p = final_model.predict_proba(X_test.iloc[[i]])[:, 1][0]
        single_preds.append(p)
    single_preds = np.array(single_preds)
    max_diff = np.max(np.abs(single_preds - test_pred))
    print(f"단일행 vs 배치 예측 최대 차이: {max_diff:.2e}")


if __name__ == "__main__":
    main()

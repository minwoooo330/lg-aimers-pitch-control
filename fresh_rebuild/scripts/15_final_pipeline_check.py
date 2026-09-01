"""Step 4-4: 최종 파이프라인 end-to-end 검증.

2019~2024 전체 데이터로 5시드 CatBoost 앙상블을 학습하고 models/에 저장한다.
test.csv(5행 샘플)로 다음을 확인한다:
  1. 제출 포맷(row_id, control_success 컬럼, sample_submission.csv와 컬럼/행수 일치)
  2. 행 독립성 (단일 행 예측 vs 배치 예측이 완전히 같아야 함 - 대회 판별 기준)
  3. 예측값이 [0,1] 범위의 유한값인지
  4. shift 적용 전/후 예측 평균 비교
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from features import build_categories, prepare_features
from final_model import FINAL_SHIFT, predict_ensemble, predict_final, train_ensemble

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def main():
    print("데이터 로드...", flush=True)
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)

    print(f"5시드 CatBoost 앙상블 학습 (전체 {len(df):,}행)...", flush=True)
    t0 = time.time()
    models = train_ensemble(df, categories, model_dir=MODEL_DIR)
    print(f"학습 완료: {time.time()-t0:.1f}s, 모델 저장: {MODEL_DIR}", flush=True)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    X_test = prepare_features(test_df, categories)
    raw_pred_batch = predict_ensemble(models, X_test)
    final_pred_batch = np.clip(raw_pred_batch + FINAL_SHIFT, 0.0, 1.0)

    print("\n=== 배치 예측 (raw / shift적용) ===")
    for i, rid in enumerate(test_df["row_id"]):
        print(f"  {rid}: raw={raw_pred_batch[i]:.6f}  final={final_pred_batch[i]:.6f}")
    print(f"  raw 평균={raw_pred_batch.mean():.6f}  final 평균={final_pred_batch.mean():.6f}")

    print("\n=== 행 독립성 검증 (단일행 예측 vs 배치 예측) ===")
    single_preds = []
    for i in range(len(X_test)):
        single_X = X_test.iloc[[i]]
        single_pred = np.clip(predict_ensemble(models, single_X) + FINAL_SHIFT, 0.0, 1.0)
        single_preds.append(single_pred[0])
    single_preds = np.array(single_preds)
    max_diff = np.max(np.abs(single_preds - final_pred_batch))
    print(f"  단일행 vs 배치 최대 차이: {max_diff:.2e}")

    print("\n=== 제출 포맷 검증 ===")
    submission = pd.DataFrame({"row_id": test_df["row_id"], "control_success": final_pred_batch})
    print(f"  row_id 일치: {(submission['row_id'].values == sample_sub['row_id'].values).all()}")
    print(f"  컬럼 일치: {list(submission.columns) == list(sample_sub.columns)}")
    print(f"  값 범위: [{submission['control_success'].min():.6f}, {submission['control_success'].max():.6f}]")
    print(f"  결측 수: {submission['control_success'].isna().sum()}")
    print(f"  유한값 여부: {np.isfinite(submission['control_success']).all()}")

    submission.to_csv(ROOT / "results" / "15_submission_check.csv", index=False)
    print(f"\n저장: results/15_submission_check.csv")


if __name__ == "__main__":
    main()

"""Step 4-3: 2025 제출용 확률 보정(shift) 산출.

방법론: walk-forward 편향(pred_mean - actual_mean)을 3개 시점(2022/2023/2024)에서 측정하고,
이 편향을 shift로 직접 사용한다(shift = -bias). "과거 편향 평균으로 다음 해를 예측"하는
이 방식 자체가 타당한지 leave-one-out 방식으로 검증한다:
  - bias_2022만으로 만든 shift를 2023 fold에 적용했을 때 보정 후 예측 평균이 실제와 얼마나 가까운가
  - bias_2022,2023 평균으로 만든 shift를 2024 fold에 적용했을 때는 어떤가
독립적인 교차검증으로 리그별(R/F) 추세외삽 기반 2025 목표치(별도 계산, ~0.473~0.474)와
최종 shift 적용 결과가 정합적인지도 확인한다.

리더보드 점수로 shift를 역산하지 않는다(대회 규정 위반). train의 시간 구조만 사용.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import CATEGORICAL_COLS, build_categories, prepare_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PARAMS = dict(iterations=200, learning_rate=0.06, depth=6, l2_leaf_reg=3.0)
SEED = 42


def fit_predict(train_df, y_train, valid_df, categories, seed):
    X_train = prepare_features(train_df, categories)
    X_valid = prepare_features(valid_df, categories)
    model = CatBoostClassifier(**PARAMS, random_seed=seed, verbose=0, cat_features=CATEGORICAL_COLS)
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)

    # 3개 walk-forward 지점에서 raw 예측(보정 전) 저장
    points = {}
    for train_end, valid_season in [(2021, 2022), (2022, 2023), (2023, 2024)]:
        train_df = df[df["season"] <= train_end]
        valid_df = df[df["season"] == valid_season]
        y_train = train_df["control_success"].to_numpy()
        y_valid = valid_df["control_success"].to_numpy()
        pred = fit_predict(train_df, y_train, valid_df, categories, SEED)

        pred_mean = pred.mean()
        actual_mean = y_valid.mean()
        bias = pred_mean - actual_mean
        points[valid_season] = {"pred": pred, "y_valid": y_valid, "pred_mean": pred_mean,
                                 "actual_mean": actual_mean, "bias": bias}
        print(f"train<=({train_end}) -> {valid_season}: pred_mean={pred_mean:.6f} "
              f"actual_mean={actual_mean:.6f} bias={bias:+.6f}")

    print()
    print("=== leave-one-out 검증: 과거 편향으로 다음 해 편향을 얼마나 잘 예측하는가 ===")

    # bias_2022만으로 2023 shift 추정 -> 2023 fold에 적용
    shift_2023 = -points[2022]["bias"]
    corrected_2023_mean = points[2023]["pred_mean"] + shift_2023
    err_2023 = corrected_2023_mean - points[2023]["actual_mean"]
    print(f"[2022 편향만] shift={shift_2023:+.6f} -> 2023 보정후평균={corrected_2023_mean:.6f} "
          f"실제={points[2023]['actual_mean']:.6f} 오차={err_2023:+.6f}")

    # bias_2022,2023 평균으로 2024 shift 추정 -> 2024 fold에 적용
    shift_2024_avg = -np.mean([points[2022]["bias"], points[2023]["bias"]])
    corrected_2024_mean_avg = points[2024]["pred_mean"] + shift_2024_avg
    err_2024_avg = corrected_2024_mean_avg - points[2024]["actual_mean"]
    print(f"[2022+2023 평균편향] shift={shift_2024_avg:+.6f} -> 2024 보정후평균={corrected_2024_mean_avg:.6f} "
          f"실제={points[2024]['actual_mean']:.6f} 오차={err_2024_avg:+.6f}")

    # bias_2023(직전 1개)만으로 2024 shift 추정 -> 2024 fold에 적용 (최근값만 사용 대안)
    shift_2024_recent = -points[2023]["bias"]
    corrected_2024_mean_recent = points[2024]["pred_mean"] + shift_2024_recent
    err_2024_recent = corrected_2024_mean_recent - points[2024]["actual_mean"]
    print(f"[2023 편향만(최근)] shift={shift_2024_recent:+.6f} -> 2024 보정후평균={corrected_2024_mean_recent:.6f} "
          f"실제={points[2024]['actual_mean']:.6f} 오차={err_2024_recent:+.6f}")

    print()
    print("=== 3개 지점 전체 편향 패턴 ===")
    biases = [points[y]["bias"] for y in [2022, 2023, 2024]]
    print(f"bias_2022={biases[0]:+.6f}  bias_2023={biases[1]:+.6f}  bias_2024={biases[2]:+.6f}")
    print(f"평균={np.mean(biases):+.6f}  표준편차={np.std(biases, ddof=1):.6f}")

    # 최종 2025용 shift: 3개 지점 평균 편향의 음수 (leave-one-out 검증 결과로 방법 확정)
    final_shift = -np.mean(biases)
    print(f"\n2025 적용 최종 shift (3개 지점 평균 편향의 음수) = {final_shift:+.6f}")

    # 독립 교차검증: 추세외삽 목표치(별도 계산, 0.4728~0.4743)와의 정합성
    # 검증 방법: 2019~2024 전체 학습 모델의 train 자기예측 평균에 shift를 더하면
    # 대략 목표 범위 안에 들어오는지 확인 (참고용 sanity check, in-sample이라 완전한 검증은 아님)
    X_full = prepare_features(df, categories)
    y_full = df["control_success"].to_numpy()
    final_model = CatBoostClassifier(**PARAMS, random_seed=SEED, verbose=0, cat_features=CATEGORICAL_COLS)
    final_model.fit(X_full, y_full)
    full_pred_mean = final_model.predict_proba(X_full)[:, 1].mean()
    print(f"\n[참고, in-sample] 전체학습모델 자기예측평균={full_pred_mean:.6f}, +shift={full_pred_mean+final_shift:.6f}")
    print("[참고] 추세외삽 2025 목표범위: [0.4728, 0.4743]")


if __name__ == "__main__":
    main()

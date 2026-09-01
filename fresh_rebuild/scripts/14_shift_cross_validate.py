"""Step 4-3b: shift 방법론 교차검증.

13_shift_calibration.py에서 3개 지점(2022/2023/2024) 편향을 측정했는데, 2023은 F(퓨처스)
ABS 전환이 그 해에 처음 일어나 모델이 전혀 못 본 레짐으로 예측한 경우라 편향이 유독 컸다
(+0.0213, 나머지 두 배 이상). 2025 예측 시 최종 모델은 이미 2023~2024 F-ABS, 2024 R-ABS를
전부 학습에 포함하므로 이런 '눈뜬장님' 충격이 재발하지 않는다. 따라서 2023을 뺀
2022+2024 평균(+0.0063)이 2025에 더 적합한 추정치일 가능성을 검증한다.

4번째 지점(train<=2020 -> 2021)을 추가한다. 2021도 2022처럼 두 리그 모두 ABS 이전
안정 레짐이라, bias_2022(+0.0039)가 '안정기의 재현 가능한 자연 편향'인지 우연인지 확인할
수 있는 독립적 관측치가 된다.
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

# 13_shift_calibration.py 결과 재사용 (이미 확인됨, 재계산 비용 절약)
KNOWN_BIAS = {
    2022: 0.003918,
    2023: 0.021279,
    2024: 0.008664,
}


def fit_predict(train_df, y_train, valid_df, categories, seed):
    X_train = prepare_features(train_df, categories)
    X_valid = prepare_features(valid_df, categories)
    model = CatBoostClassifier(**PARAMS, random_seed=seed, verbose=0, cat_features=CATEGORICAL_COLS)
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)

    # 4번째 지점: train<=2020 -> 2021
    train_df = df[df["season"] <= 2020]
    valid_df = df[df["season"] == 2021]
    y_train = train_df["control_success"].to_numpy()
    y_valid = valid_df["control_success"].to_numpy()
    pred = fit_predict(train_df, y_train, valid_df, categories, SEED)
    pred_mean = pred.mean()
    actual_mean = y_valid.mean()
    bias_2021 = pred_mean - actual_mean
    print(f"train<=(2020) -> 2021: pred_mean={pred_mean:.6f} actual_mean={actual_mean:.6f} bias={bias_2021:+.6f}")

    print()
    print("=== '안정 레짐(전환 없음)' 지점끼리 비교 ===")
    print(f"bias_2021(안정) = {bias_2021:+.6f}")
    print(f"bias_2022(안정) = {KNOWN_BIAS[2022]:+.6f}")
    print(f"bias_2023(F전환 충격) = {KNOWN_BIAS[2023]:+.6f}")
    print(f"bias_2024(R전환 일부 반영) = {KNOWN_BIAS[2024]:+.6f}")
    stable_avg = np.mean([bias_2021, KNOWN_BIAS[2022]])
    print(f"\n안정 지점 평균(2021+2022) = {stable_avg:+.6f}")

    print()
    print("=== leave-one-out 재검증 (4점 기준) ===")
    # 2021 편향만으로 2022 shift -> 2022 fold (둘다 안정레짐, 가장 깨끗한 케이스)
    shift_a = -bias_2021
    # 실제 2022 fold의 raw pred_mean이 필요 (13번 스크립트 결과에서: 0.532839)
    pred_mean_2022 = 0.532839
    actual_2022 = 0.528920
    err_a = (pred_mean_2022 + shift_a) - actual_2022
    print(f"[2021편향만] shift={shift_a:+.6f} -> 2022 보정후오차={err_a:+.6f}")

    # 2021+2022 평균(둘다 안정)으로 2023 shift -> 2023 fold (전환 충격 예측 시도, 실패 예상)
    shift_b = -stable_avg
    pred_mean_2023 = 0.521236
    actual_2023 = 0.499957
    err_b = (pred_mean_2023 + shift_b) - actual_2023
    print(f"[안정지점평균(2021+2022)] shift={shift_b:+.6f} -> 2023 보정후오차={err_b:+.6f} (전환년도라 실패 예상)")

    # 2021+2022 평균(2023 제외)으로 2024 shift -> 2024 fold
    pred_mean_2024 = 0.494769
    actual_2024 = 0.486105
    err_c = (pred_mean_2024 + shift_b) - actual_2024
    print(f"[안정지점평균(2021+2022), 2023 제외]로 2024 예측: shift={shift_b:+.6f} 보정후오차={err_c:+.6f}")

    # 비교: 3점 전체평균(2022+2023+2024)으로 2024를 '자기 자신 포함'해서 만든 것 대신,
    # 정직하게 2021+2022+2023 평균으로 2024를 예측(모든 점 포함 버전)
    shift_d = -np.mean([bias_2021, KNOWN_BIAS[2022], KNOWN_BIAS[2023]])
    err_d = (pred_mean_2024 + shift_d) - actual_2024
    print(f"[전체평균(2021+2022+2023, 2023포함)]로 2024 예측: shift={shift_d:+.6f} 보정후오차={err_d:+.6f}")

    print()
    print("=== 2025용 최종 후보 비교 ===")
    cand_exclude_2023 = -np.mean([bias_2021, KNOWN_BIAS[2022], KNOWN_BIAS[2024]])
    cand_all4 = -np.mean([bias_2021, KNOWN_BIAS[2022], KNOWN_BIAS[2023], KNOWN_BIAS[2024]])
    cand_recent_only = -KNOWN_BIAS[2024]
    print(f"2023 제외 3점 평균(2021,2022,2024) shift = {cand_exclude_2023:+.6f}")
    print(f"4점 전체 평균 shift = {cand_all4:+.6f}")
    print(f"최근 1점(2024)만 shift = {cand_recent_only:+.6f}")


if __name__ == "__main__":
    main()

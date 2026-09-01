"""Step 0 스모크 테스트: fold 구성이 의도대로 되는지, 채점 함수가 정상 동작하는지 확인.
상수(=train 평균) 예측으로 하네스 자체를 검증한다 (모델 성능 판단용이 아님).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from validation import FOLD_DEFS, evaluate, make_fold, summarize_fold

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"


def main():
    df = pd.read_csv(DATA_PATH, usecols=["season", "game_type", "control_success"])

    for name in FOLD_DEFS:
        fold = make_fold(df, name)
        summary = summarize_fold(df, fold)
        print(f"--- fold={summary['fold']} (train {summary['train_seasons']} -> valid {summary['valid_season']}) ---")
        print(f"  n_train={summary['n_train']:,}  n_valid={summary['n_valid']:,}")
        print(f"  valid success_rate={summary['valid_success_rate']:.6f}")
        print(f"  valid game_type counts={summary['valid_game_type_counts']}")

        y_train = df.loc[fold.train_idx, "control_success"].to_numpy()
        y_valid = df.loc[fold.valid_idx, "control_success"].to_numpy()
        const_pred = np.full_like(y_valid, fill_value=y_train.mean(), dtype=np.float64)

        result = evaluate(y_valid, const_pred)
        print(f"  [constant baseline] pred={result['pred_mean']:.6f} true={result['true_mean']:.6f} "
              f"brier_raw={result['brier_raw']:.8f} brier_mean_aligned={result['brier_mean_aligned']:.8f}")
        print()


if __name__ == "__main__":
    main()

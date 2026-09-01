"""Step 3-1: 모델 계열 비교 (HGB / LightGBM / XGBoost / CatBoost / Logistic Regression).

모두 features.py의 동일한 전처리 출력(prepare_features)을 재사용한다 - 범주형 저카디널리티
컬럼은 pandas 'category' dtype, pitcher_id/batter_id는 raw 숫자(Step2-1 결론).
각 라이브러리는 이 category dtype을 자기 방식으로 인식하게만 설정한다.

Logistic Regression만 예외: 선형모델은 raw ID 숫자가 의미 없으므로(순서 없음) 별도로
저카디널리티는 one-hot, 고카디널리티 ID는 빈도 인코딩을 적용한다. 성능 경쟁용이 아니라
'방향성이 말이 되는가'를 보는 sanity check 목적.

하이퍼파라미터는 Step2 외부 인사이트(레포 회고: 무리한 튜닝보다 견고한 기본값)를 따라
모델별로 서로 대략 상응하는 수준의 순한 기본값만 쓴다. 정밀 튜닝은 승자를 정한 뒤에 한다.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS, build_categories, prepare_features
from model_config import DEFAULT_HGB_PARAMS
from validation import FOLD_DEFS, evaluate, make_fold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 42

ID_COLS = ["pitcher_id", "batter_id"]


def fit_predict_hgb(X_train, y_train, X_valid, cat_mask, seed):
    model = HistGradientBoostingClassifier(categorical_features=cat_mask, **{**DEFAULT_HGB_PARAMS, "random_state": seed})
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]


def fit_predict_lgbm(X_train, y_train, X_valid, cat_cols, seed):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.06, num_leaves=31, min_child_samples=200,
        random_state=seed, verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    return model.predict_proba(X_valid)[:, 1]


def fit_predict_xgb(X_train, y_train, X_valid, seed):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.06, max_depth=6, min_child_weight=50,
        tree_method="hist", enable_categorical=True, random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]


def fit_predict_catboost(X_train, y_train, X_valid, cat_cols, seed):
    from catboost import CatBoostClassifier
    Xt = X_train.copy()
    Xv = X_valid.copy()
    for c in cat_cols:
        Xt[c] = Xt[c].astype(str)
        Xv[c] = Xv[c].astype(str)
    model = CatBoostClassifier(
        iterations=200, learning_rate=0.06, depth=6, l2_leaf_reg=3.0,
        random_seed=seed, verbose=0, cat_features=cat_cols,
    )
    model.fit(Xt, y_train)
    return model.predict_proba(Xv)[:, 1]


def fit_predict_logreg(train_df, y_train, valid_df, seed):
    low_card = CATEGORICAL_COLS  # 저카디널리티 범주형
    num_cols = [c for c in NUMERIC_COLS if c not in ID_COLS]
    freq_cols = ID_COLS

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ohe.fit(train_df[low_card].astype(str))

    freq_maps = {c: train_df[c].value_counts() for c in freq_cols}

    def build(df):
        cat_sparse = ohe.transform(df[low_card].astype(str))
        num = df[num_cols].astype("float64").fillna(0.0).to_numpy()
        freq = np.column_stack([
            df[c].map(freq_maps[c]).fillna(0).to_numpy() for c in freq_cols
        ])
        from scipy.sparse import hstack, csr_matrix
        return hstack([cat_sparse, csr_matrix(num), csr_matrix(freq)]).tocsr()

    scaler = StandardScaler(with_mean=False)
    X_train = build(train_df)
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(build(valid_df))

    model = LogisticRegression(max_iter=200, C=1.0, random_state=seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]


def main():
    print("데이터 로드...", flush=True)
    df = pd.read_csv(DATA_DIR / "train.csv")
    categories = build_categories(df)
    cat_mask = [c in CATEGORICAL_COLS for c in FEATURE_COLS]

    rows = []
    for fold_name in FOLD_DEFS:
        fold = make_fold(df, fold_name)
        train_df = df.loc[fold.train_idx]
        valid_df = df.loc[fold.valid_idx]
        y_train = train_df["control_success"].to_numpy()
        y_valid = valid_df["control_success"].to_numpy()

        X_train = prepare_features(train_df, categories)
        X_valid = prepare_features(valid_df, categories)

        models = {
            "hgb": lambda: fit_predict_hgb(X_train, y_train, X_valid, cat_mask, SEED),
            "lightgbm": lambda: fit_predict_lgbm(X_train, y_train, X_valid, CATEGORICAL_COLS, SEED),
            "xgboost": lambda: fit_predict_xgb(X_train, y_train, X_valid, SEED),
            "catboost": lambda: fit_predict_catboost(X_train, y_train, X_valid, CATEGORICAL_COLS, SEED),
            "logreg": lambda: fit_predict_logreg(train_df, y_train, valid_df, SEED),
        }

        for name, fn in models.items():
            t0 = time.time()
            pred = fn()
            result = evaluate(y_valid, pred)
            dt = time.time() - t0
            rows.append({"fold": fold_name, "model": name, "brier_mean_aligned": result["brier_mean_aligned"],
                         "brier_raw": result["brier_raw"], "time_s": dt})
            print(f"[{fold_name}] {name}: brier_mean_aligned={result['brier_mean_aligned']:.8f} ({dt:.1f}s)", flush=True)

    res_df = pd.DataFrame(rows)
    print()
    print(res_df.to_string(index=False))
    out_path = ROOT / "results"
    out_path.mkdir(exist_ok=True)
    res_df.to_csv(out_path / "08_model_family_comparison.csv", index=False)


if __name__ == "__main__":
    main()

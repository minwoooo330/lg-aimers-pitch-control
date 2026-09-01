# -*- coding: utf-8 -*-
"""실험 38: ABS 도입에 따른 라벨 체제 변화 처리.

발견: 퓨처스(F)는 2023년, 1군(R)은 2024년에 제구 성공률이 급락한다.
ABS(자동 볼판정) 도입 시점과 일치하며, 라벨 생성 규칙 자체가 바뀐 것으로 보인다.
ABS 이전 F 데이터(2019~2022)는 다른 규칙으로 만들어진 라벨이므로 학습에 해로울 수 있다.

변형:
  A base       전체 데이터 (현행)
  B dropF      ABS 이전 F 행 제거
  C era        전체 + abs_era 플래그
  D dropF_era  B + abs_era
"""
from pathlib import Path
import sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from features import add_features

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
PARAMS = dict(learning_rate=0.026902980804302, max_iter=600, max_leaf_nodes=15,
              min_samples_leaf=84, l2_regularization=14.625327121207684,
              max_features=0.6298202574719083, max_bins=128,
              early_stopping=False, random_state=42)

# ABS 시행 시점 가설: 퓨처스 2023~, 1군 2024~
def abs_era(df):
    return (((df.game_type == "F") & (df.season >= 2023)) |
            ((df.game_type == "R") & (df.season >= 2024))).astype(np.int8)

def encode(train, valid, cols, add_era):
    xt, xv = train[cols].copy(), valid[cols].copy()
    for c in CAT_COLS:
        vals = sorted(train[c].dropna().astype(str).unique())
        m = {v: i for i, v in enumerate(vals)}
        xt[c] = train[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xv[c] = valid[c].astype(str).map(m).fillna(-1).astype(np.int16)
    xt = pd.concat([xt, add_features(train)], axis=1)
    xv = pd.concat([xv, add_features(valid)], axis=1)
    if add_era:
        xt["abs_era"] = abs_era(train).to_numpy()
        xv["abs_era"] = abs_era(valid).to_numpy()
    return xt, xv

def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    print(f"전체 {len(df):,}행")
    variants = [("A_base", False, False), ("B_dropF", True, False),
                ("C_era", False, True), ("D_dropF_era", True, True)]
    rows = []
    for valid_year in (2023, 2024):
        valid = df[df.season == valid_year]
        for name, drop_pre_abs_f, add_era in variants:
            train = df[df.season < valid_year]
            if drop_pre_abs_f:
                train = train[~((train.game_type == "F") & (train.season <= 2022))]
            t0 = time.time()
            xt, xv = encode(train, valid, cols, add_era)
            model = HistGradientBoostingClassifier(**PARAMS)
            model.fit(xt, train[TARGET])
            p = model.predict_proba(xv)[:, 1]
            sec = time.time() - t0
            y = valid[TARGET].to_numpy()
            rec = dict(valid_year=valid_year, variant=name, n_train=len(train),
                       brier=brier_score_loss(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                       auc=roc_auc_score(y, p), pred_mean=float(p.mean()),
                       bias=float(p.mean() - y.mean()), seconds=round(sec, 1))
            for gt in ("R", "F"):
                m = (valid.game_type == gt).to_numpy()
                rec[f"brier_{gt}"] = brier_score_loss(y[m], p[m])
                rec[f"bias_{gt}"] = float(p[m].mean() - y[m].mean())
            rows.append(rec)
            print(f"  {valid_year} {name:12s} brier={rec['brier']:.6f} "
                  f"R={rec['brier_R']:.6f} F={rec['brier_F']:.6f} "
                  f"bias={rec['bias']:+.4f} biasF={rec['bias_F']:+.4f} ({sec:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "exp38_abs_regime_walkforward.csv", index=False, encoding="utf-8-sig")
    print("\n=== 요약: A_base 대비 Brier 개선 ===")
    for vy in (2023, 2024):
        sub = out[out.valid_year == vy].set_index("variant")
        base = sub.loc["A_base"]
        for v in sub.index:
            print(f"  {vy} {v:12s} 전체 {base.brier - sub.loc[v,'brier']:+.6f}  "
                  f"R만 {base.brier_R - sub.loc[v,'brier_R']:+.6f}")

if __name__ == "__main__":
    main()

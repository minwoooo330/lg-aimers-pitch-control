# -*- coding: utf-8 -*-
"""실험 63: exp62 손잡이 분할의 대조군 — 같은 이득이 시드만 바꿔도 나오는가.

exp62에서 손잡이 분할 GBDT를 기존 모델과 혼합하니 세 fold 모두 개선됐다.
  2022 +9.97e-5 / 2023 +3.57e-5 / 2024 +7.39e-5 (최적 가중)
그러나 오차 상관이 0.9995로 사실상 같은 모델이다. 이 이득이
'손잡이 분할' 때문인지 '모델 하나 더 평균한' 분산 감소인지 구분되지 않는다.

대조군: 완전히 같은 설정에서 random_state만 바꾼 전체 모델을 혼합한다.
  - 대조군이 비슷한 이득을 내면 -> 손잡이 분할은 기여 없음, 시드 평균과 동일
  - 손잡이 분할이 뚜렷이 크면 -> 분할 자체에 값이 있음

공정 비교를 위해 exp62와 같은 fold, 같은 학습 구간, 같은 파라미터를 쓴다.
"""
from pathlib import Path
import sys, time, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from features import add_features

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
PARAMS = dict(learning_rate=0.026902980804302, max_iter=600, max_leaf_nodes=15,
              min_samples_leaf=84, l2_regularization=14.625327121207684,
              max_features=0.6298202574719083, max_bins=128,
              early_stopping=False, random_state=42)


def encode(train, valid, cols):
    xt, xv = train[cols].copy(), valid[cols].copy()
    for c in CAT_COLS:
        vals = sorted(train[c].dropna().astype(str).unique())
        m = {v: i for i, v in enumerate(vals)}
        xt[c] = train[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xv[c] = valid[c].astype(str).map(m).fillna(-1).astype(np.int16)
    xt = pd.concat([xt.reset_index(drop=True), add_features(train).reset_index(drop=True)], axis=1)
    xv = pd.concat([xv.reset_index(drop=True), add_features(valid).reset_index(drop=True)], axis=1)
    return xt, xv


def fit_predict(train, valid, cols, seed):
    xt, xv = encode(train, valid, cols)
    p = dict(PARAMS); p["random_state"] = seed
    m = HistGradientBoostingClassifier(**p)
    m.fit(xt, train[TARGET])
    out = m.predict_proba(xv)[:, 1]
    del xt, xv, m; gc.collect()
    return out


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    prev = pd.read_csv(HERE / "results" / "exp62_hand_split_oof.csv.gz")
    rows = []
    for vy in (2022, 2023, 2024):
        valid = df[df.season == vy].reset_index(drop=True)
        train = df[df.season < vy]
        train = train[~((train.game_type == "F") & (train.season <= 2022))].reset_index(drop=True)
        y = valid[TARGET].to_numpy()
        pv = prev[prev.season == vy].set_index(ID).reindex(valid[ID])
        p_base = pv["p_base"].to_numpy()
        p_split = pv["p_split"].to_numpy()
        t0 = time.time()
        p_seed = fit_predict(train, valid, cols, seed=7)
        sec = round(time.time() - t0)

        rec = dict(fold=vy, base=brier_score_loss(y, p_base),
                   seed7=brier_score_loss(y, p_seed),
                   corr_seed=float(np.corrcoef(y - p_base, y - p_seed)[0, 1]),
                   corr_split=float(np.corrcoef(y - p_base, y - p_split)[0, 1]), sec=sec)
        for tag, q in (("seed", p_seed), ("split", p_split)):
            best_w, best_b = 0.0, rec["base"]
            for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                b = brier_score_loss(y, (1 - w) * p_base + w * q)
                if b < best_b:
                    best_b, best_w = b, w
            rec["%s_w" % tag], rec["%s_blend" % tag] = best_w, best_b
            rec["%s_gain" % tag] = (rec["base"] - best_b) * 1e5
        # 셋 다 섞으면?
        b3 = brier_score_loss(y, (p_base + p_seed + p_split) / 3)
        rec["three_gain"] = (rec["base"] - b3) * 1e5
        rows.append(rec)
        print("  %d 시드혼합 %+.2fe-5 (w=%.1f, 상관 %.4f) | 분할혼합 %+.2fe-5 (w=%.1f, 상관 %.4f) "
              "| 3개평균 %+.2fe-5  (%ds)"
              % (vy, rec["seed_gain"], rec["seed_w"], rec["corr_seed"],
                 rec["split_gain"], rec["split_w"], rec["corr_split"],
                 rec["three_gain"], sec), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "exp63_seed_control.csv", index=False, encoding="utf-8-sig")
    print("\n=== 판정 ===")
    for tag in ("seed", "split"):
        g = out["%s_gain" % tag]
        print("  %-6s 2022 %+.2f / 2023 %+.2f / 2024 %+.2f  평균 %+.2fe-5"
              % (tag, g.iloc[0], g.iloc[1], g.iloc[2], g.mean()))
    d = out["split_gain"] - out["seed_gain"]
    print("\n  분할 - 시드 차이: 2022 %+.2f / 2023 %+.2f / 2024 %+.2f  평균 %+.2fe-5"
          % (d.iloc[0], d.iloc[1], d.iloc[2], d.mean()))
    print("  -> 차이가 0 근처면 손잡이 분할은 시드 평균과 구분되지 않는다")
    print("\n  3개(base+seed+split) 평균: 2022 %+.2f / 2023 %+.2f / 2024 %+.2f  평균 %+.2fe-5"
          % tuple(list(out.three_gain) + [out.three_gain.mean()]))


if __name__ == "__main__":
    main()

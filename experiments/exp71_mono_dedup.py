# -*- coding: utf-8 -*-
"""실험 71: 단조 제약과 완전 중복 제거 — '정보 추가'가 아닌 '노이즈 차단' 프레임.

지금까지 도메인 제안은 전부 "정보를 추가한다"였고 전부 실패했다. 그런데
현행 튜닝값은 max_leaf_nodes=15로 깊이가 4단 정도라, 91개 피처 사이에서
3중 상호작용도 겨우 만든다. 신호가 AUC 0.55로 약하니 대부분의 분할은
노이즈를 맞추는 데 쓰인다. 그렇다면 이길 방법은 정보 추가가 아니라
**모델이 노이즈를 맞추지 못하게 막는 것**이고, 그건 도메인 지식이 있어야 한다.

[A] 완전 중복 제거 (정보 손실 0)
  inning ↔ inning_c                 +1.000  (clip(1,12)인데 12 초과가 희소)
  home_win_expectancy ↔ away_win_expectancy  -1.000  (합이 정확히 1)
  asof_pitcher_n ↔ asof_pitcher_pitchmix_n   +1.000
  mid_share ↔ rev_share             -1.000
  max_features=0.63이므로 매 분할에서 91개 중 57개만 후보로 뽑힌다.
  정보가 정확히 0인 컬럼 4개가 그 슬롯을 훔치고 있다.

[B] 단조 제약 (monotonic_cst) — 미시도
  2025 유형 행에서 10분위별 실제 성공률의 스피어만 상관으로 방향을 실측했다.
  strikes_before(-0.500), li(+0.588), asof_pitcher_n(+0.370)은 비단조이므로
  제약하지 않는다. 특히 strikes_before의 비단조성은 2스트라이크 유인구
  때문이며, 야구적으로 제약하면 안 되는 대표 사례다.
"""
from pathlib import Path
import sys, time, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
PARAMS = dict(learning_rate=0.026902980804302, max_iter=600, max_leaf_nodes=15,
              min_samples_leaf=84, l2_regularization=14.625327121207684,
              max_features=0.6298202574719083, max_bins=128,
              early_stopping=False, random_state=42)

DUPES = ["inning_c", "away_win_expectancy", "asof_pitcher_pitchmix_n", "rev_share"]

# 실측 스피어만 |0.90| 이상만. 값은 (피처, 방향).
MONO_A = {  # 보수: 원본 asof 비율만. 정의상 실패 유형이거나 성공 이력이다.
    "asof_pitcher_success_rate": 1, "asof_pitcher_strike_rate": 1,
    "asof_batter_success_rate": 1,
    "asof_pitcher_prev1_game_success_rate": 1,
    "asof_pitcher_prev3_game_success_rate": 1,
    "asof_pitcher_prev5_game_success_rate": 1,
    "asof_pitcher_middle_rate": -1, "asof_pitcher_reverse_rate": -1,
    "asof_pitcher_ball_rate": -1, "asof_batter_middle_rate": -1,
    "asof_pitcher_prev3_game_middle_rate": -1,
    "asof_pitcher_prev5_game_middle_rate": -1,
}
MONO_B = dict(MONO_A)  # 전체: 상황 변수와 도메인 미러 추가
MONO_B.update({"balls_before": -1, "inning": -1, "inning_c": -1,
               "p_succ_shrunk": 1, "p_mid_shrunk": -1})


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


def run(train, valid, cols, drop, mono):
    xt, xv = encode(train, valid, cols)
    if drop:
        keep = [c for c in xt.columns if c not in DUPES]
        xt, xv = xt[keep], xv[keep]
    cst = None
    if mono:
        cst = np.array([mono.get(c, 0) for c in xt.columns], dtype=int)
        for i, c in enumerate(xt.columns):      # 범주형은 제약 불가
            if c in CAT_COLS:
                cst[i] = 0
    p = dict(PARAMS)
    if cst is not None:
        p["monotonic_cst"] = cst
    m = HistGradientBoostingClassifier(**p)
    m.fit(xt, train[TARGET])
    out = m.predict_proba(xv)[:, 1]
    n_feat, n_cst = xt.shape[1], (0 if cst is None else int((cst != 0).sum()))
    del xt, xv, m; gc.collect()
    return out, n_feat, n_cst


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    VAR = [("base", False, None), ("dedup", True, None),
           ("monoA", False, MONO_A), ("dedup_monoB", True, MONO_B)]
    rows, oof = [], []
    for vy in (2022, 2023, 2024):
        valid = df[df.season == vy].reset_index(drop=True)
        train = df[df.season < vy]
        train = train[~((train.game_type == "F") & (train.season <= 2022))].reset_index(drop=True)
        # 채점은 2025 유형 행만 (채점 기준 2차 개정)
        keep = ((valid.game_type == "R") | (valid.season >= 2023)).to_numpy()
        y = valid[TARGET].to_numpy()
        preds = {}
        for name, drop, mono in VAR:
            t0 = time.time()
            p, nf, nc = run(train, valid, cols, drop, mono)
            preds[name] = p
            rows.append(dict(fold=vy, variant=name, n_feat=nf, n_cst=nc,
                             brier=brier_score_loss(y[keep], p[keep]),
                             auc=roc_auc_score(y[keep], p[keep]),
                             sec=round(time.time() - t0)))
            print("  %d %-12s n=%d cst=%d brier=%.6f auc=%.4f (%ds)"
                  % (vy, name, nf, nc, rows[-1]["brier"], rows[-1]["auc"], rows[-1]["sec"]),
                  flush=True)
        o = pd.DataFrame({ID: valid[ID], "season": vy, TARGET: y, "keep": keep})
        for k, v in preds.items():
            o["p_" + k] = v
        oof.append(o)

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "exp71_mono_dedup.csv", index=False, encoding="utf-8-sig")
    pd.concat(oof, ignore_index=True).to_csv(
        HERE / "results" / "exp71_mono_dedup_oof.csv.gz", index=False, compression="gzip")
    print("\n=== base 대비 (e-5, 2025 유형 행 기준) ===")
    print("%-14s %9s %9s %9s %10s" % ("변형", "2022", "2023", "2024", "2022+2024"))
    piv = out.pivot(index="variant", columns="fold", values="brier")
    for v in ["dedup", "monoA", "dedup_monoB"]:
        g = [(piv.loc["base", f] - piv.loc[v, f]) * 1e5 for f in (2022, 2023, 2024)]
        print("%-14s %+9.2f %+9.2f %+9.2f %+10.2f" % (v, g[0], g[1], g[2], (g[0] + g[2]) / 2))
    print("\n※ 판정 우선순위: 1) 2024  2) 2022  3) 2023은 참고만")


if __name__ == "__main__":
    main()

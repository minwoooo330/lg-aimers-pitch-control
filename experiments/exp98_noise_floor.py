# -*- coding: utf-8 -*-
"""실험 98: 노이즈 바닥 측정 — 피처를 바꾸지 않고 시드만 바꿨을 때 Brier 변동폭.

동기: exp97 군별 절제에서 네 군 모두 fold 간 부호가 뒤집히고(G1 -10.92/+1.11,
G4 +2.48/-1.54) 군별 효과의 합이 전체 배치와 부호·크기 모두 어긋났다
(2024 군합 -3.93 vs 전체 +2.99). 이는 학습 무작위성이 지배한다는 징후다.

시드만 바꾼 base를 여러 번 돌려 표준편차를 재면 그것이 노이즈 바닥이다.
이 값은 앞으로 모든 채택/기각 판정의 잣대가 된다. 예컨대 exp63 6분할은
2024 +1.55e-5로 채택됐는데, 바닥이 그보다 크면 그 판정도 재검토 대상이다.

체인 파라미터, 2025 유형 행 채점, clean fold(2022/2024)만.
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
CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
PRM = dict(max_iter=200, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=200,
           l2_regularization=1., early_stopping=False)
SEEDS = [42, 7, 2024, 13, 777, 31]


def enc(tr, va, cols):
    a, b = tr[cols].copy(), va[cols].copy()
    for c in CAT:
        v = sorted(tr[c].dropna().astype(str).unique()); m = {x: i for i, x in enumerate(v)}
        a[c] = tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
        b[c] = va[c].astype(str).map(m).fillna(-1).astype(np.int16)
    return (pd.concat([a.reset_index(drop=True), add_features(tr).reset_index(drop=True)], axis=1),
            pd.concat([b.reset_index(drop=True), add_features(va).reset_index(drop=True)], axis=1))


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    out = {}
    for vy in (2022, 2024):
        va = df[df.season == vy].reset_index(drop=True)
        tr = df[df.season < vy].reset_index(drop=True)
        keep = ((va.game_type == "R") | (va.season >= 2023)).to_numpy()
        y = va[TARGET].to_numpy()
        xa, xb = enc(tr, va, cols)
        bs = []
        for sd in SEEDS:
            t0 = time.time()
            p = dict(PRM); p["random_state"] = sd
            m = HistGradientBoostingClassifier(**p); m.fit(xa, tr[TARGET])
            pr = m.predict_proba(xb)[:, 1]
            b = brier_score_loss(y[keep], pr[keep]); bs.append(b)
            print("  %d seed=%-5d brier=%.6f (%ds)" % (vy, sd, b, round(time.time() - t0)), flush=True)
            del m; gc.collect()
        out[vy] = np.array(bs)
        del xa, xb; gc.collect()
    print("\n=== 노이즈 바닥 (피처 동일, 시드만 변경) ===")
    for vy in (2022, 2024):
        a = out[vy]
        print("  %d: 평균 %.6f  표준편차 %.2fe-5  범위 %.2fe-5 (최소 %.6f ~ 최대 %.6f)"
              % (vy, a.mean(), a.std(ddof=1) * 1e5, (a.max() - a.min()) * 1e5, a.min(), a.max()))
    sd22, sd24 = out[2022].std(ddof=1) * 1e5, out[2024].std(ddof=1) * 1e5
    print("\n=== 판정 잣대 ===")
    print("  단일 시드 비교의 1시그마 = sqrt(2)*SD  ->  2022 %.2fe-5 / 2024 %.2fe-5"
          % (sd22 * np.sqrt(2), sd24 * np.sqrt(2)))
    print("  즉 두 모델(각 1시드)의 차이가 이 값보다 작으면 노이즈와 구분 불가")
    print("\n=== 기존 판정 재조명 ===")
    for nm, g22, g24 in [("exp63 6분할 NN", -0.58, 1.55), ("exp96 휴리스틱 배치", -3.32, 2.99),
                         ("exp97 G4 상황구종", 2.48, -1.54), ("exp71 monoA", 0.99, 8.79)]:
        f22 = "노이즈내" if abs(g22) < sd22 * np.sqrt(2) else "유의"
        f24 = "노이즈내" if abs(g24) < sd24 * np.sqrt(2) else "유의"
        print("  %-22s 2022 %+6.2f(%s)  2024 %+6.2f(%s)" % (nm, g22, f22, g24, f24))


if __name__ == "__main__":
    main()

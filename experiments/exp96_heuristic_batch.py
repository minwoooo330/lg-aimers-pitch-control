# -*- coding: utf-8 -*-
"""실험 76: 휴리스틱 파생변수 2차 배치를 그룹으로 검증.

근거: domain43의 44개 중 내 단독 문턱(clean fold 상관 0.0105)을 통과하는 것은
1개(2.3%)뿐인데 그룹으로는 리더보드 +18.81점을 냈다. 단독 스크리닝은
그룹 효과를 구조적으로 놓친다. 따라서 개별로 거르지 않고 배치로 넣어 잰다.

체인 파라미터 사용(max_iter=200, lr=.06, leaves=31, msl=200).
채점은 2025 유형 행(R 또는 season>=2023)만.
"""
from pathlib import Path
import sys, time, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features
from hfeatures import add_hfeatures

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ID, TARGET = "row_id", "control_success"
CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
PRM = dict(max_iter=200, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=200,
           l2_regularization=1., early_stopping=False, random_state=42)


def enc(tr, va, cols, use_h):
    a, b = tr[cols].copy(), va[cols].copy()
    for c in CAT:
        v = sorted(tr[c].dropna().astype(str).unique()); m = {x: i for i, x in enumerate(v)}
        a[c] = tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
        b[c] = va[c].astype(str).map(m).fillna(-1).astype(np.int16)
    pa = [a.reset_index(drop=True), add_features(tr).reset_index(drop=True)]
    pb = [b.reset_index(drop=True), add_features(va).reset_index(drop=True)]
    if use_h:
        pa.append(add_hfeatures(tr).reset_index(drop=True))
        pb.append(add_hfeatures(va).reset_index(drop=True))
    return pd.concat(pa, axis=1), pd.concat(pb, axis=1)


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    rows, oof = [], []
    for vy in (2022, 2023, 2024):
        va = df[df.season == vy].reset_index(drop=True)
        tr = df[df.season < vy].reset_index(drop=True)
        keep = ((va.game_type == "R") | (va.season >= 2023)).to_numpy()
        y = va[TARGET].to_numpy()
        pr = {}
        for name, use_h in [("base", False), ("heur", True)]:
            t0 = time.time()
            xa, xb = enc(tr, va, cols, use_h)
            m = HistGradientBoostingClassifier(**PRM); m.fit(xa, tr[TARGET])
            p = m.predict_proba(xb)[:, 1]; pr[name] = p
            rows.append(dict(fold=vy, variant=name, n_feat=xa.shape[1],
                             brier=brier_score_loss(y[keep], p[keep]),
                             auc=roc_auc_score(y[keep], p[keep]), sec=round(time.time() - t0)))
            print("  %d %-5s n=%3d brier=%.6f auc=%.4f (%ds)" % (
                vy, name, xa.shape[1], rows[-1]["brier"], rows[-1]["auc"], rows[-1]["sec"]), flush=True)
            del xa, xb, m; gc.collect()
        o = pd.DataFrame({ID: va[ID], "season": vy, TARGET: y, "keep": keep})
        for k, v in pr.items(): o["p_" + k] = v
        oof.append(o)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "exp76_heuristic_batch.csv", index=False, encoding="utf-8-sig")
    pd.concat(oof, ignore_index=True).to_csv(
        HERE / "results" / "exp76_heuristic_oof.csv.gz", index=False, compression="gzip")
    piv = out.pivot(index="variant", columns="fold", values="brier")
    print("\n=== base 대비 (e-5, 2025 유형 행) ===")
    g = [(piv.loc["base", f] - piv.loc["heur", f]) * 1e5 for f in (2022, 2023, 2024)]
    print("  2022 %+.2f / 2023 %+.2f / 2024 %+.2f | clean평균 %+.2f" % (g[0], g[1], g[2], (g[0] + g[2]) / 2))
    ap = out.pivot(index="variant", columns="fold", values="auc")
    print("  AUC  2022 %.4f->%.4f  2024 %.4f->%.4f" % (
        ap.loc["base", 2022], ap.loc["heur", 2022], ap.loc["base", 2024], ap.loc["heur", 2024]))


if __name__ == "__main__":
    main()

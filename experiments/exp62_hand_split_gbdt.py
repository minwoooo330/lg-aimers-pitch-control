# -*- coding: utf-8 -*-
"""실험 62: 손잡이 매치업 분할 GBDT.

손잡이NN(p_same/p_opp 임베딩 분리)이 성공했다. 축 자체는 3-fold 검증을
통과한 유일한 축이다(깨끗한 fold 최소 상관 0.0105, 기각된 카운트축은 0.0073).
그런데 그 구조는 NN에만 들어갔고, 앙상블 가중의 다수는 GBDT다.

여기서는 같은 축을 GBDT로 옮긴다. 임베딩이 없으니 데이터를 층화해서
매치업별로 트리를 따로 키운다. 트리 전체가 그 매치업에 특화되므로
임베딩보다 강한 분할이다.

  same : pitcher_hand == batter_hand   (약 654k행)
  opp  : pitcher_hand != batter_hand   (약 660k행)

위험은 데이터가 반으로 줄어드는 것이다(exp60 최근시즌창이 이걸로 실패).
다만 무작위 절반이 아니라 투구 메커니즘이 실제로 다른 층화라는 점이 다르다.

판정은 단독 Brier가 아니라 기존 모델과의 혼합 이득으로 한다.
2023 fold는 퓨처스 오염으로 모든 신호를 2~5배 부풀리므로 R리그 성능을 함께 본다.
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


def fit_predict(train, valid, cols, seed=42):
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
    df["_same"] = (df.pitcher_hand == df.batter_hand).to_numpy()
    print("같은손 %d행 / 반대손 %d행" % ((df._same).sum(), (~df._same).sum()), flush=True)

    rows, oof = [], []
    for vy in (2022, 2023, 2024):
        valid = df[df.season == vy].reset_index(drop=True)
        train = df[df.season < vy]
        train = train[~((train.game_type == "F") & (train.season <= 2022))].reset_index(drop=True)
        y = valid[TARGET].to_numpy()
        mR = (valid.game_type == "R").to_numpy()
        t0 = time.time()

        p_base = fit_predict(train, valid, cols)
        t1 = time.time()
        p_split = np.zeros(len(valid))
        for same in (True, False):
            tr = train[train._same == same]
            va_idx = np.flatnonzero((valid._same == same).to_numpy())
            va = valid.iloc[va_idx]
            p_split[va_idx] = fit_predict(tr, va, cols)
        t2 = time.time()

        rec = dict(fold=vy, n_train=len(train),
                   base=brier_score_loss(y, p_base), split=brier_score_loss(y, p_split),
                   base_R=brier_score_loss(y[mR], p_base[mR]),
                   split_R=brier_score_loss(y[mR], p_split[mR]),
                   auc_base=roc_auc_score(y, p_base), auc_split=roc_auc_score(y, p_split),
                   corr=float(np.corrcoef(y - p_base, y - p_split)[0, 1]),
                   sec_base=round(t1 - t0), sec_split=round(t2 - t1))
        # 혼합 격자
        best_w, best_b = 0.0, rec["base"]
        for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            b = brier_score_loss(y, (1 - w) * p_base + w * p_split)
            if b < best_b:
                best_b, best_w = b, w
            rec["blend_%.1f" % w] = b
        rec["best_w"], rec["best_blend"] = best_w, best_b
        rows.append(rec)
        oof.append(pd.DataFrame({ID: valid[ID], "season": vy, TARGET: y,
                                 "p_base": p_base, "p_split": p_split}))
        print("  %d 단독 base=%.6f split=%.6f (R만 %.6f vs %.6f) 상관=%.4f "
              "최적혼합 w=%.1f %.6f (%ds+%ds)"
              % (vy, rec["base"], rec["split"], rec["base_R"], rec["split_R"],
                 rec["corr"], best_w, best_b, rec["sec_base"], rec["sec_split"]), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "exp62_hand_split_gbdt.csv", index=False, encoding="utf-8-sig")
    pd.concat(oof, ignore_index=True).to_csv(
        HERE / "results" / "exp62_hand_split_oof.csv.gz", index=False, compression="gzip")

    print("\n=== 요약 (base 대비, e-5 단위) ===")
    print("%-6s %10s %10s %12s %10s" % ("fold", "split단독", "R만", "혼합w=0.3", "최적혼합"))
    for _, r in out.iterrows():
        print("%-6d %10.2f %10.2f %12.2f %10.2f (w=%.1f)" % (
            r.fold, (r.base - r.split) * 1e5, (r.base_R - r.split_R) * 1e5,
            (r.base - r["blend_0.3"]) * 1e5, (r.base - r.best_blend) * 1e5, r.best_w))
    pooled_base = float(np.mean(out.base)); pooled_split = float(np.mean(out.split))
    print("\n3-fold 평균  base %.6f  split %.6f  차이 %+.2fe-5"
          % (pooled_base, pooled_split, (pooled_base - pooled_split) * 1e5))
    print("2022/2024만  base %.6f  split %.6f  차이 %+.2fe-5"
          % (out[out.fold != 2023].base.mean(), out[out.fold != 2023].split.mean(),
             (out[out.fold != 2023].base.mean() - out[out.fold != 2023].split.mean()) * 1e5))


if __name__ == "__main__":
    main()

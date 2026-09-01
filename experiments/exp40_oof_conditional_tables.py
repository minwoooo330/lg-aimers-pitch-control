# -*- coding: utf-8 -*-
"""실험 40: 시즌 단위 out-of-fold 투수 조건부 통계표.

exp39는 학습 행 자신이 포함된 데이터로 표를 만들어 target encoding 누수가 생겼고
2023 fold에서 Brier 0.2487 -> 0.2508, AUC 0.5397 -> 0.5324로 악화했다.
여기서는 시즌 s의 행에는 s 이전 시즌만으로 만든 표를 붙여 학습·검증 조건을 맞춘다.

선별 근거(잔차 상관 스크리닝, 2024 기준):
  투수x타자손잡이x카운트우열 +0.0188  <- 최상
  투수x타자손잡이            +0.0158
  투수x카운트(12셀)          +0.0101
  투수x이닝 / 주자수 / LI구간  ~0.000  <- 제외
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
K = 200.0
CELLS = {"hand": ["pitcher_id", "batter_hand"],
         "cnt": ["pitcher_id", "_cnt"],
         "handadv": ["pitcher_id", "batter_hand", "_adv"]}


def prep(d):
    d = d.copy()
    d["_cnt"] = d.balls_before.astype(str) + d.strikes_before.astype(str)
    d["_adv"] = np.sign(d.strikes_before - d.balls_before).astype(int)
    return d


def build(hist):
    """퓨처스 ABS 이전(2019~2022)은 라벨 규칙이 달라 표 산출에서 제외한다."""
    h = hist[~((hist.game_type == "F") & (hist.season <= 2022))]
    if len(h) == 0:
        return None
    prior = h[TARGET].mean()
    g = h.groupby("pitcher_id")[TARGET].agg(["sum", "count"])
    marg = (g["sum"] + K * prior) / (g["count"] + K)
    t = {"marg": marg, "n": np.log1p(g["count"]), "prior": prior}
    gR = h[h.game_type == "R"].groupby("pitcher_id")[TARGET].agg(["sum", "count"])
    t["margR"] = (gR["sum"] + K * prior) / (gR["count"] + K)
    last = h[h.season == h.season.max()].groupby("pitcher_id")[TARGET].agg(["sum", "count"])
    t["marglast"] = (last["sum"] + K * prior) / (last["count"] + K)
    for name, cols in CELLS.items():
        gg = h.groupby(cols, observed=True)[TARGET].agg(["sum", "count"])
        pm = gg.index.get_level_values(0).map(marg)
        t[name] = (((gg["sum"] + K * pm) / (gg["count"] + K)) - pm)
        t[name + "_n"] = np.log1p(gg["count"])
    return t


def apply_t(d, t, groups):
    out = pd.DataFrame(index=range(len(d)))
    if t is None:
        for c in (["pt_cmd_all", "pt_n_all", "pt_cmd_R", "pt_cmd_last"] if "static" in groups else []):
            out[c] = np.nan
        for name in (CELLS if "cond" in groups else []):
            out[f"pt_dev_{name}"] = np.nan; out[f"pt_n_{name}"] = np.nan
        return out
    pid = d.pitcher_id.to_numpy()
    if "static" in groups:
        out["pt_cmd_all"] = pd.Series(pid).map(t["marg"]).to_numpy()
        out["pt_n_all"] = pd.Series(pid).map(t["n"]).to_numpy()
        out["pt_cmd_R"] = pd.Series(pid).map(t["margR"]).to_numpy()
        out["pt_cmd_last"] = pd.Series(pid).map(t["marglast"]).to_numpy()
    if "cond" in groups:
        keys = {"hand": [d.pitcher_id, d.batter_hand],
                "cnt": [d.pitcher_id, d._cnt],
                "handadv": [d.pitcher_id, d.batter_hand, d._adv]}
        for name, arrs in keys.items():
            idx = pd.MultiIndex.from_arrays([a.to_numpy() for a in arrs])
            out[f"pt_dev_{name}"] = t[name].reindex(idx).to_numpy()
            out[f"pt_n_{name}"] = t[name + "_n"].reindex(idx).to_numpy()
    return out


def encode(train, valid, cols, df_all, valid_year, groups):
    xt, xv = train[cols].copy(), valid[cols].copy()
    for c in CAT_COLS:
        vals = sorted(train[c].dropna().astype(str).unique())
        m = {v: i for i, v in enumerate(vals)}
        xt[c] = train[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xv[c] = valid[c].astype(str).map(m).fillna(-1).astype(np.int16)
    xt = pd.concat([xt.reset_index(drop=True), add_features(train).reset_index(drop=True)], axis=1)
    xv = pd.concat([xv.reset_index(drop=True), add_features(valid).reset_index(drop=True)], axis=1)
    if not groups:
        return xt, xv
    # 학습 행: 자기 시즌 이전만으로 만든 표 (시즌 단위 out-of-fold)
    parts = []
    for s in sorted(train.season.unique()):
        sub = train[train.season == s]
        parts.append(apply_t(sub, build(df_all[df_all.season < s]), groups))
    tf = pd.concat(parts, ignore_index=True)
    # 검증 행: 검증 연도 이전 전체 (2025 운용과 동일)
    vf = apply_t(valid, build(df_all[df_all.season < valid_year]), groups)
    return (pd.concat([xt, tf], axis=1), pd.concat([xv, vf.reset_index(drop=True)], axis=1))


def main():
    df = prep(pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig"))
    cols = [c for c in df.columns if c not in (ID, TARGET, "_cnt", "_adv")]
    rows = []
    for vy in (2023, 2024):
        valid = df[df.season == vy]
        y = valid[TARGET].to_numpy()
        train = df[df.season < vy]
        train = train[~((train.game_type == "F") & (train.season <= 2022))]
        train = train.sort_values("season").reset_index(drop=True)
        for name, groups in [("base", []), ("cond", ["cond"]), ("static_cond", ["static", "cond"])]:
            t0 = time.time()
            xt, xv = encode(train, valid, cols, df, vy, groups)
            m = HistGradientBoostingClassifier(**PARAMS)
            m.fit(xt, train[TARGET])
            p = m.predict_proba(xv)[:, 1]
            mR = (valid.game_type == "R").to_numpy()
            rec = dict(valid_year=vy, variant=name, n_feat=xt.shape[1],
                       brier=brier_score_loss(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                       auc=roc_auc_score(y, p), brier_R=brier_score_loss(y[mR], p[mR]),
                       seconds=round(time.time() - t0, 1))
            rows.append(rec)
            print(f"  {vy} {name:12s} n={xt.shape[1]:3d} brier={rec['brier']:.6f} "
                  f"R={rec['brier_R']:.6f} auc={rec['auc']:.4f} ({rec['seconds']:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "exp40_oof_conditional_tables.csv", index=False, encoding="utf-8-sig")
    print("\n=== base 대비 ===")
    for vy in (2023, 2024):
        s = out[out.valid_year == vy].set_index("variant")
        for v in s.index:
            print(f"  {vy} {v:12s} 전체 {s.loc['base','brier']-s.loc[v,'brier']:+.6f}  "
                  f"R만 {s.loc['base','brier_R']-s.loc[v,'brier_R']:+.6f}  AUC {s.loc[v,'auc']:.4f}")


if __name__ == "__main__":
    main()

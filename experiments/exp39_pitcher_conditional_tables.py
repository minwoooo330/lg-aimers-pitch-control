# -*- coding: utf-8 -*-
"""실험 39: 학습 데이터로 미리 계산한 투수 조건부 통계표.

근거: 공식 asof_*는 투수의 '주변(marginal)' 성공률만 준다. 투수×카운트,
투수×타자손잡이, 투수×압박구간 같은 조건부 성향은 제공되지 않고,
pitcher_id가 원시 숫자로 들어가 트리가 이 상호작용을 만들지 못한다.

규칙 적합성: 과거 연도 학습 데이터로만 표를 만들어 각 행에 독립적으로 결합한다.
test 내부 집계를 쓰지 않으므로 한 행만 넣어도 같은 예측이 나온다.

체제 보정: 퓨처스 2019~2022는 다른 라벨 규칙이므로 표 산출에서 제외한다.
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
K = 200.0   # 조건부 셀을 투수 주변값으로 당기는 shrinkage 강도


def li_bucket(s):
    return pd.cut(s, [-np.inf, 0.7, 1.3, 2.5, np.inf], labels=[0, 1, 2, 3]).astype(float)


def build_tables(hist):
    """과거 연도만으로 투수 통계표를 만든다. 퓨처스 ABS 이전은 제외."""
    h = hist[~((hist.game_type == "F") & (hist.season <= 2022))].copy()
    prior = h[TARGET].mean()
    t = {}
    # 1) 투수 주변값 (전체 / 1군만 / 최근시즌)
    g = h.groupby("pitcher_id")[TARGET].agg(["sum", "count"])
    t["marg"] = ((g["sum"] + K * prior) / (g["count"] + K)).rename("pt_cmd_all")
    t["marg_n"] = np.log1p(g["count"]).rename("pt_n_all")
    hR = h[h.game_type == "R"]
    gR = hR.groupby("pitcher_id")[TARGET].agg(["sum", "count"])
    t["margR"] = ((gR["sum"] + K * prior) / (gR["count"] + K)).rename("pt_cmd_R")
    last = h.season.max()
    hL = h[h.season == last]
    gL = hL.groupby("pitcher_id")[TARGET].agg(["sum", "count"])
    t["marg_last"] = ((gL["sum"] + K * prior) / (gL["count"] + K)).rename("pt_cmd_last")
    t["prior"] = prior
    # 2) 조건부 셀: 투수 주변값으로 shrink 후 편차만 남긴다
    base = t["marg"]
    h = h.assign(_cnt=h.balls_before.astype(str) + h.strikes_before.astype(str),
                 _li=li_bucket(h.li))
    for key, cols in [("count", ["pitcher_id", "_cnt"]),
                      ("hand", ["pitcher_id", "batter_hand"]),
                      ("li", ["pitcher_id", "_li"])]:
        gg = h.groupby(cols, observed=True)[TARGET].agg(["sum", "count"])
        pm = gg.index.get_level_values(0).map(base)
        val = (gg["sum"] + K * pm) / (gg["count"] + K)
        t[key] = (val - pm).rename(f"pt_dev_{key}")
        t[f"{key}_n"] = np.log1p(gg["count"]).rename(f"pt_n_{key}")
    return t


def apply_tables(df, t, groups):
    out = pd.DataFrame(index=df.index)
    pid = df.pitcher_id
    if "static" in groups:
        out["pt_cmd_all"] = pid.map(t["marg"]).to_numpy()
        out["pt_n_all"] = pid.map(t["marg_n"]).to_numpy()
        out["pt_cmd_R"] = pid.map(t["margR"]).to_numpy()
        out["pt_cmd_last"] = pid.map(t["marg_last"]).to_numpy()
        out["pt_cmd_last_minus_all"] = out["pt_cmd_last"] - out["pt_cmd_all"]
    if "cond" in groups:
        keys = {"count": pd.MultiIndex.from_arrays(
                    [pid, df.balls_before.astype(str) + df.strikes_before.astype(str)]),
                "hand": pd.MultiIndex.from_arrays([pid, df.batter_hand]),
                "li": pd.MultiIndex.from_arrays([pid, li_bucket(df.li)])}
        for k, idx in keys.items():
            out[f"pt_dev_{k}"] = t[k].reindex(idx).to_numpy()
            out[f"pt_n_{k}"] = t[f"{k}_n"].reindex(idx).to_numpy()
    return out


def encode(train, valid, cols, tables, groups):
    xt, xv = train[cols].copy(), valid[cols].copy()
    for c in CAT_COLS:
        vals = sorted(train[c].dropna().astype(str).unique())
        m = {v: i for i, v in enumerate(vals)}
        xt[c] = train[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xv[c] = valid[c].astype(str).map(m).fillna(-1).astype(np.int16)
    xt = pd.concat([xt.reset_index(drop=True), add_features(train).reset_index(drop=True)], axis=1)
    xv = pd.concat([xv.reset_index(drop=True), add_features(valid).reset_index(drop=True)], axis=1)
    if groups:
        xt = pd.concat([xt, apply_tables(train, tables, groups).reset_index(drop=True)], axis=1)
        xv = pd.concat([xv, apply_tables(valid, tables, groups).reset_index(drop=True)], axis=1)
    return xt, xv


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    rows = []
    for valid_year in (2023, 2024):
        valid = df[df.season == valid_year]
        y = valid[TARGET].to_numpy()
        # 표는 검증 연도 이전 데이터로만 만든다 (2025 운용 방식과 동일)
        tables = build_tables(df[df.season < valid_year])
        train = df[df.season < valid_year]
        train = train[~((train.game_type == "F") & (train.season <= 2022))]
        for name, groups in [("base", []), ("static", ["static"]),
                             ("cond", ["cond"]), ("static_cond", ["static", "cond"])]:
            t0 = time.time()
            xt, xv = encode(train, valid, cols, tables, groups)
            m = HistGradientBoostingClassifier(**PARAMS)
            m.fit(xt, train[TARGET])
            p = m.predict_proba(xv)[:, 1]
            rec = dict(valid_year=valid_year, variant=name, n_feat=xt.shape[1],
                       brier=brier_score_loss(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                       auc=roc_auc_score(y, p), seconds=round(time.time() - t0, 1))
            mR = (valid.game_type == "R").to_numpy()
            rec["brier_R"] = brier_score_loss(y[mR], p[mR])
            rows.append(rec)
            print(f"  {valid_year} {name:12s} n_feat={xt.shape[1]:3d} brier={rec['brier']:.6f} "
                  f"R={rec['brier_R']:.6f} auc={rec['auc']:.4f} ({rec['seconds']:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "exp39_pitcher_conditional_tables.csv", index=False, encoding="utf-8-sig")
    print("\n=== base 대비 개선 ===")
    for vy in (2023, 2024):
        s = out[out.valid_year == vy].set_index("variant")
        for v in s.index:
            print(f"  {vy} {v:12s} 전체 {s.loc['base','brier']-s.loc[v,'brier']:+.6f}  "
                  f"R만 {s.loc['base','brier_R']-s.loc[v,'brier_R']:+.6f}  AUC {s.loc[v,'auc']:.4f}")


if __name__ == "__main__":
    main()

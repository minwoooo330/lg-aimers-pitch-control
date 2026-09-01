# -*- coding: utf-8 -*-
"""실험 41: Trackman을 cold-start 보조 정보로 사용한다.

기존 exp07~09은 Trackman 프로필을 전체 행에 균일하게 넣어 무효였다. 표본이 많은
투수에게는 asof_*가 제구력을 직접 재고 있어 물리 프로필이 중복이기 때문이다.

2026-08-18 잔차 진단에서 신호가 저표본 구간에 집중됨을 확인했다(2024 기준).
  구종 가짓수    전체 +0.0078 / 저표본 +0.0322 / 고표본 +0.0012
  구종 엔트로피   전체 +0.0066 / 저표본 +0.0200 / 고표본 +0.0022
  구속 표준편차   전체 +0.0071 / 저표본 +0.0153 / 고표본 +0.0047
야구적 해석: 구종이 많고 확립된 투수는 경력이 쌓인 선수다. 표본이 적어 asof가
아직 그를 모를 때 Trackman이 그 공백을 메운다. KBO 신인은 퓨처스를 거치므로
메인 데이터 F 행으로 ID 대응이 되고 과거 Trackman 프로필이 존재한다.
"""
from pathlib import Path
import sys, time, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from features import add_features
from trackman_features import match_exact_games, build_pitcher_mapping

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
PARAMS = dict(learning_rate=0.026902980804302, max_iter=600, max_leaf_nodes=15,
              min_samples_leaf=84, l2_regularization=14.625327121207684,
              max_features=0.6298202574719083, max_bins=128,
              early_stopping=False, random_state=42)
K = 200.0
TMC = ["trackman_id", "season", "trackman_game_id", "pitch_no", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before", "pitcher_trackman_id", "pitcher_hand",
       "batter_hand", "tagged_pitch_type", "pitch_type_group", "rel_speed", "spin_rate",
       "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
P = "pitcher_trackman_id"
COLD_CUT = 1646.0


def prep(d):
    d = d.copy()
    d["_cnt"] = d.balls_before.astype(str) + d.strikes_before.astype(str)
    d["_adv"] = np.sign(d.strikes_before - d.balls_before).astype(int)
    return d


def tm_profile(hist):
    """저표본 구간에서 신호가 확인된 항목만 만든다. 타깃을 쓰지 않는 물리 프로필."""
    h = hist.copy()
    sign = h.pitcher_hand.map({"Right": 1.0, "Left": -1.0})
    h["rs"] = h.rel_side * sign
    h["brk"] = np.hypot(h.induced_vert_break, h.horz_break)
    out = pd.DataFrame(index=pd.Index(sorted(h[P].unique()), name=P))
    sh = h.groupby([P, "tagged_pitch_type"]).size().rename("n").reset_index()
    sh["p"] = sh.n / sh.groupby(P).n.transform("sum")
    out["tmc_repertoire_n"] = sh[sh.p >= 0.05].groupby(P).size()
    out["tmc_repertoire_entropy"] = sh.assign(e=-sh.p * np.log(sh.p + 1e-12)).groupby(P).e.sum()
    out["tmc_velo_std"] = h.groupby(P).rel_speed.std()
    out["tmc_velo_loss"] = (h.rel_speed - h.zone_speed).groupby(h[P]).mean()
    out["tmc_break_std"] = h.groupby(P).brk.std()
    fb = h[h.pitch_type_group == "fastball"]
    g = fb.groupby([P, "trackman_game_id"])
    n_in = g.size()
    ext = g.extension.std()[n_in >= 15]
    out["tmc_ingame_ext_scatter"] = ext.groupby(level=0).mean()
    out["tmc_total_n"] = np.log1p(h.groupby(P).size())
    return out


def cond_tables(hist):
    h = hist[~((hist.game_type == "F") & (hist.season <= 2022))]
    if len(h) == 0:
        return None
    prior = h[TARGET].mean()
    g = h.groupby("pitcher_id")[TARGET].agg(["sum", "count"])
    marg = (g["sum"] + K * prior) / (g["count"] + K)
    t = {}
    cells = {"hand": ["pitcher_id", "batter_hand"], "cnt": ["pitcher_id", "_cnt"],
             "handadv": ["pitcher_id", "batter_hand", "_adv"]}
    for name, colset in cells.items():
        gg = h.groupby(colset, observed=True)[TARGET].agg(["sum", "count"])
        pm = gg.index.get_level_values(0).map(marg)
        t[name] = ((gg["sum"] + K * pm) / (gg["count"] + K)) - pm
        t[name + "_n"] = np.log1p(gg["count"])
    return t


def cond_apply(d, t):
    out = pd.DataFrame(index=range(len(d)))
    names = ["hand", "cnt", "handadv"]
    if t is None:
        for n in names:
            out["pt_dev_" + n] = np.nan
            out["pt_n_" + n] = np.nan
        return out
    keys = {"hand": [d.pitcher_id, d.batter_hand], "cnt": [d.pitcher_id, d._cnt],
            "handadv": [d.pitcher_id, d.batter_hand, d._adv]}
    for n, arrs in keys.items():
        idx = pd.MultiIndex.from_arrays([a.to_numpy() for a in arrs])
        out["pt_dev_" + n] = t[n].reindex(idx).to_numpy()
        out["pt_n_" + n] = t[n + "_n"].reindex(idx).to_numpy()
    return out


def tm_apply(d, prof, m2t, interact):
    tmid = d.pitcher_id.map(m2t)
    out = prof.reindex(tmid.to_numpy()).reset_index(drop=True)
    if interact:
        # 표본이 적을수록 Trackman에 무게가 실리도록 명시적 상호작용을 준다
        w = 1.0 / (1.0 + d.asof_pitcher_n.fillna(0).to_numpy() / 1000.0)
        out["tmc_cold_weight"] = w
        for c in ["tmc_repertoire_n", "tmc_repertoire_entropy", "tmc_velo_std"]:
            out[c + "_x_cold"] = out[c].to_numpy() * w
    return out


def main():
    df = prep(pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig"))
    cols = [c for c in df.columns if c not in (ID, TARGET, "_cnt", "_adv")]
    tm = pd.read_csv(HERE / "data" / "trackman_history.csv", usecols=TMC)
    mg, ts, matches = match_exact_games(df, tm)
    del tm
    gc.collect()
    rows = []
    for vy in (2023, 2024):
        valid = df[df.season == vy]
        y = valid[TARGET].to_numpy()
        train = df[df.season < vy]
        train = train[~((train.game_type == "F") & (train.season <= 2022))]
        train = train.sort_values("season").reset_index(drop=True)
        mapping = build_pitcher_mapping(mg, ts, matches, vy)
        m2t = dict(zip(mapping.pitcher_id, mapping.pitcher_trackman_id))
        prof = tm_profile(ts[ts.season < vy])
        print("[%d] 학습 %d행, 투수 대응 %d명" % (vy, len(train), len(mapping)), flush=True)

        xt0, xv0 = train[cols].copy(), valid[cols].copy()
        for c in CAT_COLS:
            vals = sorted(train[c].dropna().astype(str).unique())
            m = {v: i for i, v in enumerate(vals)}
            xt0[c] = train[c].astype(str).map(m).fillna(-1).astype(np.int16)
            xv0[c] = valid[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xt0 = pd.concat([xt0.reset_index(drop=True), add_features(train).reset_index(drop=True)], axis=1)
        xv0 = pd.concat([xv0.reset_index(drop=True), add_features(valid).reset_index(drop=True)], axis=1)

        # 조건부 표는 타깃을 쓰므로 시즌 단위 out-of-fold로 만든다
        ctr = pd.concat([cond_apply(train[train.season == s], cond_tables(df[df.season < s]))
                         for s in sorted(train.season.unique())], ignore_index=True)
        cva = cond_apply(valid, cond_tables(df[df.season < vy])).reset_index(drop=True)
        # Trackman 프로필은 타깃을 쓰지 않으므로 폴드 공통 프로필로 충분하다
        ttr = tm_apply(train, prof, m2t, True).reset_index(drop=True)
        tva = tm_apply(valid, prof, m2t, True).reset_index(drop=True)
        plain = [c for c in ttr.columns if not c.endswith("_x_cold") and c != "tmc_cold_weight"]

        variants = [("base", []), ("tm_plain", ["tmp"]), ("tm_cold", ["tm"]),
                    ("cond", ["c"]), ("cond_tm_cold", ["c", "tm"])]
        for name, parts in variants:
            t0 = time.time()
            xt, xv = [xt0], [xv0]
            if "c" in parts:
                xt.append(ctr); xv.append(cva)
            if "tmp" in parts:
                xt.append(ttr[plain]); xv.append(tva[plain])
            if "tm" in parts:
                xt.append(ttr); xv.append(tva)
            XT, XV = pd.concat(xt, axis=1), pd.concat(xv, axis=1)
            model = HistGradientBoostingClassifier(**PARAMS)
            model.fit(XT, train[TARGET])
            p = model.predict_proba(XV)[:, 1]
            low = (valid.asof_pitcher_n.fillna(0) <= COLD_CUT).to_numpy()
            rec = dict(valid_year=vy, variant=name, n_feat=XT.shape[1],
                       brier=brier_score_loss(y, p), auc=roc_auc_score(y, p),
                       logloss=log_loss(y, p, labels=[0, 1]),
                       brier_lowN=brier_score_loss(y[low], p[low]),
                       brier_highN=brier_score_loss(y[~low], p[~low]),
                       auc_lowN=roc_auc_score(y[low], p[low]),
                       seconds=round(time.time() - t0, 1))
            rows.append(rec)
            print("  %d %-13s n=%3d brier=%.6f 저표본=%.6f 고표본=%.6f auc=%.4f (%ds)"
                  % (vy, name, XT.shape[1], rec["brier"], rec["brier_lowN"],
                     rec["brier_highN"], rec["auc"], rec["seconds"]), flush=True)
            del XT, XV
            gc.collect()

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results" / "exp41_trackman_coldstart.csv", index=False, encoding="utf-8-sig")
    print("\n=== base 대비 ===")
    for vy in (2023, 2024):
        s = out[out.valid_year == vy].set_index("variant")
        for v in s.index:
            print("  %d %-13s 전체 %+.6f  저표본 %+.6f  고표본 %+.6f  AUC %.4f  저표본AUC %.4f"
                  % (vy, v, s.loc["base", "brier"] - s.loc[v, "brier"],
                     s.loc["base", "brier_lowN"] - s.loc[v, "brier_lowN"],
                     s.loc["base", "brier_highN"] - s.loc[v, "brier_highN"],
                     s.loc[v, "auc"], s.loc[v, "auc_lowN"]))


if __name__ == "__main__":
    main()

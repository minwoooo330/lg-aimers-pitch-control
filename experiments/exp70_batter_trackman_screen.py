# -*- coding: utf-8 -*-
"""실험 70: 타자 Trackman ID 매핑 사전 검사 (학습 없음, 잔차 상관만).

배경: Trackman 상한 0.9점 계산은 '투수의 고정 실력' 채널에 대한 것이었다.
투수간 제구율 SD 0.0435 x Trackman 설명분산 7.1%로 나온 값이며,
asof_pitcher_success_rate가 같은 대상을 직접 재고 있어 증분이 소멸한다는 논리다.
타자 축에는 그 논리가 적용되지 않으므로 이론으로 기각하지 않고 싸게 재본다.

절차: 경기 지문 매칭 -> 행 단위 정렬 -> batter_id <-> batter_trackman_id 투표
     -> 타자 Trackman 프로필 -> 2022/2024 잔차 상관

문턱(자체 교정): 채택된 손잡이 축 clean-fold 최소 0.0105, 기각된 카운트 축 0.0073.
2023은 퓨처스 오염으로 2~5배 부풀므로 판정에서 제외(참고만).

타자 프로필의 의미: Trackman에 남은 것은 '투수들이 이 타자에게 어떻게 던졌는가'다.
메인의 asof_batter_*는 결과(제구 성공률)만 주고 물리적 접근 방식은 주지 않는다.
"""
from pathlib import Path
import sys, gc
from collections import defaultdict
import numpy as np
import pandas as pd
from trackman_features import match_exact_games

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
B = "batter_trackman_id"
TARGET = "control_success"
MAIN = ["row_id", "season", "game_type", "inning", "top_bottom", "balls_before",
        "strikes_before", "outs_before", "pitcher_id", "batter_id", "pitcher_hand",
        "batter_hand", "asof_batter_n", "asof_batter_success_rate", TARGET]
TMC = ["season", "trackman_game_id", "pitch_no", "inning", "top_bottom", "balls_before",
       "strikes_before", "outs_before", "pitcher_trackman_id", "batter_trackman_id",
       "pitcher_hand", "batter_hand", "tagged_pitch_type", "pitch_type_group",
       "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
       "rel_height", "rel_side", "zone_speed"]


def build_batter_mapping(main_g, tm_sorted, matches, cutoff, min_votes=20, min_purity=0.99):
    """투수 매핑과 같은 절차를 타자에 적용한다. 타깃을 쓰지 않으므로 누수가 아니다."""
    pairs = matches[matches["season"] < cutoff]
    mi = main_g.groupby("_game_idx", sort=False).indices
    ti = tm_sorted.groupby("trackman_game_id", sort=False).indices
    left, right = [], []
    for row in pairs.itertuples(index=False):
        a, b = mi[row.main_game_idx], ti[row.trackman_game_id]
        if len(a) != len(b):
            continue
        left.append(main_g.iloc[a]["batter_id"].to_numpy())
        right.append(tm_sorted.iloc[b][B].to_numpy())
    if not left:
        return pd.DataFrame(columns=["batter_id", B, "votes", "purity"])
    v = pd.DataFrame({"batter_id": np.concatenate(left), B: np.concatenate(right)})
    votes = v.groupby(["batter_id", B]).size().rename("votes").reset_index()
    tot = votes.groupby("batter_id")["votes"].sum().rename("total")
    votes = votes.join(tot, on="batter_id")
    votes["purity"] = votes["votes"] / votes["total"]
    best = (votes.sort_values(["batter_id", "votes"], ascending=[True, False])
            .drop_duplicates("batter_id"))
    best = best[(best["votes"] >= min_votes) & (best["purity"] >= min_purity)]
    best = best.sort_values(["purity", "votes"], ascending=False).drop_duplicates(B)
    return best.reset_index(drop=True)


def batter_profile(hist):
    """이 타자에게 투수들이 어떻게 던졌는가. 메인 asof_batter_*는 결과만 준다."""
    h = hist.copy()
    sgn = h.pitcher_hand.map({"Right": 1.0, "Left": -1.0})
    h["hb"] = h.horz_break * sgn
    h["brk"] = np.hypot(h.induced_vert_break, h.horz_break)
    g = h.groupby(B)
    out = pd.DataFrame(index=pd.Index(sorted(h[B].unique()), name=B))
    out["bt_velo_mean"] = g.rel_speed.mean()
    out["bt_velo_std"] = g.rel_speed.std()
    out["bt_brk_mean"] = g.brk.mean()
    out["bt_spin_mean"] = g.spin_rate.mean()
    out["bt_relheight_mean"] = g.rel_height.mean()   # 투수 팔높이 노출 분포
    out["bt_ext_mean"] = g.extension.mean()
    out["bt_zonespeed_loss"] = (h.rel_speed - h.zone_speed).groupby(h[B]).mean()
    out["bt_n"] = np.log1p(g.size())
    sh = h.groupby([B, "tagged_pitch_type"]).size().rename("n").reset_index()
    sh["p"] = sh.n / sh.groupby(B).n.transform("sum")
    piv = sh.pivot(index=B, columns="tagged_pitch_type", values="p").fillna(0.0)
    for c in piv.columns:
        if (piv[c] > 0.02).mean() >= 0.15:
            out["bt_mix_" + c] = piv[c]
    sh["e"] = -sh.p * np.log(sh.p + 1e-12)
    out["bt_mix_entropy"] = sh.groupby(B).e.sum()
    return out


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", usecols=MAIN)
    tm = pd.read_csv(HERE / "data" / "trackman_history.csv", usecols=TMC)
    mg, ts, matches = match_exact_games(df, tm)
    del tm; gc.collect()
    print("경기 매칭 %d개" % len(matches), flush=True)

    o1 = pd.read_csv(HERE / "results" / "exp30_tuned_league_role_oof.csv.gz")
    o2 = pd.read_csv(HERE / "results" / "exp55_hand_seedavg_oof.csv.gz")
    c1 = [c for c in o1.columns if c in ("prediction", "pred", "y_pred")][0]
    c2 = [c for c in o2.columns if c in ("prediction", "pred", "y_pred")][0]
    o = (o1[["row_id", "season", TARGET, c1]].rename(columns={c1: "pg"})
         .merge(o2[["row_id", c2]].rename(columns={c2: "pn"}), on="row_id"))
    o["resid"] = o[TARGET] - (0.75 * o.pg + 0.25 * o.pn)
    d = df.merge(o[["row_id", "resid"]], on="row_id")

    for vy in (2022, 2024):
        mp = build_batter_mapping(mg, ts, matches, vy)
        prof = batter_profile(ts[ts.season < vy])
        cur = d[d.season == vy].copy()
        # 2025 유형 행만 채점 (채점 기준 2차 개정)
        cur = cur[(cur.game_type == "R") | (cur.season >= 2023)]
        b2t = dict(zip(mp.batter_id, mp[B]))
        cur["tmid"] = cur.batter_id.map(b2t)
        cov = cur.tmid.notna().mean()
        print("\n=== %d fold | 타자 대응 %d명, 행 적용률 %.3f, 평가행 %d ==="
              % (vy, len(mp), cov, len(cur)), flush=True)
        if len(mp) < 50:
            print("  매핑 실패 — 타자 축은 여기서 종료")
            continue
        r = cur.resid.to_numpy()
        lown = cur.asof_batter_n.fillna(0).to_numpy()
        q33 = np.nanquantile(lown, 0.33)
        print("%-26s %8s %9s %9s %5s" % ("피처", "적용률", "상관", "저표본", "단조"))
        print("-" * 62)
        rows = []
        for c in prof.columns:
            v = cur.tmid.map(prof[c]).to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() < 5000:
                continue
            cc = np.corrcoef(v[ok], r[ok])[0, 1]
            lo = ok & (lown <= q33)
            cl = np.corrcoef(v[lo], r[lo])[0, 1] if lo.sum() > 3000 else np.nan
            q = pd.qcut(pd.Series(v[ok]), 5, labels=False, duplicates="drop")
            rm = pd.DataFrame({"r": r[ok], "q": q}).groupby("q").r.mean().to_numpy()
            mono = "Y" if (np.all(np.diff(rm) > -0.0012) or np.all(np.diff(rm) < 0.0012)) else "-"
            rows.append((c, abs(cc)))
            print("%-26s %8.3f %+9.5f %+9.5f %5s" % (c, ok.mean(), cc, cl, mono))
        if rows:
            best = max(rows, key=lambda x: x[1])
            print("  -> 최대 |상관| %.5f (%s)  [문턱 0.0105 통과 / 0.0073 미달]"
                  % (best[1], best[0]))


if __name__ == "__main__":
    main()

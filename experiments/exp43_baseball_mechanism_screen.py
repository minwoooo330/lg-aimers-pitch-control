# -*- coding: utf-8 -*-
"""실험 43: 야구 메커니즘 기반 Trackman 피처 스크리닝.

지금까지 시험한 Trackman 피처는 전부 '투수 단위 물리 요약'이었다. 여기서는
메인 데이터가 구조적으로 표현할 수 없는 야구 메커니즘 네 가지를 검증한다.

A 구종군 내부 세분 구성
  메인은 fastball/breaking/offspeed 3군 비율만 준다. 그러나 breaking 안의
  커브와 슬라이더는 제구 난이도가 전혀 다르다. Trackman의 tagged_pitch_type
  으로만 구분할 수 있다.

B 팔각도(arm slot) x 타자 손잡이
  릴리스 높이가 낮은 사이드암은 반대 손 타자를 상대할 때 제구가 무너진다.
  메인에는 pitcher_hand/batter_hand만 있고 팔각도가 없어 표현이 불가능하다.

C 터널링/티핑 - 구종 간 릴리스 분산
  구종마다 릴리스 포인트가 흩어지면 타자에게 읽히고(티핑) 제구도 불안정하다.
  구종 '내' 산포가 아니라 구종 '간' 산포라는 점이 기존 피처와 다르다.

D 카운트별 구속 조절 폭
  3볼에서 구속을 낮춰 스트라이크를 확보하는 정도 = 제구 우선 성향.
  투수의 전략 성향이며 카운트에 따라 행마다 값이 달라진다.

평가: 현재 최고 모델에 가까운 앙상블 잔차와의 상관으로 스크리닝한다.
(0.85 x exp30_tuned_league_role + 0.15 x exp37_nn_seedavg)
"""
from pathlib import Path
import sys, gc
import numpy as np
import pandas as pd
from trackman_features import match_exact_games, build_pitcher_mapping

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
P = "pitcher_trackman_id"
K = 200.0

MAIN = ["row_id", "season", "inning", "top_bottom", "game_type", "balls_before",
        "strikes_before", "outs_before", "pitcher_id", "pitcher_hand", "batter_hand",
        "asof_pitcher_n", "control_success"]
TMC = ["trackman_id", "season", "trackman_game_id", "pitch_no", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before", "pitcher_trackman_id",
       "pitcher_hand", "batter_hand", "tagged_pitch_type", "pitch_type_group",
       "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
       "rel_height", "rel_side", "zone_speed"]


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", usecols=MAIN)
    tm = pd.read_csv(HERE / "data" / "trackman_history.csv", usecols=TMC)
    mg, ts, matches = match_exact_games(df, tm)
    del tm; gc.collect()
    print("경기 매칭 %d개" % len(matches), flush=True)

    # ---------- 행 단위 정렬: 매칭 경기에서 구종을 메인 행에 붙인다 ----------
    mi = mg.groupby("_game_idx", sort=False).indices
    ti = ts.groupby("trackman_game_id", sort=False).indices
    parts = []
    for row in matches.itertuples(index=False):
        a, b = mi[row.main_game_idx], ti[row.trackman_game_id]
        if len(a) != len(b):
            continue
        left = mg.iloc[a][["row_id", "season", "balls_before", "strikes_before",
                           "pitcher_hand", "batter_hand", "control_success"]].reset_index(drop=True)
        right = ts.iloc[b][["tagged_pitch_type", "pitch_type_group", "rel_speed",
                            "induced_vert_break", "horz_break", "pitcher_hand",
                            "batter_hand"]].reset_index(drop=True)
        right.columns = ["tagged", "grp", "rel_speed", "ivb", "hb", "tm_ph", "tm_bh"]
        parts.append(pd.concat([left, right], axis=1))
    al = pd.concat(parts, ignore_index=True)
    print("행 단위 정렬 %d행" % len(al), flush=True)

    # 손잡이 코드 대응 확인 (메인은 숫자 코드, Trackman은 문자열)
    print("\n=== 손잡이 코드 대응 확인 ===")
    print(pd.crosstab(al.pitcher_hand, al.tm_ph).to_string())

    # ---------- A. 구종별 실제 제구 성공률 ----------
    print("\n=== [A] 구종별 실제 제구 성공률 (세분) ===")
    t = al.groupby("tagged").agg(n=("control_success", "size"),
                                 성공률=("control_success", "mean"),
                                 구속=("rel_speed", "mean"),
                                 무브먼트=("ivb", lambda s: np.hypot(s, al.loc[s.index, "hb"]).mean()))
    t = t[t.n >= 2000].sort_values("성공률")
    print(t.round(4).to_string())
    print("\n=== 구종군별 (비교) ===")
    print(al.groupby("grp").agg(n=("control_success", "size"),
                                성공률=("control_success", "mean")).round(4).to_string())

    # ---------- 잔차 준비 ----------
    o1 = pd.read_csv(HERE / "results" / "exp30_tuned_league_role_oof.csv.gz")
    o2 = pd.read_csv(HERE / "results" / "exp37_nn_seedavg_oof.csv.gz")
    c1 = [c for c in o1.columns if c in ("prediction", "pred", "y_pred")][0]
    c2 = [c for c in o2.columns if c in ("prediction", "pred", "y_pred")][0]
    a = o1[["row_id", "season", "control_success", c1]].rename(columns={c1: "p_gbdt"})
    b = o2[["row_id", c2]].rename(columns={c2: "p_nn"})
    o = a.merge(b, on="row_id", how="inner")
    o = o[o.season == 2024]
    o["ens"] = 0.85 * o["p_gbdt"] + 0.15 * o["p_nn"]
    o["resid"] = o.control_success - o["ens"]
    print("\n앙상블 근사 2024 Brier = %.6f" % (o.resid ** 2).mean())

    cur = df[df.season == 2024].merge(o[["row_id", "resid"]], on="row_id", how="inner")
    mapping = build_pitcher_mapping(mg, ts, matches, 2024)
    m2t = dict(zip(mapping.pitcher_id, mapping.pitcher_trackman_id))
    cur["tmid"] = cur.pitcher_id.map(m2t)
    r = cur.resid.to_numpy()
    lown = cur.asof_pitcher_n.fillna(0).to_numpy()
    q33 = np.nanquantile(lown, 0.33)

    h = ts[ts.season < 2024].copy()
    sgn = h.pitcher_hand.map({"Right": 1.0, "Left": -1.0})
    h["rs"] = h.rel_side * sgn
    h["brk"] = np.hypot(h.induced_vert_break, h.horz_break)

    def rep(name, v):
        v = np.asarray(v, float); ok = np.isfinite(v)
        if ok.sum() < 5000:
            print("%-38s 표본부족" % name); return
        c = np.corrcoef(v[ok], r[ok])[0, 1]
        lo = ok & (lown <= q33)
        cl = np.corrcoef(v[lo], r[lo])[0, 1] if lo.sum() > 3000 else np.nan
        q = pd.qcut(pd.Series(v[ok]), 5, labels=False, duplicates="drop")
        rm = pd.DataFrame({"r": r[ok], "q": q}).groupby("q").r.mean().to_numpy()
        mono = "Y" if (np.all(np.diff(rm) > -0.0012) or np.all(np.diff(rm) < 0.0012)) else "-"
        print("%-38s %6.3f %+9.5f %+9.5f  %3s  %.6f" % (name, ok.mean(), c, cl, mono, c * c * 0.248))

    print("\n%-38s %6s %9s %9s %4s  추정이득" % ("피처", "적용률", "전체상관", "저표본", "단조"))
    print("-" * 88)

    # ---------- A 피처: 구종군 내부 구성 ----------
    sh = h.groupby([P, "tagged_pitch_type"]).size().rename("n").reset_index()
    sh["p"] = sh.n / sh.groupby(P).n.transform("sum")
    piv = sh.pivot(index=P, columns="tagged_pitch_type", values="p").fillna(0.0)
    print("[A] 구종군 내부 세분 구성")
    for col in piv.columns:
        if (piv[col] > 0.02).mean() >= 0.15:
            rep("  비중_%s" % col, cur.tmid.map(piv[col]).to_numpy())

    # ---------- B 피처: 팔각도 x 타자 손잡이 ----------
    print("[B] 팔각도 x 타자 손잡이")
    fb = h[h.pitch_type_group == "fastball"]
    slot_h = fb.groupby(P).rel_height.mean()
    slot_s = fb.groupby(P).rs.mean()
    rep("  릴리스높이(낮을수록 사이드암)", cur.tmid.map(slot_h).to_numpy())
    rep("  릴리스좌우(클수록 옆구리)", cur.tmid.map(slot_s).to_numpy())
    ph = cur.pitcher_hand.to_numpy(); bh = cur.batter_hand.to_numpy()
    opp = (ph != bh).astype(float)
    sh_v = cur.tmid.map(slot_h).to_numpy()
    rep("  릴리스높이 x 반대손타자", sh_v * opp)
    rep("  릴리스높이 x 같은손타자", sh_v * (1 - opp))
    ss_v = cur.tmid.map(slot_s).to_numpy()
    rep("  릴리스좌우 x 반대손타자", ss_v * opp)

    # ---------- C 피처: 터널링 / 티핑 ----------
    print("[C] 터널링/티핑 (구종 간 릴리스 분산)")
    gt = h.groupby([P, "pitch_type_group"]).agg(rs=("rs", "mean"), rh=("rel_height", "mean"),
                                                n=("rs", "size")).reset_index()
    gt = gt[gt.n >= 100]
    tun_s = gt.groupby(P).rs.std()
    tun_h = gt.groupby(P).rh.std()
    rep("  구종간_릴리스좌우_분산", cur.tmid.map(tun_s).to_numpy())
    rep("  구종간_릴리스높이_분산", cur.tmid.map(tun_h).to_numpy())
    rep("  구종간_릴리스_합성분산", np.hypot(cur.tmid.map(tun_s).to_numpy(),
                                       cur.tmid.map(tun_h).to_numpy()))
    gt2 = h.groupby([P, "tagged_pitch_type"]).agg(rs=("rs", "mean"), rh=("rel_height", "mean"),
                                                  n=("rs", "size")).reset_index()
    gt2 = gt2[gt2.n >= 100]
    rep("  세분구종간_릴리스좌우_분산", cur.tmid.map(gt2.groupby(P).rs.std()).to_numpy())

    # ---------- D 피처: 카운트별 구속 조절 ----------
    print("[D] 카운트별 구속 조절 성향")
    vb = h.groupby(P).rel_speed.mean()
    three = h[h.balls_before == 3].groupby(P).rel_speed.agg(["mean", "size"])
    thr = ((three["mean"] * three["size"] + K * three.index.map(vb)) /
           (three["size"] + K)) - three.index.map(vb)
    rep("  3볼_구속조절폭(정적)", cur.tmid.map(thr).to_numpy())
    is3 = (cur.balls_before == 3).to_numpy().astype(float)
    rep("  3볼_구속조절폭 x 현재3볼", cur.tmid.map(thr).to_numpy() * is3)
    two = h[h.strikes_before == 2].groupby(P).rel_speed.agg(["mean", "size"])
    thr2 = ((two["mean"] * two["size"] + K * two.index.map(vb)) /
            (two["size"] + K)) - two.index.map(vb)
    is2 = (cur.strikes_before == 2).to_numpy().astype(float)
    rep("  2스트라이크_구속변화 x 현재2S", cur.tmid.map(thr2).to_numpy() * is2)


if __name__ == "__main__":
    main()

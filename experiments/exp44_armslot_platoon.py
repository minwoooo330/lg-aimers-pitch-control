# -*- coding: utf-8 -*-
"""실험 44: 팔각도 x 좌우 매치업 합성 피처와 레퍼토리 난이도 지수.

exp43 스크리닝 결과:
  릴리스높이 단독            +0.00056  (신호 없음)
  릴리스높이 x 반대손타자      +0.01056  단조
  릴리스높이 x 같은손타자      -0.01049  단조
단독으로는 0인데 매치업으로 쪼개면 부호가 뒤집힌다. 사이드암이 같은 손 타자에
강하고 반대 손 타자에 약하다는 고전적 platoon 메커니즘이며, 메인 데이터에는
팔각도가 없어 트리가 발견할 수 없다. 여기서는 하나의 부호 피처로 합성한다.

또한 exp43에서 구종 세분 제구 난이도를 측정했다(커브 0.4500 ~ 커터 0.5473).
운영진이 주는 3군 분류는 breaking 내부 스프레드 0.097을 파괴한다. 이를
레퍼토리 난이도 지수(RDI)로 압축해 함께 검증한다.

기각된 가설(재시도 불필요): 터널링/티핑(구종 간 릴리스 분산) 전부 ~0,
카운트별 구속 조절 폭도 실제 카운트에 걸면 ~0.
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
MAIN = ["row_id", "season", "game_type", "inning", "top_bottom", "balls_before",
        "strikes_before", "outs_before", "pitcher_id", "pitcher_hand", "batter_hand",
        "asof_pitcher_n", "control_success"]
TMC = ["season", "trackman_game_id", "pitch_no", "inning", "top_bottom", "balls_before",
       "strikes_before", "outs_before", "pitcher_trackman_id", "pitcher_hand",
       "batter_hand", "tagged_pitch_type", "pitch_type_group", "rel_speed",
       "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
# exp43에서 행 단위 정렬로 측정한 구종별 리그 제구 성공률
CMD = {"Curveball": 0.4500, "Splitter": 0.4966, "Slider": 0.5019, "ChangeUp": 0.5191,
       "Fastball": 0.5377, "Sinker": 0.5383, "Cutter": 0.5473}


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", usecols=MAIN)
    tm = pd.read_csv(HERE / "data" / "trackman_history.csv", usecols=TMC)
    mg, ts, matches = match_exact_games(df, tm)
    del tm; gc.collect()

    o1 = pd.read_csv(HERE / "results" / "exp30_tuned_league_role_oof.csv.gz")
    o2 = pd.read_csv(HERE / "results" / "exp37_nn_seedavg_oof.csv.gz")
    c1 = [c for c in o1.columns if c in ("prediction", "pred", "y_pred")][0]
    c2 = [c for c in o2.columns if c in ("prediction", "pred", "y_pred")][0]
    o = (o1[["row_id", "season", "control_success", c1]].rename(columns={c1: "pg"})
         .merge(o2[["row_id", c2]].rename(columns={c2: "pn"}), on="row_id"))
    o = o[o.season == 2024]
    o["resid"] = o.control_success - (0.85 * o.pg + 0.15 * o.pn)

    cur = df[df.season == 2024].merge(o[["row_id", "resid"]], on="row_id")
    mapping = build_pitcher_mapping(mg, ts, matches, 2024)
    m2t = dict(zip(mapping.pitcher_id, mapping.pitcher_trackman_id))
    cur["tmid"] = cur.pitcher_id.map(m2t)
    r = cur.resid.to_numpy()
    lown = cur.asof_pitcher_n.fillna(0).to_numpy()
    q33 = np.nanquantile(lown, 0.33)
    # 메인 손잡이 코드는 1=Left, 2=Right (exp43 교차표로 확인)
    opp_sign = np.where(cur.pitcher_hand.to_numpy() != cur.batter_hand.to_numpy(), 1.0, -1.0)

    h = ts[ts.season < 2024].copy()
    sgn = h.pitcher_hand.map({"Right": 1.0, "Left": -1.0})
    h["rs"] = h.rel_side * sgn
    fb = h[h.pitch_type_group == "fastball"]

    def z(s):
        return (s - s.mean()) / s.std()

    slot_h = z(fb.groupby(P).rel_height.mean())
    slot_s = z(fb.groupby(P).rs.mean())
    slot_ext = z(fb.groupby(P).extension.mean())

    def rep(name, v):
        v = np.asarray(v, float); ok = np.isfinite(v)
        if ok.sum() < 5000:
            print("%-40s 표본부족" % name); return
        c = np.corrcoef(v[ok], r[ok])[0, 1]
        lo = ok & (lown <= q33)
        cl = np.corrcoef(v[lo], r[lo])[0, 1] if lo.sum() > 3000 else np.nan
        q = pd.qcut(pd.Series(v[ok]), 5, labels=False, duplicates="drop")
        rm = pd.DataFrame({"r": r[ok], "q": q}).groupby("q").r.mean().to_numpy()
        mono = "Y" if (np.all(np.diff(rm) > -0.0012) or np.all(np.diff(rm) < 0.0012)) else "-"
        print("%-40s %6.3f %+9.5f %+9.5f  %3s  %.6f" % (name, ok.mean(), c, cl, mono, c * c * 0.248))
        return v

    print("\n%-40s %6s %9s %9s %4s  추정이득" % ("피처", "적용률", "전체상관", "저표본", "단조"))
    print("-" * 90)
    print("[B] 팔각도 x 매치업 합성")
    v_h = cur.tmid.map(slot_h).to_numpy()
    v_s = cur.tmid.map(slot_s).to_numpy()
    v_e = cur.tmid.map(slot_ext).to_numpy()
    rep("  armslot_height x platoon(부호)", v_h * opp_sign)
    rep("  armslot_side x platoon(부호)", v_s * opp_sign)
    rep("  armslot_ext x platoon(부호)", v_e * opp_sign)
    rep("  armslot_합성(높이-|좌우|) x platoon", (v_h - np.abs(v_s)) * opp_sign)

    print("[A] 레퍼토리 난이도 지수 (RDI)")
    sh = h.groupby([P, "tagged_pitch_type"]).size().rename("n").reset_index()
    sh = sh[sh.tagged_pitch_type.isin(CMD)]
    sh["p"] = sh.n / sh.groupby(P).n.transform("sum")
    sh["w"] = sh.p * sh.tagged_pitch_type.map(CMD)
    rdi = sh.groupby(P).w.sum() / sh.groupby(P).p.sum()
    v_rdi = cur.tmid.map(rdi).to_numpy()
    rep("  RDI_세분7종(정적)", v_rdi)
    # 3군만으로 만든 대조군: 운영진 분류가 잃는 정보량을 보여준다
    grp_cmd = {"fastball": 0.5397, "breaking": 0.4812, "offspeed": 0.5108}
    sg = h[h.pitch_type_group.isin(grp_cmd)].groupby([P, "pitch_type_group"]).size().rename("n").reset_index()
    sg["p"] = sg.n / sg.groupby(P).n.transform("sum")
    sg["w"] = sg.p * sg.pitch_type_group.map(grp_cmd)
    rdi3 = sg.groupby(P).w.sum() / sg.groupby(P).p.sum()
    rep("  RDI_3군(대조군)", cur.tmid.map(rdi3).to_numpy())
    rep("  RDI_세분 - RDI_3군 (순수 증분)", v_rdi - cur.tmid.map(rdi3).to_numpy())

    # 상황 조건부 RDI: 카운트/타자손잡이별 구종 선택이 바뀐다
    hh = h[h.tagged_pitch_type.isin(CMD)].copy()
    hh["_adv"] = np.sign(hh.strikes_before - hh.balls_before).astype(int)
    hh["cmd"] = hh.tagged_pitch_type.map(CMD)
    cur["_adv"] = np.sign(cur.strikes_before - cur.balls_before).astype(int)
    bh_map = {1: "Left", 2: "Right"}
    cur["_bh"] = cur.batter_hand.map(bh_map)
    for nm, keys, curkeys in [("RDI x 카운트우열", [P, "_adv"], [cur.tmid, cur._adv]),
                              ("RDI x 타자손잡이", [P, "batter_hand"], [cur.tmid, cur._bh]),
                              ("RDI x 손잡이 x 카운트우열", [P, "batter_hand", "_adv"],
                               [cur.tmid, cur._bh, cur._adv])]:
        g = hh.groupby(keys, observed=True).cmd.agg(["mean", "size"])
        pm = g.index.get_level_values(0).map(rdi)
        dev = ((g["mean"] * g["size"] + K * pm) / (g["size"] + K)) - pm
        idx = pd.MultiIndex.from_arrays([a.to_numpy() for a in curkeys])
        rep("  %s (편차)" % nm, dev.reindex(idx).to_numpy())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""실험 49 본 스크리닝: 실패 유형별 메커니즘 피처 후보.

평가 = 현행 최고 앙상블 근사 잔차(2024)와의 상관. 채택 하한 0.035.
모든 표는 시즌<2024 만으로 만든다 (walk-forward, 누수 없음).
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
THR = 0.035

d = pd.read_pickle(HERE/"results"/"exp49_base.pkl")
d["rev"] = (d.cls==1).astype(np.float32)
d["mid"] = (d.cls==2).astype(np.float32)
d["thr"] = (d.cls==3).astype(np.float32)   # 3번째 실패(이름 없음)
d["suc"] = (d.cls==0).astype(np.float32)
d["cnt"] = (d.balls_before*3 + d.strikes_before).astype(np.int16)

past = d[d.season < 2024]
cur  = d[(d.season == 2024) & d.resid.notna()].copy()
r = cur.resid.to_numpy()
print("2024 평가행 %d, Brier=%.6f" % (len(cur), (r**2).mean()))
isR = (cur.game_type=="R").to_numpy()
lown = cur.asof_pitcher_n.fillna(0).to_numpy()
q33 = np.nanquantile(lown, 0.33)

RES = []
def rep(name, v, note=""):
    v = np.asarray(v, float); ok = np.isfinite(v)
    if ok.sum() < 20000:
        print("%-42s  표본부족(%d)" % (name, ok.sum())); return
    c  = np.corrcoef(v[ok], r[ok])[0,1]
    okR = ok & isR
    cR = np.corrcoef(v[okR], r[okR])[0,1]
    lo = ok & (lown <= q33)
    cl = np.corrcoef(v[lo], r[lo])[0,1] if lo.sum()>5000 else np.nan
    q = pd.qcut(pd.Series(v[ok]), 5, labels=False, duplicates="drop")
    rm = pd.DataFrame({"r":r[ok],"q":q}).groupby("q").r.mean().to_numpy()
    mono = "Y" if (np.all(np.diff(rm)>-0.0012) or np.all(np.diff(rm)<0.0012)) else "-"
    flag = "***" if abs(c)>=THR else ""
    print("%-42s %5.2f %+8.5f %+8.5f %+8.5f  %s %s %s"
          % (name, ok.mean(), c, cR, cl, mono, flag, note))
    RES.append(dict(feature=name, cover=ok.mean(), corr=c, corr_R=cR, corr_low=cl, mono=mono))

print("\n%-42s %5s %8s %8s %8s" % ("후보","적용률","전체상관","1군상관","저표본"))
print("-"*90)

# =====================================================================
# A. 3번째 실패(이름 없는 실패)의 직접 측정
# =====================================================================
print("\n[A] 3번째 실패 유형 - 공식 피처에 없는 축")
# A0 대조군: asof 파생 (기존 exp25에서 기각된 형태)
derived = (1 - cur.asof_pitcher_success_rate - cur.asof_pitcher_reverse_rate
             - cur.asof_pitcher_middle_rate).to_numpy()
rep("A0 파생 1-s-r-m (대조군)", derived)

# A1 과거시즌 직접 집계 (평활)
K = 200.0
gm = past.thr.mean()
t = past.groupby("pitcher_id").thr.agg(["sum","size"])
thr_tab = ((t["sum"] + K*gm) / (t["size"] + K)).rename("p_thr")
v_thr = cur.pitcher_id.map(thr_tab).to_numpy()
rep("A1 투수 3번째실패율 (과거시즌 직접)", v_thr)
rep("A1b  동 - 파생값 차이", v_thr - derived)

# 파생 vs 직접의 투수단위 일치도
gg = past.groupby("pitcher_id").agg(n=("thr","size"), a=("thr","mean"),
      b=("asof_pitcher_success_rate","mean"), c2=("asof_pitcher_reverse_rate","mean"),
      d2=("asof_pitcher_middle_rate","mean"))
gg = gg[gg.n>=1000]; gg["der"] = 1-gg.b-gg.c2-gg.d2
print("   [진단] 투수단위 직접측정 vs 파생 상관 = %+.4f (투수 %d명)"
      % (np.corrcoef(gg.a, gg.der)[0,1], len(gg)))

# =====================================================================
# B. 상황 노출 x 투수 유형 성향  (matching)
# =====================================================================
print("\n[B] 상황별 실패유형 노출 x 투수 유형 성향")
TYPES = ["rev","mid","thr"]
gbar = {t_: past[t_].mean() for t_ in TYPES}
# 투수 유형 성향 (행 단위 asof, 누수 없음) - thr은 과거표에서
prop = {
    "rev": cur.asof_pitcher_reverse_rate.to_numpy()/gbar["rev"],
    "mid": cur.asof_pitcher_middle_rate.to_numpy()/gbar["mid"],
    "thr": v_thr/gbar["thr"],
}
def exposure(keys, name):
    pi = past.groupby(keys)[TYPES].mean()
    idx = cur.set_index(keys).index
    pv = {t_: pd.Series(pi[t_]).reindex(idx).to_numpy() for t_ in TYPES}
    # 매칭 예측 실패확률
    pf = sum(pv[t_]*prop[t_] for t_ in TYPES)
    rep("B %s: 매칭 성공확률" % name, 1-pf)
    # 매칭 - 무매칭(투수 성향 무시) 차이 = 순수 상호작용 항
    pf0 = sum(pv[t_] for t_ in TYPES)
    rep("B %s: 매칭 증분(상호작용만)" % name, pf0-pf)
    return pv

pv_cnt = exposure(["cnt"], "카운트")
cur["inn_b"] = np.clip(cur.inning,1,9)
past2 = past.copy(); past2["inn_b"] = np.clip(past2.inning,1,9)
exposure(["cnt","num_runners_on"], "카운트x주자수")
_p = past.copy(); _p["hm"] = (_p.pitcher_hand==_p.batter_hand).astype(int)
cur["hm"] = (cur.pitcher_hand==cur.batter_hand).astype(int)
pi = _p.groupby(["cnt","hm"])[TYPES].mean()
idx = cur.set_index(["cnt","hm"]).index
pv = {t_: pd.Series(pi[t_]).reindex(idx).to_numpy() for t_ in TYPES}
pf = sum(pv[t_]*prop[t_] for t_ in TYPES); pf0 = sum(pv[t_] for t_ in TYPES)
rep("B 카운트x동손: 매칭 성공확률", 1-pf)
rep("B 카운트x동손: 매칭 증분", pf0-pf)

# 개별 유형 매칭항 (어느 유형이 신호원인지)
for t_ in TYPES:
    rep("B  카운트 노출x성향 [%s]" % t_, pv_cnt[t_]*prop[t_])

# =====================================================================
# C. 유형별 최근 폼 (career 대비)
# =====================================================================
print("\n[C] 유형별 최근 폼")
rep("C 한가운데 prev1-career", (cur.asof_pitcher_prev1_game_middle_rate
                              - cur.asof_pitcher_middle_rate).to_numpy())
rep("C 한가운데 prev3-career", (cur.asof_pitcher_prev3_game_middle_rate
                              - cur.asof_pitcher_middle_rate).to_numpy())
rep("C 한가운데 prev5-career", (cur.asof_pitcher_prev5_game_middle_rate
                              - cur.asof_pitcher_middle_rate).to_numpy())
# 성공 폼에서 한가운데 폼을 뺀 것 = 비가운데 실패 폼
rep("C 비가운데 실패 폼 prev3", (1-cur.asof_pitcher_prev3_game_success_rate
                              -cur.asof_pitcher_prev3_game_middle_rate).to_numpy())

# =====================================================================
# D. 타자 측 유형 노출 (과거시즌 OOF 표)
# =====================================================================
print("\n[D] 타자 측 실패유형 노출 (과거시즌 표)")
Kb = 300.0
for t_ in TYPES+["suc"]:
    m = past[t_].mean()
    tb = past.groupby("batter_id")[t_].agg(["sum","size"])
    tab = ((tb["sum"]+Kb*m)/(tb["size"]+Kb))
    rep("D 타자 %s 유발률" % t_, cur.batter_id.map(tab).to_numpy())

pd.DataFrame(RES).to_csv(HERE/"results"/"exp49_screen.csv", index=False)
print("\n저장: results/exp49_screen.csv")

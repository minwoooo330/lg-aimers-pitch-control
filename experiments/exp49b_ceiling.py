# -*- coding: utf-8 -*-
"""실험 49b: 추가 후보 + 오라클 상한(ceiling) 측정.

핵심 질문: 잔차 상관 0.035는 애초에 도달 가능한가?
정답을 훔쳐본 오라클 그룹평균의 상관을 재면 어떤 축의 물리적 상한이 나온다.
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)

d = pd.read_pickle(HERE/"results"/"exp49_base.pkl")
d["rev"]=(d.cls==1).astype(np.float32); d["mid"]=(d.cls==2).astype(np.float32)
d["thr"]=(d.cls==3).astype(np.float32); d["suc"]=(d.cls==0).astype(np.float32)
d["cnt"]=(d.balls_before*3+d.strikes_before).astype(np.int16)
past = d[d.season<2024]
cur  = d[(d.season==2024)&d.resid.notna()].copy()
r = cur.resid.to_numpy()
lown = cur.asof_pitcher_n.fillna(0).to_numpy(); q33=np.nanquantile(lown,0.33)
THR=0.035

def rep(name,v,note=""):
    v=np.asarray(v,float); ok=np.isfinite(v)
    if ok.sum()<20000: print("%-44s 표본부족"%name); return np.nan
    c=np.corrcoef(v[ok],r[ok])[0,1]
    lo=ok&(lown<=q33)
    cl=np.corrcoef(v[lo],r[lo])[0,1] if lo.sum()>5000 else np.nan
    print("%-44s %5.2f %+8.5f %+8.5f  %s %s"%(name,ok.mean(),c,cl,
          "***" if abs(c)>=THR else "   ", note))
    return c

print("%-44s %5s %8s %8s"%("후보","적용률","전체상관","저표본")); print("-"*80)

# ---------- E. 중기 지평(직전 시즌) - 커리어와 최근5경기 사이의 공백 ----------
print("\n[E] 중기 지평: 직전 시즌 유형별 성적 (커리어=stale, prev5경기=noisy 사이)")
K=150.0
prev_season = past[past.season==2023]
for t_,col in [("suc","asof_pitcher_success_rate"),("mid","asof_pitcher_middle_rate"),
               ("rev","asof_pitcher_reverse_rate")]:
    m=prev_season[t_].mean()
    tb=prev_season.groupby("pitcher_id")[t_].agg(["sum","size"])
    tab=(tb["sum"]+K*m)/(tb["size"]+K)
    v=cur.pitcher_id.map(tab).to_numpy()
    rep("E 직전시즌 %s율"%t_, v)
    rep("E 직전시즌 %s - 커리어asof"%t_, v-cur[col].to_numpy())

# ---------- F. 투수 카운트믹스 보정 ----------
print("\n[F] 투수 카운트 노출 편향 보정")
lg = past.groupby("cnt").suc.mean()
past_ = past.assign(exp_s=past.cnt.map(lg))
pm = past_.groupby("pitcher_id").agg(n=("suc","size"), act=("suc","mean"), exp=("exp_s","mean"))
pm = pm[pm.n>=200]; pm["adj"]=pm.act-pm.exp
rep("F 카운트믹스 보정 투수실력", cur.pitcher_id.map(pm.adj).to_numpy())
rep("F 투수 평균 카운트난이도", cur.pitcher_id.map(pm.exp).to_numpy())

# ---------- G. 오라클 상한 ----------
print("\n\n===== 오라클 상한 (2024 정답을 훔쳐본 그룹평균) =====")
print("어떤 피처도 이 값을 넘을 수 없다. honest = 2024를 반으로 나눠 교차")
half = rng.random(len(cur))<0.5
def oracle(keys, name, minn=30):
    g = cur.assign(_h=half)
    a = g[g._h].groupby(keys).resid.agg(["mean","size"])
    b = g[~g._h].groupby(keys).resid.agg(["mean","size"])
    # in-sample 상한
    full = g.groupby(keys).resid.transform("mean").to_numpy()
    c_in = np.corrcoef(full, r)[0,1]
    # honest: 반대편 절반의 평균을 예측으로
    pred = np.full(len(g), np.nan)
    ia = g.set_index(keys).index
    ma = a["mean"].where(a["size"]>=minn); mb = b["mean"].where(b["size"]>=minn)
    pa = pd.Series(mb).reindex(ia).to_numpy()   # h=True 행 -> b표
    pb = pd.Series(ma).reindex(ia).to_numpy()
    pred = np.where(half, pa, pb)
    ok=np.isfinite(pred)
    c_h = np.corrcoef(pred[ok], r[ok])[0,1]
    print("%-40s in-sample=%+.5f  honest=%+.5f (cov %.2f)"%(name,c_in,c_h,ok.mean()))

oracle(["pitcher_id"], "투수 단위 (모든 정적 투수피처 상한)")
oracle(["batter_id"], "타자 단위")
oracle(["cnt"], "카운트 12셀")
oracle(["cnt","num_runners_on","outs_before"], "카운트x주자x아웃")
cur["hm"]=(cur.pitcher_hand==cur.batter_hand).astype(int)
oracle(["pitcher_id","hm"], "투수x동손")
oracle(["pitcher_id","cnt"], "투수x카운트")
oracle(["pitcher_id","game_month"], "투수x월 (컨디션 시변)")
oracle(["pitcher_id","batter_id"], "투수x타자 매치업", minn=8)

print("\n[참고] 2024 잔차 표준편차 = %.5f" % r.std())
print("[참고] 상관 c의 Brier 이득 ~= c^2*0.248 ; 0.035 -> %.6f (약 %.0f점)"
      % (0.035**2*0.248, 0.035**2*0.248*400641))

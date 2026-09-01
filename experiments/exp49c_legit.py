# -*- coding: utf-8 -*-
"""실험 49c: 오라클 상한 vs 실제 구축 가능한 표의 격차.
오라클은 2024 자기 시즌 라벨을 쓴다(누수). 합법 피처는 과거 시즌만 쓴다.
그 격차가 곧 '축은 존재하나 이전 불가능한 신호'의 양이다.
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent
d=pd.read_pickle(HERE/"results"/"exp49_base.pkl")
d["suc"]=(d.cls==0).astype(np.float32)
d["cnt"]=(d.balls_before*3+d.strikes_before).astype(np.int16)
d["hm"]=(d.pitcher_hand==d.batter_hand).astype(np.int8)
d["adv"]=np.select([d.strikes_before>d.balls_before, d.balls_before>d.strikes_before],[0,2],1)
past=d[d.season<2024]; cur=d[(d.season==2024)&d.resid.notna()].copy()
r=cur.resid.to_numpy()

def legit(keys,name,K=200.0):
    gm=past.suc.mean()
    t=past.groupby(keys).suc.agg(["sum","size"])
    tab=(t["sum"]+K*gm)/(t["size"]+K)
    # 투수 주변값 대비 편차만 (exp40 채택 방식)
    pt=past.groupby("pitcher_id").suc.agg(["sum","size"])
    ptab=(pt["sum"]+K*gm)/(pt["size"]+K)
    idx=cur.set_index(keys).index
    v=pd.Series(tab).reindex(idx).to_numpy()
    vm=cur.pitcher_id.map(ptab).to_numpy()
    for lbl,val in [("(절대)",v),("(투수주변값 대비 편차)",v-vm)]:
        val=np.asarray(val,float); ok=np.isfinite(val)
        c=np.corrcoef(val[ok],r[ok])[0,1] if ok.sum()>20000 else np.nan
        print("%-46s %5.2f %+8.5f %s"%(name+lbl,ok.mean(),c,"***" if abs(c)>=0.035 else ""))

print("%-46s %5s %8s"%("과거시즌만으로 만든 합법 표","적용률","잔차상관")); print("-"*70)
legit(["pitcher_id"],"투수 단위")
legit(["pitcher_id","hm"],"투수x동손")
legit(["pitcher_id","hm","adv"],"투수x동손x카운트우열")
legit(["pitcher_id","cnt"],"투수x카운트")
legit(["batter_id"],"타자 단위")

print("\n\n===== 잔차 분산 분해 =====")
v0=r.var()
for keys,nm in [(["pitcher_id"],"투수"),(["batter_id"],"타자"),(["cnt"],"카운트"),
                (["pitcher_id","hm"],"투수x동손"),
                (["pitcher_id","cnt"],"투수x카운트"),
                (["game_month"],"월"),(["inning"],"이닝")]:
    g=cur.groupby(keys).resid.agg(["mean","size"])
    g=g[g["size"]>=25]
    # 그룹평균 분산에서 표본잡음 기대분산을 빼면 '진짜' 그룹간 분산
    obs=np.average(g["mean"]**2,weights=g["size"])
    noise=np.average(v0/g["size"],weights=g["size"])
    true=max(obs-noise,0.0)
    print("  %-14s 관측그룹분산=%.3e 잡음=%.3e 진짜=%.3e -> 최대상관 %.4f"
          %(nm,obs,noise,true,np.sqrt(true/v0)))
print("\n총 잔차분산 = %.5f" % v0)

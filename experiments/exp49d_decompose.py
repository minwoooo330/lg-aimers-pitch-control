# -*- coding: utf-8 -*-
"""실험 49d: 실패유형 분해가 집계 조건부표를 이기는가 + 결합 설명력."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent
d=pd.read_pickle(HERE/"results"/"exp49_base.pkl")
for k,v in [("suc",0),("rev",1),("mid",2),("thr",3)]: d[k]=(d.cls==v).astype(np.float32)
d["cnt"]=(d.balls_before*3+d.strikes_before).astype(np.int16)
d["hm"]=(d.pitcher_hand==d.batter_hand).astype(np.int8)
d["adv"]=np.select([d.strikes_before>d.balls_before,d.balls_before>d.strikes_before],[0,2],1)
past=d[d.season<2024]; cur=d[(d.season==2024)&d.resid.notna()].copy()
r=cur.resid.to_numpy(); K=200.0

def dev(keys,col):
    gm=past[col].mean()
    t=past.groupby(keys)[col].agg(["sum","size"]); tab=(t["sum"]+K*gm)/(t["size"]+K)
    p=past.groupby("pitcher_id")[col].agg(["sum","size"]); pt=(p["sum"]+K*gm)/(p["size"]+K)
    v=pd.Series(tab).reindex(cur.set_index(keys).index).to_numpy()
    return v-cur.pitcher_id.map(pt).to_numpy()

def c_of(v):
    v=np.asarray(v,float); ok=np.isfinite(v)
    return np.corrcoef(v[ok],r[ok])[0,1], ok

print("=== 조건부표: 집계 vs 실패유형 분해 (투수 주변값 대비 편차) ===")
print("%-40s %+9s"%("피처","잔차상관"))
KEYS=[(["pitcher_id","hm"],"투수x동손"),(["pitcher_id","hm","adv"],"투수x동손x카운트우열")]
store={}
for keys,nm in KEYS:
    agg=dev(keys,"suc"); c,_=c_of(agg); print("%-40s %+9.5f"%(nm+" 집계성공률",c))
    store[nm+"_agg"]=agg
    recomb=np.zeros(len(cur))
    for t_ in ["rev","mid","thr"]:
        v=dev(keys,t_); c2,_=c_of(v); print("   %-37s %+9.5f"%(nm+" ["+t_+"]",c2))
        store[nm+"_"+t_]=v; recomb=recomb-v
    c3,_=c_of(recomb); print("%-40s %+9.5f"%(nm+" 유형합성(-rev-mid-thr)",c3))

print("\n\n=== 결합 설명력 (교차적합 선형결합, 낙관편향 없음) ===")
names=list(store); X=np.column_stack([store[n] for n in names])
ok=np.isfinite(X).all(1)&np.isfinite(r); Xo=X[ok]; ro=r[ok]
rng=np.random.default_rng(1); h=rng.random(ok.sum())<0.5
pred=np.empty(ok.sum())
for tr,te in [(h,~h),(~h,h)]:
    A=np.column_stack([np.ones(tr.sum()),Xo[tr]])
    b=np.linalg.lstsq(A,ro[tr],rcond=None)[0]
    pred[te]=np.column_stack([np.ones(te.sum()),Xo[te]])@b
c=np.corrcoef(pred,ro)[0,1]
print("후보 %d개 전부 결합 -> 잔차상관 %+.5f (%d행)"%(len(names),c,ok.sum()))
print("추정 Brier 이득 = %.7f  (약 %.1f점)"%(c*c*0.248, c*c*0.248*400641))
print("\n채택 하한 0.035 대비: %s"%("통과" if abs(c)>=0.035 else "미달"))

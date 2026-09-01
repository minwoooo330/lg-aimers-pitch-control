
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
nn=0.5*((3*a3+5*a5)/8)+0.5*b8
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",usecols=["row_id","game_type","season"])
d=b.merge(tr,on="row_id",how="left",suffixes=("","_x"))
rel=((d.game_type=="R")|(d.season>=2023)).to_numpy()
y=d[TARGET].to_numpy(); season=d.season.to_numpy(); v7=d.v7.to_numpy()
EPS=1e-6
def lo(p): p=np.clip(p,EPS,1-EPS); return np.log(p/(1-p))
def sg(z): return 1/(1+np.exp(-z))
def ev(p,yr):
    m=(season==yr)&rel&np.isfinite(p)
    z=p[m]+(y[m].mean()-p[m].mean()); return float(np.mean((z-y[m])**2))
def sc_(p,s):
    l=lo(p); c=np.nanmean(l[np.isfinite(l)]); return sg(s*(l-c)+c)
def rep(nm,p):
    print(f"  {nm:38s} 2024 {ev(p,2024):.8f}   2022 {ev(p,2022):.8f}")
print("=== 혼합 형태 비교 (각 최적 scale 적용) ===")
def best_scale(p):
    bb=None
    for s in np.arange(1.00,1.26,0.01):
        q=sc_(p,s); v=ev(q,2024)+ev(q,2022)
        if bb is None or v<bb[1]: bb=(s,v,q)
    return bb
# 1) 현행: 확률공간 혼합 -> 샤프닝
p_cur=0.55*v7+0.45*nn
s1,_,q1=best_scale(p_cur); rep(f"확률혼합 55/45 -> scale {s1:.2f}",q1)
# 2) 로그오즈 공간 혼합
p_lg=sg(0.55*lo(v7)+0.45*lo(nn))
s2,_,q2=best_scale(p_lg); rep(f"로그오즈혼합 55/45 -> scale {s2:.2f}",q2)
# 3) 성분별 개별 샤프닝 후 혼합
bv=None
for sv in np.arange(1.00,1.31,0.02):
    for sn in np.arange(1.00,1.31,0.02):
        q=0.55*sc_(v7,sv)+0.45*sc_(nn,sn)
        v=ev(q,2024)+ev(q,2022)
        if bv is None or v<bv[0]: bv=(v,sv,sn,q)
rep(f"성분별 샤프닝(v7 {bv[1]:.2f}/nn {bv[2]:.2f}) -> 혼합",bv[3])
# 4) 혼합가중치 + scale 동시 최적화 (확률공간)
bw=None
for w in np.arange(0.30,0.76,0.05):
    p=w*v7+(1-w)*nn
    s,val,q=best_scale(p)
    if bw is None or val<bw[0]: bw=(val,w,s,q)
rep(f"가중치 {bw[1]:.2f} + scale {bw[2]:.2f} 동시최적",bw[3])
print(f"\n  [기준] 현행 제출본(확률혼합 55/45, scale 1.08) 2024 {ev(sc_(p_cur,1.08),2024):.8f}  2022 {ev(sc_(p_cur,1.08),2022):.8f}")

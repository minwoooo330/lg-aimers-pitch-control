
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    p=RES/fn
    if not p.exists(): return None
    d=pd.read_csv(p).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
nn=0.5*((3*a3+5*a5)/8)+0.5*b8
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",usecols=["row_id","game_type","season"])
d=b.merge(tr,on="row_id",how="left",suffixes=("","_x"))
rel=((d.game_type=="R")|(d.season>=2023)).to_numpy()
y=d[TARGET].to_numpy(); season=d.season.to_numpy(); v7=d.v7.to_numpy()
EPS=1e-6
def lo(p): p=np.clip(p,EPS,1-EPS); return np.log(p/(1-p))
def sg(z): return 1/(1+np.exp(-z))
def sc_(p,s):
    l=lo(p); c=np.nanmean(l[np.isfinite(l)]); return sg(s*(l-c)+c)
base=sc_(0.55*v7+0.45*nn,1.08)      # 현재 제출본과 동일 형태
def paired(p_new,yr):
    """짝지은 Brier 차이와 그 표준오차(e-5 단위). 양수=개선."""
    m=(season==yr)&rel&np.isfinite(p_new)&np.isfinite(base)
    yy=y[m]
    a=base[m]+(yy.mean()-base[m].mean()); c=p_new[m]+(yy.mean()-p_new[m].mean())
    di=(a-yy)**2-(c-yy)**2
    return di.mean()*1e5, di.std(ddof=1)/np.sqrt(len(di))*1e5, len(di)
cands=[]
v157=g("exp157_v7plus_oof.csv.gz")
if v157 is not None:
    for w in (0.25,0.50):
        cands.append((f"exp157 v7슬롯 {int(w*100)}%", sc_(0.55*((1-w)*v7+w*v157)+0.45*nn,1.08)))
v143=g("exp143_nn_sdall_oof.csv.gz")
if v143 is not None:
    for w in (0.05,0.10):
        p=0.55*v7+0.45*nn; mm=np.isfinite(v143); q=p.copy(); q[mm]=(1-w)*p[mm]+w*v143[mm]
        cands.append((f"exp143 전채널 추가 {int(w*100)}%", sc_(q,1.08)))
v129=None
for f in ("exp129_rank_oof_2024.csv.gz",):
    pass
cands.append(("scale 1.09", sc_(0.55*v7+0.45*nn,1.09)))
cands.append(("scale 1.10", sc_(0.55*v7+0.45*nn,1.10)))
cands.append(("가중치 v7 0.50", sc_(0.50*v7+0.50*nn,1.08)))
cands.append(("가중치 v7 0.60", sc_(0.60*v7+0.40*nn,1.08)))
print(f"{'후보':26s} {'2024 Δ(e-5)':>14s} {'SE':>7s} {'σ':>6s} | {'2022 Δ':>9s} {'SE':>7s} {'σ':>6s}")
print("-"*86)
for nm,p in cands:
    d24,s24,n24=paired(p,2024); d22,s22,n22=paired(p,2022)
    z24=d24/s24 if s24>0 else 0; z22=d22/s22 if s22>0 else 0
    flag=" <<<" if (d24>0 and d22>0 and min(z24,z22)>2) else ""
    print(f"{nm:26s} {d24:+14.4f} {s24:7.4f} {z24:+6.1f} | {d22:+9.4f} {s22:7.4f} {z22:+6.1f}{flag}")

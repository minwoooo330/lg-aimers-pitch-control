
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn,col="prediction"):
    d=pd.read_csv(RES/fn).set_index(ID)[col]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
b["nn"]=0.5*((3*a3+5*a5)/8)+0.5*b8
b["rmse"]=g("exp163_lowreg_bag_oof.csv.gz")
tr=pd.read_csv(HERE/"data"/"train.csv",usecols=[ID,"season","game_type",TARGET],encoding="utf-8-sig")
d=b.drop(columns=[c for c in ("season",TARGET) if c in b.columns]).merge(tr,on=ID,how="left")
d=d[(d.game_type=="R")|(d.season>=2023)]
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def sharp(p,c=-0.027330,k=1.08): return 1/(1+np.exp(-(c+k*(lg(p)-c))))
def sc(p,y): p=p+(y.mean()-p.mean()); return float(np.mean((y-p)**2))
print("=== exp161: v7 구조 + 저규제·배깅(l2=1, subsample=0.5, ES) — 다양성 멤버 판정 ===\n")
for yr in (2024,2022):
    s=d[d.season==yr]; y=s[TARGET].to_numpy()
    v7,nn,rm=s.v7.to_numpy(),s.nn.to_numpy(),s.rmse.to_numpy()
    base=0.55*v7+0.45*nn
    print(f"[{yr}]  v7(Logloss) {sc(v7,y):.8f}   저규제배깅 {sc(rm,y):.8f}   차 {(sc(v7,y)-sc(rm,y))*1e5:+.2f}e-5")
    er_v=(y-(v7+(y.mean()-v7.mean())));  er_r=(y-(rm+(y.mean()-rm.mean())))
    er_b=(y-(base+(y.mean()-base.mean())))
    print(f"      예측상관(v7,rmse) {np.corrcoef(v7,rm)[0,1]:.4f}   오차상관 {np.corrcoef(er_v,er_r)[0,1]:.6f}   베이스와 오차상관 {np.corrcoef(er_b,er_r)[0,1]:.6f}")
    b0=sc(sharp(base),y)
    row=[]
    for w in (0.05,0.10,0.15,0.20,0.30):
        q=sharp((1-w)*base+w*rm); row.append(f"{w:.0%} {(b0-sc(q,y))*1e5:+6.2f}")
    print("      추가 비중별 기여: "+" | ".join(row))
    # v7 슬롯 내부 교체 경로
    row2=[]
    for w in (0.2,0.35,0.5):
        q=sharp(0.55*((1-w)*v7+w*rm)+0.45*nn); row2.append(f"{w:.0%} {(b0-sc(q,y))*1e5:+6.2f}")
    print("      v7슬롯 내 혼합:   "+" | ".join(row2)+"\n")

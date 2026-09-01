
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
b["nn"]=0.5*((3*a3+5*a5)/8)+0.5*b8
tr=pd.read_csv(HERE/"data"/"train.csv",usecols=[ID,"season","game_type",TARGET],encoding="utf-8-sig")
d=b.drop(columns=[c for c in ("season",TARGET) if c in b.columns]).merge(tr,on=ID,how="left")
d=d[(d.game_type=="R")|(d.season>=2023)]
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def sg(z): return 1/(1+np.exp(-z))
def sc(p,y): p=p+(y.mean()-p.mean()); return float(np.mean((y-p)**2))
CEN=-0.027330
print("=== 샤프닝 scale: 로컬 OOF가 예측한 이득 vs 리더보드 실측 ===\n")
print("리더보드 실측:  scale 1.00 -> 1.08 에서 1116 -> 1127.02  (+11.02점 = 2.75e-5)")
print("               (단 shift도 -0.015397 -> -0.013911 로 함께 바뀜)\n")
for yr in (2024,2022):
    s=d[d.season==yr]; y=s[TARGET].to_numpy()
    p=0.55*s.v7.to_numpy()+0.45*s.nn.to_numpy()
    b100=sc(p,y)
    print(f"[{yr}] scale 1.00 기준 {b100:.8f}")
    for k in (1.04,1.08,1.12,1.16,1.20,1.25,1.30):
        q=sg(CEN+k*(lg(p)-CEN))
        print(f"      scale {k:.2f}  {(b100-sc(q,y))*1e5:+7.3f}e-5  ({(b100-sc(q,y))*1e5*4:+6.1f}점 상당)")
    print()


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
def ev(p,yr):
    m=(season==yr)&rel&np.isfinite(p)
    z=p[m]+(y[m].mean()-p[m].mean()); return float(np.mean((z-y[m])**2))
base=0.55*v7+0.45*nn; b24=ev(base,2024)
print(f"[베이스] 2024 {b24:.8f}   (2022는 exp156 미완이라 생략)\n")
for tag in ("rf","et"):
    v=g(f"exp156_{tag}_oof.csv.gz")
    if v is None: print(f"{tag}: 없음"); continue
    m=np.isfinite(v)&(season==2024)
    print(f"=== {tag.upper()} (2024 fold만) ===")
    print(f"  단독 {ev(v,2024):.8f}   상관(베이스) {np.corrcoef(base[m],v[m])[0,1]:.4f}   상관(nnsd16) {np.corrcoef(nn[m],v[m])[0,1]:.4f}")
    for w in (0.05,0.10,0.15,0.20,0.25):
        q=base.copy(); mm=np.isfinite(v); q[mm]=(1-w)*base[mm]+w*v[mm]
        print(f"    추가 {int(w*100):>2}%: 2024 {(b24-ev(q,2024))*1e5:+6.2f}")
    print()

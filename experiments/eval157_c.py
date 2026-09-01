
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
nn=0.5*((3*a3+5*a5)/8)+0.5*b8
v7p=g("exp157_v7plus_oof.csv.gz")
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",usecols=["row_id","game_type","season"])
d=b.merge(tr,on="row_id",how="left",suffixes=("","_x"))
rel=((d.game_type=="R")|(d.season>=2023)).to_numpy()
y=d[TARGET].to_numpy(); season=d.season.to_numpy(); v7=d.v7.to_numpy()
def ev(p,yr):
    m=(season==yr)&rel&np.isfinite(p)
    z=p[m]+(y[m].mean()-p[m].mean()); return float(np.mean((z-y[m])**2))
base=0.55*v7+0.45*nn
b24,b22=ev(base,2024),ev(base,2022)
print(f"[베이스] v7 55% + nnsd16 45%      2024 {b24:.8f}   2022 {b22:.8f}\n")
print("=== (C) v7 슬롯을 exp157로 교체/혼합 (nnsd16 45% 고정) ===")
for w in (0.0,0.25,0.5,0.75,1.0):
    v_=(1-w)*v7+w*v7p
    q=0.55*v_+0.45*nn
    d24=(b24-ev(q,2024))*1e5; d22=(b22-ev(q,2022))*1e5
    print(f"  exp157 비중 {int(w*100):>3}%:  2024 {d24:+6.2f}  2022 {d22:+6.2f}{'  <<<' if d24>0 and d22>0 else ''}")
print("\n=== (D) 3성분: v7 + exp157 + nnsd16 (가중치 그리드) ===")
best=None
for wv in (0.20,0.30,0.40,0.55):
    for wp in (0.10,0.15,0.25,0.35):
        wn=1-wv-wp
        if wn<0.25: continue
        q=wv*v7+wp*v7p+wn*nn
        d24=(b24-ev(q,2024))*1e5; d22=(b22-ev(q,2022))*1e5
        if d24>0 and d22>0:
            print(f"  v7 {wv:.2f} / exp157 {wp:.2f} / nn {wn:.2f}:  2024 {d24:+6.2f}  2022 {d22:+6.2f}  <<<")
            if best is None or d24+d22>best[0]+best[1]: best=(d24,d22,wv,wp,wn)
if best: print(f"\n  최적 3성분: v7 {best[2]:.2f} / exp157 {best[3]:.2f} / nn {best[4]:.2f}  →  2024 {best[0]:+.2f} / 2022 {best[1]:+.2f}")
else: print("  통과 조합 없음")

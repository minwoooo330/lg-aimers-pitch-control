
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
b["nn"]=0.5*((3*a3+5*a5)/8)+0.5*b8
b["base"]=0.55*b["v7"]+0.45*b["nn"]
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",
               usecols=["row_id","game_type","balls_before","strikes_before","season"])
d=b.merge(tr,on="row_id",how="left",suffixes=("","_x"))
d=d[(d.game_type=="R")|(d.season>=2023)].copy()
d["cnt"]=d.balls_before.astype(str)+"-"+d.strikes_before.astype(str)
out={}
for yr in (2024,2022):
    s=d[d.season==yr].copy()
    p=s.base.to_numpy(); yy=s[TARGET].to_numpy()
    p=p+(yy.mean()-p.mean()); s["res"]=yy-p
    gg=s.groupby("cnt").agg(n=("res","size"),bias=("res","mean"))
    gg["se"]=np.sqrt(0.25/gg.n)          # 편향의 표준오차(대략)
    out[yr]=gg
m=out[2024][["n","bias","se"]].join(out[2022][["n","bias","se"]],lsuffix="_24",rsuffix="_22")
m["일치"]=np.sign(m.bias_24)==np.sign(m.bias_22)
m["유의_24"]=m.bias_24.abs()>2*m.se_24
m["유의_22"]=m.bias_22.abs()>2*m.se_22
print("=== 카운트별 편향: 2024 vs 2022 ===")
print(m.assign(bias_24=m.bias_24.round(5),bias_22=m.bias_22.round(5),
               se_24=m.se_24.round(5),se_22=m.se_22.round(5))
       [["n_24","bias_24","se_24","유의_24","n_22","bias_22","se_22","유의_22","일치"]].to_string())
r=np.corrcoef(m.bias_24,m.bias_22)[0,1]
print(f"\n두 fold 편향의 상관 = {r:.4f}")
print(f"부호 일치 {int(m['일치'].sum())}/{len(m)}개")
print(f"2024에서 2σ 유의 {int(m['유의_24'].sum())}개,  2022에서 {int(m['유의_22'].sum())}개")

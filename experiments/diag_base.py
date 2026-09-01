
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
cols=["row_id","game_type","game_month","balls_before","strikes_before","asof_pitcher_n",
      "pitcher_hand","batter_hand","inning","li","num_runners_on","asof_batter_n","season"]
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",usecols=cols)
d=b.merge(tr,on="row_id",how="left",suffixes=("","_x"))
d=d[(d.game_type=="R")|(d.season>=2023)].copy()
d=d[d.season==2024].copy()                      # 주 판정 fold
p=d.base.to_numpy(); y=d[TARGET].to_numpy()
p=p+(y.mean()-p.mean())                          # 평균정렬
d["p"]=p; d["res"]=y-p; d["se"]=(y-p)**2
print(f"2024 평가행 {len(d):,}   Brier {d.se.mean():.8f}   예측SD {p.std():.5f}\n")
print("=== 1) 보정 상태: 예측 20분위별 실제-예측 ===")
d["q"]=pd.qcut(d.p,20,labels=False)
g1=d.groupby("q").agg(n=("res","size"),pred=("p","mean"),act=(TARGET,"mean"))
g1["gap"]=g1.act-g1.pred
print(f"  |편차| 평균 {g1.gap.abs().mean():.5f}  최대 {g1.gap.abs().max():.5f}")
print(f"  신뢰도 기울기 {np.polyfit(g1.pred,g1.act,1)[0]:.4f}")
print("\n=== 2) 세그먼트별 잔차평균(편향)과 Brier 기여 ===")
def seg(name, key):
    gg=d.groupby(key).agg(n=("res","size"),bias=("res","mean"),brier=("se","mean"))
    gg=gg[gg.n>=3000]
    gg["기여"]=gg.n/len(d)*gg.brier
    contrib=float((gg.n/len(d)*gg.bias**2).sum())
    print(f"\n[{name}]  편향제곱 가중합 = {contrib*1e5:.2f}e-5  (완벽 세그먼트보정 시 최대 이득)")
    print(gg.assign(bias=gg.bias.round(5),brier=gg.brier.round(6)).sort_values("bias").head(4).to_string())
    print("  ...")
    print(gg.assign(bias=gg.bias.round(5),brier=gg.brier.round(6)).sort_values("bias").tail(3).to_string())
d["cnt"]=d.balls_before.astype(str)+"-"+d.strikes_before.astype(str)
d["nbin"]=pd.cut(d.asof_pitcher_n,[0,200,1000,3000,8000,10**9],labels=["~200","~1k","~3k","~8k","8k+"])
d["same"]=(d.pitcher_hand==d.batter_hand)
for nm,k in [("카운트","cnt"),("투수표본","nbin"),("월","game_month"),("리그","game_type"),
             ("이닝",pd.cut(d.inning,[0,3,6,9,99],labels=["1-3","4-6","7-9","10+"])),("동손","same")]:
    seg(nm,k)

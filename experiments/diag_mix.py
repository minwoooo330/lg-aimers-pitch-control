
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
b["nn"]=0.5*((3*a3+5*a5)/8)+0.5*b8
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",usecols=["row_id","game_type","season"])
d=b.merge(tr,on="row_id",how="left",suffixes=("","_x"))
d=d[(d.game_type=="R")|(d.season>=2023)]
for yr in (2024,2022):
    s=d[d.season==yr]
    y=s[TARGET].to_numpy(); v=s.v7.to_numpy(); n=s.nn.to_numpy()
    v=v+(y.mean()-v.mean()); n=n+(y.mean()-n.mean())     # 각각 평균정렬 후 비교
    e1=v-y; e2=n-y
    A=np.mean(e1*e1); B=np.mean(e2*e2); Cv=np.mean(e1*e2)
    w=(B-Cv)/(A+B-2*Cv)
    best=w*w*A+2*w*(1-w)*Cv+(1-w)*(1-w)*B
    cur=0.55*0.55*A+2*0.55*0.45*Cv+0.45*0.45*B
    rho=Cv/np.sqrt(A*B)
    print(f"[{yr}]")
    print(f"  v7 오차분산 {A:.8f}   nn 오차분산 {B:.8f}   오차상관 {rho:.6f}")
    print(f"  이론 최적 w(v7 비중) = {w:.4f}   →  Brier {best:.8f}")
    print(f"  현재 w=0.55           →  Brier {cur:.8f}   차이 {(cur-best)*1e5:+.3f}e-5")
    print(f"  단일 최고(v7 {A:.8f}) 대비 혼합 이득 {(min(A,B)-best)*1e5:+.2f}e-5")
    # 두 모델이 완전 무상관이었다면?
    bi=A*B/(A+B)
    print(f"  [참고] 오차상관이 0이었다면 Brier {bi:.8f} ({(min(A,B)-bi)*1e5:+.2f}e-5 이득 가능했음)")
    print()

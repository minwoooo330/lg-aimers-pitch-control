
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
nn_const3=g("exp120_nn_sd_oof.csv.gz")          # 4에폭 상수LR, 시드3
nn_cos3  =g("exp166_nncos_ep4_cos1_oof.csv.gz") # 4에폭 코사인, 시드3
a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
nn16=0.5*((3*nn_const3+5*a5)/8)+0.5*b8          # 현행 배포 NN(16시드)
tr=pd.read_csv(HERE/"data"/"train.csv",usecols=[ID,"season","game_type",TARGET],encoding="utf-8-sig")
d=b.drop(columns=[c for c in ("season",TARGET) if c in b.columns]).merge(tr,on=ID,how="left")
d["nn_const3"]=nn_const3; d["nn_cos3"]=nn_cos3; d["nn16"]=nn16
d=d[(d.game_type=="R")|(d.season>=2023)]
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def sg(z): return 1/(1+np.exp(-z))
CEN=-0.027330
def chain(v7,nn,k=1.08): p=0.55*v7+0.45*nn; return sg(CEN+k*(lg(p)-CEN))
def paired(pa,pb,y):
    qa=pa+(y.mean()-pa.mean()); qb=pb+(y.mean()-pb.mean())
    dd=(y-qa)**2-(y-qb)**2
    return dd.mean()*1e5, dd.std(ddof=1)/np.sqrt(len(dd))*1e5
print("="*76)
print("코사인 스케줄 — 체인 기여 (시드 3개로 통일해 스케줄 효과만 분리)")
print("="*76)
for yr in (2024,2022):
    s=d[d.season==yr]; y=s[TARGET].to_numpy(); v7=s.v7.to_numpy()
    A=chain(v7,s.nn_const3.to_numpy())   # 상수LR 3시드
    B=chain(v7,s.nn_cos3.to_numpy())     # 코사인 3시드
    m,se=paired(A,B,y)
    print(f"\n[{yr}] NN슬롯 전체를 코사인3시드로  {m:+7.3f}e-5  (SE {se:.3f}, {m/se:+.1f}σ)")
    # 현행 16시드 배포본 기준: 코사인 3시드를 추가 멤버로
    base=chain(v7,s.nn16.to_numpy())
    for w in (0.10,0.20,0.30,0.45):
        nnmix=(1-w)*s.nn16.to_numpy()+w*s.nn_cos3.to_numpy()
        m2,se2=paired(base,chain(v7,nnmix),y)
        print(f"      현행16시드에 코사인 {int(w*100):2d}% 섞기   {m2:+7.3f}e-5  ({m2/se2:+.1f}σ)")

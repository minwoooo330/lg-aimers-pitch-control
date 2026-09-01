
import numpy as np, pandas as pd, itertools
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
b["nn_prob"]=0.5*((3*a3+5*a5)/8)+0.5*b8
tr=pd.read_csv(HERE/"data"/"train.csv",usecols=[ID,"season","game_type",TARGET],encoding="utf-8-sig")
d=b.drop(columns=[c for c in ("season",TARGET) if c in b.columns]).merge(tr,on=ID,how="left")
d=d[(d.game_type=="R")|(d.season>=2023)].reset_index(drop=True)
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def sg(z): return 1/(1+np.exp(-z))
def sc(p,y): p=p+(y.mean()-p.mean()); return float(np.mean((y-p)**2))
def pse(p1,p2,y):
    """짝지은 표준오차: 두 예측의 Brier 차이에 대한 SE"""
    q1=p1+(y.mean()-p1.mean()); q2=p2+(y.mean()-p2.mean())
    dd=(y-q1)**2-(y-q2)**2
    return dd.mean(), dd.std(ddof=1)/np.sqrt(len(dd))
# 로짓공간 시드평균 버전
d["nn_logit"]=sg(0.5*((3*lg(a3[d.index.values] if False else 0)) ) ) if False else 0.0
la3=lg(a3); la5=lg(a5); lb8=lg(b8)
d["nn_logit"]=sg((0.5*((3*la3+5*la5)/8)+0.5*lb8)[d.index.values] if False else np.nan)
# 인덱스 정합을 위해 재계산
mask=b[ID].isin(d[ID])
d["nn_logit"]=sg(0.5*((3*la3+5*la5)/8)+0.5*lb8)[mask.to_numpy()]
CEN=-0.027330
print("="*80)
print("세부 항목 일괄 점검 — 기존 OOF만 사용, 두 fold 모두 보고")
print("="*80)
for yr in (2024,2022):
    s=d[d.season==yr]; y=s[TARGET].to_numpy()
    v7=s.v7.to_numpy(); npb=s.nn_prob.to_numpy(); npl=s.nn_logit.to_numpy()
    def blend(nn,w=0.55,c=CEN,k=1.08):
        p=w*v7+(1-w)*nn; return sg(c+k*(lg(p)-c))
    base=blend(npb)
    b0=sc(base,y)
    print(f"\n[{yr}]  현행 베이스 {b0:.8f}")
    # A) NN 시드평균: 확률공간 vs 로짓공간
    m,se=pse(blend(npl),base,y)
    print(f"  A) NN시드평균을 로짓공간에서       {(-m)*1e5:+7.3f}e-5  (SE {se*1e5:.3f}, {-m/se:+.1f}σ)")
    # B) 혼합 가중치 미세조정
    for w in (0.50,0.525,0.575,0.60):
        m,se=pse(blend(npb,w=w),base,y)
        print(f"  B) 혼합가중 v7 {w:.3f}              {(-m)*1e5:+7.3f}e-5  ({-m/se:+.1f}σ)")
    # C) 샤프닝 center
    for c in (-0.10,-0.06,0.0,0.03):
        m,se=pse(blend(npb,c=c),base,y)
        print(f"  C) 샤프닝 center {c:+.3f}          {(-m)*1e5:+7.3f}e-5  ({-m/se:+.1f}σ)")
    # D) 샤프닝 scale 재확인
    for k in (1.06,1.07,1.09,1.10):
        m,se=pse(blend(npb,k=k),base,y)
        print(f"  D) 샤프닝 scale {k:.2f}             {(-m)*1e5:+7.3f}e-5  ({-m/se:+.1f}σ)")

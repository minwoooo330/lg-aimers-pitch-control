# -*- coding: utf-8 -*-
"""병합 베이스(oct: v7 55% + nnsd16 45%) 기준 후보 판정.
   사용: python eval_oct.py <oof파일명> [예측컬럼]
   출력: 상관, 단독 성능, 그리고 (A)추가형 (B)nnsd 교체형 두 경로의 fold별 기여."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"

def load_base():
    b=pd.read_csv(RES/"exp148_v7_oof.csv.gz")          # v7 재현 (2024/2022)
    b=b.rename(columns={"prediction":"v7"})
    def g(fn):
        d=pd.read_csv(RES/fn).set_index(ID)["prediction"]
        return b[ID].map(d).to_numpy(np.float64)
    a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz")
    b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
    avg8=(3*a3+5*a5)/8.0; avg16=0.5*avg8+0.5*b8
    b["nnsd16"]=avg16
    b["base"]=0.55*b["v7"]+0.45*b["nnsd16"]
    return b

def ev(p,y,season,rel,yr):
    m=(season==yr)&rel&np.isfinite(p)
    z=p[m]+(y[m].mean()-p[m].mean())
    return float(np.mean((z-y[m])**2))

def main():
    b=load_base()
    y=b[TARGET].to_numpy(); season=b["season"].to_numpy()
    gt=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",usecols=[ID,"game_type"])
    gtm=b[ID].map(gt.set_index(ID)["game_type"])
    rel=((gtm=="R")|(season>=2023)).to_numpy()
    BASE=b["base"].to_numpy()
    b24,b22=ev(BASE,y,season,rel,2024),ev(BASE,y,season,rel,2022)
    print(f"[베이스] v7 55% + nnsd16 45%   2024 {b24:.8f}   2022 {b22:.8f}")
    if len(sys.argv)<2:
        print("  (후보 파일 없이 실행: 베이스만 출력)"); return
    col=sys.argv[2] if len(sys.argv)>2 else "prediction"
    d=pd.read_csv(RES/sys.argv[1]).set_index(ID)[col]
    v=b[ID].map(d).to_numpy(np.float64); m=np.isfinite(v)
    print(f"\n[후보] {sys.argv[1]}  결측 {int((~m).sum()):,}")
    print(f"  상관(베이스) {np.corrcoef(BASE[m],v[m])[0,1]:.4f}   상관(nnsd16) {np.corrcoef(b['nnsd16'].to_numpy()[m],v[m])[0,1]:.4f}")
    print(f"  단독  2024 {ev(v,y,season,rel,2024):.8f}   2022 {ev(v,y,season,rel,2022):.8f}")
    print(f"  (참고 nnsd16 단독 2024 {ev(b['nnsd16'].to_numpy(),y,season,rel,2024):.8f})")
    print("\n  (A) 베이스에 추가")
    for w in (0.05,0.10,0.15,0.20,0.25,0.30):
        q=BASE.copy(); q[m]=(1-w)*BASE[m]+w*v[m]
        d24=(b24-ev(q,y,season,rel,2024))*1e5; d22=(b22-ev(q,y,season,rel,2022))*1e5
        print(f"    {int(w*100):>2}%: 2024 {d24:+6.2f}  2022 {d22:+6.2f}{'  <<<' if d24>0 and d22>0 else ''}")
    print("\n  (B) nnsd16 슬롯을 후보로 교체/혼합 (v7 55% 고정)")
    nn=b["nnsd16"].to_numpy()
    for w in (0.0,0.25,0.5,0.75,1.0):
        newnn=np.where(m,(1-w)*nn+w*v,nn)
        q=0.55*b["v7"].to_numpy()+0.45*newnn
        d24=(b24-ev(q,y,season,rel,2024))*1e5; d22=(b22-ev(q,y,season,rel,2022))*1e5
        print(f"    후보비중 {int(w*100):>3}%: 2024 {d24:+6.2f}  2022 {d22:+6.2f}{'  <<<' if d24>0 and d22>0 else ''}")
if __name__=="__main__": main()

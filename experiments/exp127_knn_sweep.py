# -*- coding: utf-8 -*-
"""exp127 — k-NN 한정 튜닝 스윕. '튜닝하면 격차 60e-5 안에 들어오는가'만 판정한다.

  개선 후보 4가지를 동시에 본다:
   (1) 참조 집합 크기  200k -> 600k (이웃 밀도 증가)
   (2) k값            100 / 500 / 2000 / 8000  (거리행렬 1회 계산으로 동시 평가)
   (3) 거리 가중 평균  uniform vs inverse-distance
   (4) 피처 가중       equal vs 중요도 가중(투수실력·당해폼 축 강화)
  속도: 검증셋 2만 행 표본 + top-8000 한 번 정렬 후 누적합으로 모든 k 동시 산출."""
from pathlib import Path
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
import importlib.util
spec=importlib.util.spec_from_file_location("knn",HERE/"exp126_knn.py")
knn=importlib.util.module_from_spec(spec); spec.loader.exec_module(knn)
ID,TARGET="row_id","control_success"
KS=[100,500,2000,8000]; KMAX=max(KS); ALPHA=50.0; NQ=20000; CHUNK=500
# 피처 순서: [투수성공, 투수가운데, 투수반대, 당해폼sd, 타자성공, 볼, 스트, 아웃, 이닝, logLI]
W_EQ =np.ones(10)
W_IMP=np.array([2.0,1.0,1.0,2.0,1.0,1.5,1.5,0.5,0.5,0.5])

def evaluate(Xq,yq,Xr,yr,relq,wt_name):
    prior=float(yr.mean())
    rn=(Xr**2).sum(1).astype(np.float32); Xr32=Xr.astype(np.float32); yr32=yr.astype(np.float32)
    preds={(k,mode):np.empty(len(Xq)) for k in KS for mode in ["uni","inv"]}
    for st in range(0,len(Xq),CHUNK):
        q=Xq[st:st+CHUNK].astype(np.float32)
        d2=rn[None,:]-2.0*(q@Xr32.T)
        idx=np.argpartition(d2,KMAX,axis=1)[:,:KMAX]
        dd=np.take_along_axis(d2,idx,axis=1)
        order=np.argsort(dd,axis=1)
        idx=np.take_along_axis(idx,order,axis=1); dd=np.take_along_axis(dd,order,axis=1)
        nb=yr32[idx]
        csum=np.cumsum(nb,axis=1)
        dd=dd-dd.min(axis=1,keepdims=True)+1e-6      # 음수 방지(순위 보존형 d2였으므로 시프트)
        w=1.0/np.sqrt(dd)
        cw=np.cumsum(w,axis=1); cwy=np.cumsum(w*nb,axis=1)
        for k in KS:
            preds[(k,"uni")][st:st+CHUNK]=(csum[:,k-1]+ALPHA*prior)/(k+ALPHA)
            preds[(k,"inv")][st:st+CHUNK]=(cwy[:,k-1]+ALPHA*prior)/(cw[:,k-1]+ALPHA)
        del d2,idx,dd,nb,csum,w,cw,cwy
    out=[]
    for (k,mode),p in preds.items():
        z=p[relq]+(yq[relq].mean()-p[relq].mean())
        out.append({"wt":wt_name,"k":k,"mode":mode,"brier":float(np.mean((z-yq[relq])**2))})
    return out

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    year=2024
    tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
    TAB=knn.build_tables(df[df.season<year], year)
    Xtr=knn.make_space(tr,TAB); Xva=knn.make_space(va,TAB)
    med=np.nanmedian(Xtr,axis=0)
    Xtr=np.where(np.isfinite(Xtr),Xtr,med); Xva=np.where(np.isfinite(Xva),Xva,med)
    mu=Xtr.mean(0); sg=Xtr.std(0); sg[sg==0]=1
    Xtr=(Xtr-mu)/sg; Xva=(Xva-mu)/sg
    ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    relv=((va.game_type=="R")|(va.season>=2023)).to_numpy()
    rng=np.random.default_rng(0)
    qsel=rng.choice(len(Xva),size=NQ,replace=False)
    Xq=Xva[qsel]; yq=yva[qsel]; relq=relv[qsel]
    print(f"질의 표본 {NQ:,}행  학습 {len(Xtr):,}행  ({time.time()-t0:.0f}s)",flush=True)
    rows=[]
    for nref in [200000,600000]:
        sel=rng.choice(len(Xtr),size=min(nref,len(Xtr)),replace=False)
        for wname,wv in [("equal",W_EQ),("imp",W_IMP)]:
            tt=time.time()
            res=evaluate(Xq*wv,yq,Xtr[sel]*wv,ytr[sel],relq,wname)
            for r in res: r["n_ref"]=nref
            rows+=res
            best=min(res,key=lambda z:z["brier"])
            print(f"n_ref={nref:,} wt={wname:5s} 최적 k={best['k']:>4d} {best['mode']} brier={best['brier']:.8f}  ({time.time()-tt:.0f}s)",flush=True)
            pd.DataFrame(rows).to_csv(RES/"exp127_knn_sweep.csv",index=False,encoding="utf-8-sig")
    r=pd.DataFrame(rows).sort_values("brier")
    print("\n=== 상위 10 ===")
    print(r.head(10).to_string(index=False))
    print(f"\n[기준] exp126 미튜닝(k=500,200k,equal,uni) 전체검증 0.24880114")
    print(f"[기준] 챔피언 2024 0.24750439  → 통과선(격차 60e-5) = 0.24810")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

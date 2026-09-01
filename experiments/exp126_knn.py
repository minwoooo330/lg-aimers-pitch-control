# -*- coding: utf-8 -*-
"""exp126 — k-NN(과거 유사 사례 조회). 규칙 학습이 아니라 '비슷했던 과거 투구'를 직접 찾아 평균낸다.

  트리/신경망과 근본적으로 다른 추정 방식(전역 규칙 vs 국소 평균)이라 오차 패턴이 다를 여지가 있다.
  규정 준수: 참조 집합(reference)은 학습 데이터로만 만들고, test 각 행은 자기 값으로만 조회한다.
             test 행끼리 서로 참조하지 않으므로 sd 조회표와 동일한 구조.

  거리 공간 설계: 160개 전 피처를 쓰면 차원의 저주로 거리가 무의미해진다.
                 검증된 핵심 축 10개만 표준화해서 사용.
  k를 크게(500) 잡는 이유: 이 문제의 신호가 약해(AUC 0.55) 작은 k는 순수 잡음이 된다."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
RATES=["success_rate","middle_rate","reverse_rate"]; BRATES=["success_rate","middle_rate"]
N_REF=200000; K=500; ALPHA=50.0; CHUNK=500

def end_state(d, upto):
    s=d[d.season<=upto]
    if len(s)==0: return None
    idx=s.groupby("pitcher_id")["asof_pitcher_n"].idxmax(); last=s.loc[idx]
    t={"n":pd.Series(last.asof_pitcher_n.to_numpy(),index=last.pitcher_id.to_numpy())}
    for r in RATES: t[r]=pd.Series(last["asof_pitcher_"+r].to_numpy(),index=last.pitcher_id.to_numpy())
    return t

def build_tables(df, max_season):
    return {S: end_state(df, S-1) for S in range(int(df.season.min())+1, max_season+1)}

def sd_success(d, tables):
    """당해 시즌 성공률(sd) — 거리 공간의 핵심 축 하나로 사용."""
    n=d.asof_pitcher_n.to_numpy(np.float64); pid=d.pitcher_id.to_numpy(); seas=d.season.to_numpy()
    n0=np.full(len(d),np.nan); r0=np.full(len(d),np.nan)
    for S,tbl in tables.items():
        if tbl is None: continue
        mm=(seas==S)
        if not mm.any(): continue
        sub=pd.Series(pid[mm])
        n0[mm]=sub.map(tbl["n"]).to_numpy(np.float64)
        r0[mm]=sub.map(tbl["success_rate"]).to_numpy(np.float64)
    dn=n-n0; valid=np.isfinite(dn)&(dn>=20)
    cur=d.asof_pitcher_success_rate.to_numpy(np.float64)
    with np.errstate(invalid="ignore",divide="ignore"):
        rate=(cur*n-r0*n0)/dn
    return np.where(valid,np.clip(rate,0,1),cur)   # 결측이면 통산으로 대체

def make_space(d, tables):
    """거리 공간 10축: 투수 실력 3 + 당해폼 1 + 타자 1 + 상황 5"""
    sd=sd_success(d,tables)
    X=np.column_stack([
        d.asof_pitcher_success_rate.to_numpy(np.float64),
        d.asof_pitcher_middle_rate.to_numpy(np.float64),
        d.asof_pitcher_reverse_rate.to_numpy(np.float64),
        sd,
        d.asof_batter_success_rate.to_numpy(np.float64),
        d.balls_before.to_numpy(np.float64),
        d.strikes_before.to_numpy(np.float64),
        d.outs_before.to_numpy(np.float64),
        np.minimum(d.inning.to_numpy(np.float64),10.0),
        np.log1p(np.maximum(d.li.to_numpy(np.float64),0)),
    ])
    return X

def knn_predict(Xq, Xr, yr, k=K, alpha=ALPHA, chunk=CHUNK):
    """||q-r||^2 = ||q||^2 + ||r||^2 - 2q.r  → 순위에는 ||q||^2 불필요."""
    prior=float(yr.mean())
    rn=(Xr**2).sum(1).astype(np.float32)
    Xr32=Xr.astype(np.float32); yr32=yr.astype(np.float32)
    out=np.empty(len(Xq),dtype=np.float64)
    for st in range(0,len(Xq),chunk):
        q=Xq[st:st+chunk].astype(np.float32)
        d2=rn[None,:]-2.0*(q@Xr32.T)          # (c, R)
        idx=np.argpartition(d2,k,axis=1)[:,:k]  # 가장 가까운 k개
        nb=yr32[idx]                            # (c, k)
        s=nb.sum(1)
        out[st:st+chunk]=(s+alpha*prior)/(k+alpha)   # prior로 축소(shrinkage)
        del d2,idx,nb
    return out

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; parts=[]
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB=build_tables(df[df.season<year], year)
        Xtr=make_space(tr,TAB); Xva=make_space(va,TAB)
        med=np.nanmedian(Xtr,axis=0)
        Xtr=np.where(np.isfinite(Xtr),Xtr,med); Xva=np.where(np.isfinite(Xva),Xva,med)
        mu=Xtr.mean(0); sg=Xtr.std(0); sg[sg==0]=1
        Xtr=(Xtr-mu)/sg; Xva=(Xva-mu)/sg
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rng=np.random.default_rng(42)
        sel=rng.choice(len(Xtr),size=min(N_REF,len(Xtr)),replace=False)
        Xr=Xtr[sel]; yr=ytr[sel]
        print(f"[{year}] 참조 {len(Xr):,}행  질의 {len(Xva):,}행  ({time.time()-t0:.0f}s)",flush=True)
        tt=time.time()
        p=knn_predict(Xva,Xr,yr)
        relm=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        z=p[relm]+(yva[relm].mean()-p[relm].mean())
        br=float(np.mean((z-yva[relm])**2))
        rows.append({"fold":year,"k":K,"n_ref":len(Xr),"brier_rel_aligned":br,"sec":round(time.time()-tt)})
        print(rows[-1],flush=True)
        parts.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}))
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp126_knn_oof.csv.gz",index=False,compression="gzip")
        pd.DataFrame(rows).to_csv(RES/"exp126_knn.csv",index=False,encoding="utf-8-sig")
        gc.collect()
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

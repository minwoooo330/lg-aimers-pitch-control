# -*- coding: utf-8 -*-
"""실험 117b: 월별 당해 성적 채널 — 효율적 재구현 (groupby+shift 단일 패스).
   exp117의 O(그룹수 x 전체행수) 버그를 제거: 투수x시즌 단위로 정렬 후 월별 마지막 상태를
   shift(1)로 한 번에 계산한다."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from features import add_features
from hfeatures import add_hfeatures
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
SEEDS=[42,7,2024]

def month_end_state(df):
    """(pitcher_id,season,month)별 '그 달 마지막 행'의 누적 성공/가운데 횟수, n.
       투수x시즌 단위로 asof_pitcher_n(시간순 정렬 키)으로 정렬 후 월별 tail(1)만 뽑는다 -> O(n)."""
    d=df[["pitcher_id","season","game_month","asof_pitcher_n",
          "asof_pitcher_success_rate","asof_pitcher_middle_rate"]].copy()
    d["cumsucc"]=d.asof_pitcher_success_rate*d.asof_pitcher_n
    d["cummid"]=d.asof_pitcher_middle_rate*d.asof_pitcher_n
    d=d.sort_values(["pitcher_id","season","asof_pitcher_n"])
    last=d.groupby(["pitcher_id","season","game_month"],sort=False).tail(1)
    last=last.sort_values(["pitcher_id","season","game_month"])
    return last  # 컬럼: pitcher_id, season, game_month, asof_pitcher_n, cumsucc, cummid (그 달 종료 시점)

def add_month_vec(d, monthend):
    """완전 벡터화: merge_asof(by=[pitcher_id,season])로 그룹별 백워드 조인을 한 번에 수행."""
    left=d[["row_id","pitcher_id","season","game_month","asof_pitcher_n",
            "asof_pitcher_success_rate","asof_pitcher_middle_rate"]].copy()
    left=left.sort_values("game_month")
    right=monthend[["pitcher_id","season","game_month","asof_pitcher_n","cumsucc","cummid"]].rename(
        columns={"asof_pitcher_n":"n_end"}).sort_values("game_month")
    merged=pd.merge_asof(left,right,on="game_month",by=["pitcher_id","season"],
                          direction="backward",allow_exact_matches=False)
    out=merged.set_index("row_id").reindex(d.row_id).reset_index()
    n=out.asof_pitcher_n.to_numpy(np.float64); n0=out.n_end.to_numpy(np.float64)
    dn=n-n0; valid=np.isfinite(dn)&(dn>=10)
    cur_cs=out.asof_pitcher_success_rate.to_numpy()*n
    cur_cm=out.asof_pitcher_middle_rate.to_numpy()*n
    f=pd.DataFrame(index=d.index)
    f["mo_logn"]=np.where(valid,np.log1p(np.maximum(dn,0)),np.nan)
    f["mo_isfirst"]=(~np.isfinite(n0)).astype(np.int8)
    rate=np.where(valid,np.clip((cur_cs-out.cumsucc.to_numpy())/np.maximum(dn,1),0,1),np.nan)
    f["mo_success_rate"]=rate
    f["mo_delta_success"]=np.where(valid,rate-out.asof_pitcher_success_rate.to_numpy(),np.nan)
    midr=np.where(valid,np.clip((cur_cm-out.cummid.to_numpy())/np.maximum(dn,1),0,1),np.nan)
    f["mo_middle_rate"]=midr
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ME_TR=month_end_state(tr); ME_VA=month_end_state(va)   # va는 자기 시즌 내부만 조회(안전)
        print(f"[{year}] 월말상태 테이블: 학습 {len(ME_TR):,}행 검증 {len(ME_VA):,}행  ({time.time()-t0:.0f}s)",flush=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,me,mo_on):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            parts=[x,add_features(d),add_hfeatures(d)]
            if mo_on: parts.append(add_month_vec(d,me))
            return pd.concat(parts,axis=1)
        for name,mo_on,me_tr,me_va in [("base",False,None,None),("month",True,ME_TR,ME_VA)]:
            tt=time.time()
            xa=enc(tr,me_tr,mo_on); xb=enc(va,me_va,mo_on)
            print(f"    {name} 피처생성 완료 {xa.shape} ({time.time()-tt:.0f}s)",flush=True)
            cm=[cc in CAT for cc in xa.columns]
            ps=[]
            for seed in SEEDS:
                m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                    min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                    random_state=seed,categorical_features=cm).fit(xa,ytr)
                ps.append(m.predict_proba(xb)[:,1]); del m; gc.collect()
            avg=np.mean(ps,axis=0)
            br=float(np.mean((avg[rel]-yva[rel])**2))
            rows.append({"fold":year,"cfg":name,"n_feat":xa.shape[1],"brier_rel":br})
            print(f"  {year} {name:6s} 피처 {xa.shape[1]:3d}  단독 Brier {br:.8f}  ({time.time()-t0:.0f}s)",flush=True)
            store[(year,name)]=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":avg})
            del xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp117b_month_todate.csv",index=False,encoding="utf-8-sig")
    for name in ["base","month"]:
        pd.concat([store[(y,name)] for y in [2024,2022] if (y,name) in store],ignore_index=True)\
          .to_csv(RES/f"exp117b_{name}_oof.csv.gz",index=False,compression="gzip")
    r=pd.DataFrame(rows); print()
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.cfg=="base")].brier_rel.iloc[0]
        b_=r[(r.fold==year)&(r.cfg=="month")].brier_rel.iloc[0]
        print(f"  {year}: base {a:.8f} -> +월당해 {b_:.8f}  delta {(a-b_)*1e5:+.3f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

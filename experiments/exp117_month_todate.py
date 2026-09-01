# -*- coding: utf-8 -*-
"""실험 117: 월별 당해 성적 채널 — 시즌당해(exp105)보다 세밀한 시간 단위.
   exp116 스크리닝: 잔차 상관 +0.02842 (시즌당해 +0.018보다 높음).
   같은 트릭을 월 단위로: '이번 달 몇 구 던졌는지, 이번 달 성공률이 얼마인지'를
   현재 행 + 학습데이터 기반 조회표(월별)로 복원. 다른 test 행 미사용."""
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

def month_tables(df):
    """(season,month) -> 그 시즌의 '해당 월 이전' 각 투수 누적 상태."""
    df=df.copy(); df["cumsucc"]=df.asof_pitcher_success_rate*df.asof_pitcher_n
    df["cummid"]=df.asof_pitcher_middle_rate*df.asof_pitcher_n
    tabs={}
    for (S,mo),_ in df.groupby(["season","game_month"]):
        prevm=df[(df.season==S)&(df.game_month<mo)]
        if len(prevm)==0: tabs[(S,mo)]=None; continue
        idx=prevm.groupby("pitcher_id")["asof_pitcher_n"].idxmax(); last=prevm.loc[idx]
        tabs[(S,mo)]={"n":pd.Series(last.asof_pitcher_n.to_numpy(),index=last.pitcher_id.to_numpy()),
                      "cs":pd.Series(last.cumsucc.to_numpy(),index=last.pitcher_id.to_numpy()),
                      "cm":pd.Series(last.cummid.to_numpy(),index=last.pitcher_id.to_numpy())}
    return tabs

def add_month(d, tabs):
    n=d.asof_pitcher_n.to_numpy(np.float64); pid=d.pitcher_id.to_numpy()
    cur_cs=d.asof_pitcher_success_rate.to_numpy()*n
    cur_cm=d.asof_pitcher_middle_rate.to_numpy()*n
    n0=np.full(len(d),np.nan); cs0=np.full(len(d),np.nan); cm0=np.full(len(d),np.nan)
    seas=d.season.to_numpy(); mo=d.game_month.to_numpy()
    for (S,M),tb in tabs.items():
        if tb is None: continue
        mm=(seas==S)&(mo==M)
        if not mm.any(): continue
        sub=pd.Series(pid[mm])
        n0[mm]=sub.map(tb["n"]).to_numpy(np.float64)
        cs0[mm]=sub.map(tb["cs"]).to_numpy(np.float64)
        cm0[mm]=sub.map(tb["cm"]).to_numpy(np.float64)
    dn=n-n0; valid=np.isfinite(dn)&(dn>=10)
    f=pd.DataFrame(index=d.index)
    f["mo_logn"]=np.where(valid,np.log1p(np.maximum(dn,0)),np.nan)
    f["mo_isfirst"]=(~np.isfinite(n0)).astype(np.int8)
    rate=np.where(valid,np.clip((cur_cs-cs0)/np.maximum(dn,1),0,1),np.nan)
    f["mo_success_rate"]=rate
    f["mo_delta_success"]=np.where(valid,rate-d.asof_pitcher_success_rate.to_numpy(),np.nan)
    midr=np.where(valid,np.clip((cur_cm-cm0)/np.maximum(dn,1),0,1),np.nan)
    f["mo_middle_rate"]=midr
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB_TR=month_tables(tr); TAB_VA=month_tables(pd.concat([tr,va]))  # va는 그 시즌 내 이전 달만 쓰므로 안전
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,tab,mo_on):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            parts=[x,add_features(d),add_hfeatures(d)]
            if mo_on: parts.append(add_month(d,tab))
            return pd.concat(parts,axis=1)
        for name,mo_on,tabtr,tabva in [("base",False,None,None),("month",True,TAB_TR,TAB_VA)]:
            xa=enc(tr,tabtr,mo_on); xb=enc(va,tabva,mo_on)
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
        pd.DataFrame(rows).to_csv(RES/"exp117_month_todate.csv",index=False,encoding="utf-8-sig")
    for name in ["base","month"]:
        pd.concat([store[(y,name)] for y in [2024,2022] if (y,name) in store],ignore_index=True)\
          .to_csv(RES/f"exp117_{name}_oof.csv.gz",index=False,compression="gzip")
    r=pd.DataFrame(rows); print()
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.cfg=="base")].brier_rel.iloc[0]
        b_=r[(r.fold==year)&(r.cfg=="month")].brier_rel.iloc[0]
        print(f"  {year}: base {a:.8f} -> +월당해 {b_:.8f}  delta {(a-b_)*1e5:+.3f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

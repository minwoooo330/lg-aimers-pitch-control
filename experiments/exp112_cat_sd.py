# -*- coding: utf-8 -*-
"""실험 112: sd 채널을 CatBoost에 적용 (계열 다양화).
   sd는 '투수 시즌당해 성적'이었다. 같은 차분 트릭이 가능한 미사용 채널:
     ① 타자 시즌당해 (asof_batter_n / success / middle)
     ② 투수 볼·스트라이크 시즌당해 (ball_rate / strike_rate)
     ③ 투수 구종믹스 시즌당해 (pitchmix_n / fastball / breaking / offspeed)
   전부 '현재 행 + 학습데이터 조회표'만 쓰므로 행 독립. 체인 한계기여로 판정한다."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from features import add_features
from hfeatures import add_hfeatures
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
SEEDS=[42,7,2024]
# (엔티티 키, 카운트 컬럼, 비율 컬럼들)
CHANNELS={
 "pit":  ("pitcher_id","asof_pitcher_n",["success_rate","middle_rate","reverse_rate"]),
 "pitbs":("pitcher_id","asof_pitcher_n",["ball_rate","strike_rate"]),
 "bat":  ("batter_id","asof_batter_n",["success_rate","middle_rate"]),
 "mix":  ("pitcher_id","asof_pitcher_pitchmix_n",["fastball_rate","breaking_rate","offspeed_rate"]),
}
PREF={"pit":"asof_pitcher_","pitbs":"asof_pitcher_","bat":"asof_batter_","mix":"asof_pitcher_"}

def end_state(d, upto, key, ncol, rates, pref):
    s=d[d.season<=upto]
    if len(s)==0: return None
    i=s.groupby(key)[ncol].idxmax(); l=s.loc[i]
    t={"n":pd.Series(l[ncol].to_numpy(),index=l[key].to_numpy())}
    for r in rates: t[r]=pd.Series(l[pref+r].to_numpy(),index=l[key].to_numpy())
    return t

def add_ch(d, tabs, tag):
    key,ncol,rates=CHANNELS[tag]; pref=PREF[tag]
    f=pd.DataFrame(index=d.index)
    n=d[ncol].to_numpy(np.float64); ids=d[key].to_numpy(); seas=d.season.to_numpy()
    n0=np.full(len(d),np.nan); prev={r:np.full(len(d),np.nan) for r in rates}
    for S,tb in tabs.items():
        if tb is None: continue
        m=(seas==S)
        if not m.any(): continue
        sub=pd.Series(ids[m])
        n0[m]=sub.map(tb["n"]).to_numpy(np.float64)
        for r in rates: prev[r][m]=sub.map(tb[r]).to_numpy(np.float64)
    dn=n-n0; valid=np.isfinite(dn)&(dn>=20)
    f[f"{tag}_logn"]=np.where(valid,np.log1p(np.maximum(dn,0)),np.nan)
    f[f"{tag}_isnew"]=(~np.isfinite(n0)).astype(np.int8)
    for r in rates:
        cur=d[pref+r].to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*n-prev[r]*n0)/dn
        rate=np.where(valid,np.clip(rate,0.0,1.0),np.nan)
        f[f"{tag}_{r}"]=rate
        f[f"{tag}_d_{r}"]=np.where(valid,rate-cur,np.nan)
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        base=df[df.season<year]
        TABS={tag:{S:end_state(base,S-1,*CHANNELS[tag],PREF[tag])
                   for S in range(int(df.season.min())+1,year+1)} for tag in CHANNELS}
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,tags):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            parts=[x,add_features(d),add_hfeatures(d)]+[add_ch(d,TABS[t],t) for t in tags]
            return pd.concat(parts,axis=1)
        # 채널 조합: 투수(기존 sd 재현) / +타자 / +볼스트 / 전부
        CA=CAT+["pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
        def encC(d,tags):
            x=d[cols].copy()
            for c in CA: x[c]=d[c].where(d[c].notna(),"__MISSING__").astype(str)
            return pd.concat([x,add_features(d),add_hfeatures(d)]+[add_ch(d,TABS[t],t) for t in tags],axis=1)
        for name,tags in [("cat_sd",["pit","bat"])]:
            xa,xb=encC(tr,tags),encC(va,tags)
            idxc=[xa.columns.get_loc(c) for c in CA]
            ps=[]
            for seed in [42]:
                m=CatBoostClassifier(iterations=300,learning_rate=0.06,depth=8,loss_function="Logloss",
                    l2_leaf_reg=3.0,random_seed=seed,thread_count=-1,allow_writing_files=False,
                    verbose=False).fit(xa,ytr,cat_features=idxc)
                ps.append(m.predict_proba(xb)[:,1]); del m; gc.collect()
            avg=np.mean(ps,axis=0)
            br=float(np.mean((avg[rel]-yva[rel])**2))
            rows.append({"fold":year,"cfg":name,"n_feat":xa.shape[1],"brier_rel":br})
            print(f"  {year} {name:11s} 피처 {xa.shape[1]:3d}  단독 Brier {br:.8f}  ({time.time()-t0:.0f}s)",flush=True)
            store[(year,name)]=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":avg})
            del xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp112_cat_sd.csv",index=False,encoding="utf-8-sig")
        for name in ["cat_sd"]:
            pd.concat([store[(y,name)] for y in [2024,2022] if (y,name) in store],ignore_index=True)\
              .to_csv(RES/f"exp112_{name}_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

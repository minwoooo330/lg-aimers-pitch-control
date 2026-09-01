# -*- coding: utf-8 -*-
"""실험 81: hf_v1 제출본과 '같은 조건'의 3-fold OOF — 체인 기여를 제출 전에 로컬로 확정.
   재학습 대상 5멤버(hgb_domain, tm_mean, league_role, catboost, catboost_time)에
   hfeatures 56개를 주입하고 fold별로 학습한다. Trackman 프로필은 fold cutoff 준수.
"""
from pathlib import Path
import gc, time, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from catboost import CatBoostClassifier
from features import add_features
from hfeatures import add_hfeatures
from trackman_features import (match_exact_games, build_pitcher_mapping,
                               build_trackman_profile, build_trackman_profile_by_league)
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent; DATA=HERE/"data"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
CAT_ALL=CATS+["pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
PRM=dict(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,
         l2_regularization=1.,early_stopping=False,random_state=42)

def main():
    t0=time.time()
    df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(DATA/"trackman_history.csv",encoding="utf-8-sig")
    main_g,tm_s,matches=match_exact_games(df,tm)
    print(f"경기매칭 {len(matches)}  ({time.time()-t0:.0f}s)",flush=True)
    rows=[]; store={}
    for year in [2022,2023,2024]:
        mp=build_pitcher_mapping(main_g,tm_s,matches,year)
        prof=build_trackman_profile(tm,mp,year); profL=build_trackman_profile_by_league(tm,mp,year)
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CATS}
        def enc(d):
            x=d[cols].copy()
            for c in CATS: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            return pd.concat([x,add_features(d),add_hfeatures(d)],axis=1)
        xa0,xb0=enc(tr),enc(va)
        def save(name,p):
            rows.append({"fold":year,"model":name,"brier":brier_score_loss(yva,p)})
            print(rows[-1],flush=True)
            store.setdefault(name,[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,
                TARGET:yva,"prediction":p}))
        cm=[c in CATS for c in xa0.columns]
        m=HistGradientBoostingClassifier(**PRM,categorical_features=cm).fit(xa0,ytr)
        save("hf_hgb_domain",m.predict_proba(xb0)[:,1]); del m; gc.collect()
        pa=tr[["pitcher_id"]].join(prof,on="pitcher_id").drop(columns=["pitcher_id"]).reset_index(drop=True)
        pb=va[["pitcher_id"]].join(prof,on="pitcher_id").drop(columns=["pitcher_id"]).reset_index(drop=True)
        xa,xb=pd.concat([xa0,pa],axis=1),pd.concat([xb0,pb],axis=1)
        m=HistGradientBoostingClassifier(**PRM,
            categorical_features=[c in CATS for c in xa.columns]).fit(xa,ytr)
        save("hf_tm_mean",m.predict_proba(xb)[:,1]); del m,xa,xb; gc.collect()
        def joinL(d,x):
            idx=pd.MultiIndex.from_arrays([d.pitcher_id.to_numpy(),d.game_type.to_numpy()])
            r=profL.reindex(idx).reset_index(drop=True)
            keep=[c for c in r.columns if "share" in c or "role" in c or "starter" in c]
            return pd.concat([x,r[keep] if keep else r],axis=1)
        xa,xb=joinL(tr,xa0),joinL(va,xb0)
        m=HistGradientBoostingClassifier(**PRM,
            categorical_features=[c in CATS for c in xa.columns]).fit(xa,ytr)
        save("hf_league_role",m.predict_proba(xb)[:,1]); del m,xa,xb; gc.collect()
        del xa0,xb0; gc.collect()
        def encC(d):
            x=d[cols].copy()
            for c in CAT_ALL: x[c]=d[c].where(d[c].notna(),"__MISSING__").astype(str)
            return pd.concat([x,add_features(d),add_hfeatures(d)],axis=1)
        ca,cb=encC(tr),encC(va); idx=[ca.columns.get_loc(c) for c in CAT_ALL]
        for name,ht in [("hf_cat_domain",False),("hf_cat_time",True)]:
            mc=CatBoostClassifier(iterations=300,learning_rate=0.06,depth=8,loss_function="Logloss",
                l2_leaf_reg=3.0,random_seed=42,thread_count=-1,allow_writing_files=False,
                verbose=False,has_time=ht).fit(ca,ytr,cat_features=idx)
            save(name,mc.predict_proba(cb)[:,1]); del mc; gc.collect()
        del ca,cb,tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp81_hf_members.csv",index=False,encoding="utf-8-sig")
        for k,v in store.items():
            pd.concat(v,ignore_index=True).to_csv(RES/f"exp81_{k}_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

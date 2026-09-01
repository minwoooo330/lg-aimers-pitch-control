# -*- coding: utf-8 -*-
"""실험 76: GBDT 시드 앙상블 (수상자 조언 이행). hgb_domain 시드 7/2024 + cat100 2종 시드 7. 3-fold."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
CAT_ALL=CATS+["pitcher_id","batter_id","pitcher_team_id","batter_team_id"]

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig"); rows=[]; store={}
    cols=[c for c in df.columns if c not in (ID,TARGET)]
    for year in [2022,2023,2024]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        xa,xb=tr[cols].copy(),va[cols].copy()
        for c in CATS:
            vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
            xa[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
            xb[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xa=pd.concat([xa,add_features(tr)],axis=1); xb=pd.concat([xb,add_features(va)],axis=1)
        for seed in [7,2024]:
            mdl=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                categorical_features=[c in CATS for c in xa.columns],random_state=seed).fit(xa,ytr)
            p=mdl.predict_proba(xb)[:,1]
            rows.append({"fold":year,"model":f"hgb_s{seed}","brier":brier_score_loss(yva,p)})
            print(rows[-1],flush=True)
            store.setdefault(f"hgb_s{seed}",[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}))
            del mdl,p; gc.collect()
        ca,cb=tr[cols].copy(),va[cols].copy()
        ca=pd.concat([ca,add_features(tr)],axis=1); cb=pd.concat([cb,add_features(va)],axis=1)
        for c in CAT_ALL:
            ca[c]=ca[c].where(ca[c].notna(),"__MISSING__").astype(str)
            cb[c]=cb[c].where(cb[c].notna(),"__MISSING__").astype(str)
        idx=[ca.columns.get_loc(c) for c in CAT_ALL]
        for name,ht in [("cat","False"),("cattime","True")]:
            for seed in [7]:
                mdl=CatBoostClassifier(iterations=100,learning_rate=0.06,depth=8,loss_function="Logloss",
                    l2_leaf_reg=3.0,random_seed=seed,thread_count=-1,allow_writing_files=False,
                    verbose=False,has_time=(ht=="True")).fit(ca,ytr,cat_features=idx)
                p=mdl.predict_proba(cb)[:,1]
                rows.append({"fold":year,"model":f"{name}100_s{seed}","brier":brier_score_loss(yva,p)})
                print(rows[-1],flush=True)
                store.setdefault(f"{name}100_s{seed}",[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}))
                del mdl,p; gc.collect()
        del xa,xb,ca,cb,tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp76_gbdt_seeds.csv",index=False,encoding="utf-8-sig")
    for k,v in store.items():
        pd.concat(v,ignore_index=True).to_csv(RES/f"exp76_{k}_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""실험 79: 용량 확대 CatBoost 600회 — '2025는 용량에 보상' 가설의 재료(리더보드 탐침용 OOF)."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss
from features import add_features
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT_ALL=["top_bottom","game_type","base_state","pitcher_hand","batter_hand",
         "pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2022,2023,2024]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        def enc(d):
            x=d[cols].copy()
            for c in CAT_ALL: x[c]=d[c].where(d[c].notna(),"__MISSING__").astype(str)
            return pd.concat([x,add_features(d)],axis=1)
        ca,cb=enc(tr),enc(va); idx=[ca.columns.get_loc(c) for c in CAT_ALL]
        for name,ht in [("cat600",False),("cattime600",True)]:
            tt=time.time()
            m=CatBoostClassifier(iterations=600,learning_rate=0.06,depth=8,loss_function="Logloss",
                l2_leaf_reg=3.0,random_seed=42,thread_count=-1,allow_writing_files=False,
                verbose=False,has_time=ht).fit(ca,ytr,cat_features=idx)
            p=m.predict_proba(cb)[:,1]
            rows.append({"fold":year,"model":name,"brier":brier_score_loss(yva,p),"sec":round(time.time()-tt)})
            print(rows[-1],flush=True)
            store.setdefault(name,[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}))
            del m,p; gc.collect()
        del ca,cb,tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp79_cat600.csv",index=False,encoding="utf-8-sig")
        for k,v in store.items():
            pd.concat(v,ignore_index=True).to_csv(RES/f"exp79_{k}_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

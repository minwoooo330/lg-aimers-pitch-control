# -*- coding: utf-8 -*-
"""exp78 잔여분: 2024 fold만 실행해 기존 OOF(2022,2023)에 이어붙인다."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss
from features import add_features
HERE=Path(__file__).resolve().parent; DATA=HERE/"data"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT_ALL=["top_bottom","game_type","base_state","pitcher_hand","batter_hand",
         "pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
t0=time.time(); df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
year=2024
tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
cols=[c for c in df.columns if c not in (ID,TARGET)]
def enc(d):
    x=d[cols].copy()
    for c in CAT_ALL: x[c]=d[c].where(d[c].notna(),"__MISSING__").astype(str)
    return pd.concat([x,add_features(d)],axis=1)
ca,cb=enc(tr),enc(va); idx=[ca.columns.get_loc(c) for c in CAT_ALL]
for name,ht in [("cat300_s7",False),("cattime300_s7",True)]:
    tt=time.time()
    m=CatBoostClassifier(iterations=300,learning_rate=0.06,depth=8,loss_function="Logloss",
        l2_leaf_reg=3.0,random_seed=7,thread_count=-1,allow_writing_files=False,
        verbose=False,has_time=ht).fit(ca,ytr,cat_features=idx)
    p=m.predict_proba(cb)[:,1]
    print({"fold":year,"model":name,"brier":brier_score_loss(yva,p),"sec":round(time.time()-tt)},flush=True)
    old=pd.read_csv(RES/f"exp78_{name}_oof.csv.gz")
    old=old[old.season!=year]
    newd=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p})
    pd.concat([old,newd],ignore_index=True).to_csv(RES/f"exp78_{name}_oof.csv.gz",index=False,compression="gzip")
    del m,p; gc.collect()
print(f"total={time.time()-t0:.1f}s",flush=True)

# -*- coding: utf-8 -*-
"""실험 78: CatBoost 300회 시드7 — '교체'가 아니라 '추가' 형태의 시드 다양성.
cs100 실패(986.86, -8.11점) 진단: 교체·제거형은 로컬이 양수여도 리더보드에서 진다(4패).
시드 평균의 이득만 추가형(11전 11승)으로 회수할 수 있는지 본다. 반복수는 체인과 동일한 300회.
"""
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

def main():
    t0=time.time(); df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
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
        for name,ht in [("cat300_s7",False),("cattime300_s7",True)]:
            tt=time.time()
            m=CatBoostClassifier(iterations=300,learning_rate=0.06,depth=8,loss_function="Logloss",
                l2_leaf_reg=3.0,random_seed=7,thread_count=-1,allow_writing_files=False,
                verbose=False,has_time=ht).fit(ca,ytr,cat_features=idx)
            p=m.predict_proba(cb)[:,1]
            rows.append({"fold":year,"model":name,"brier":brier_score_loss(yva,p),"sec":round(time.time()-tt)})
            print(rows[-1],flush=True)
            store.setdefault(name,[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,
                TARGET:yva,"prediction":p}))
            del m,p; gc.collect()
        del ca,cb,tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp78_cat300_seed7.csv",index=False,encoding="utf-8-sig")
        for k,v in store.items():
            pd.concat(v,ignore_index=True).to_csv(RES/f"exp78_{k}_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

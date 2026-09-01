# -*- coding: utf-8 -*-
"""실험 33: CatBoost를 2000회까지 학습하고 중간 지점마다 2024 성능 측정.
   한 번 학습으로 반복 횟수 곡선 전체를 얻는다 (현재 우리 설정은 300회)."""
from pathlib import Path
import time
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT_ALL=["top_bottom","game_type","base_state","pitcher_hand","batter_hand",
         "pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
CHECKS=[100,200,300,400,600,800,1000,1250,1500,1750,2000]

def main():
    start=time.time()
    df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    tr=df[df.season<2024]; va=df[df.season==2024]
    xtr=pd.concat([tr[base].copy(),add_features(tr)],axis=1)
    xva=pd.concat([va[base].copy(),add_features(va)],axis=1)
    for c in CAT_ALL:
        xtr[c]=xtr[c].where(xtr[c].notna(),"__MISSING__").astype(str)
        xva[c]=xva[c].where(xva[c].notna(),"__MISSING__").astype(str)
    ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    idx=[xtr.columns.get_loc(c) for c in CAT_ALL]

    m=CatBoostClassifier(iterations=2000,learning_rate=0.06,depth=8,loss_function="Logloss",
                         l2_leaf_reg=3.0,random_seed=42,thread_count=-1,
                         allow_writing_files=False,verbose=200)
    print("학습 시작 (2000회)",flush=True)
    m.fit(xtr,ytr,cat_features=idx)
    print(f"학습 완료 {time.time()-start:.0f}s",flush=True)

    rows=[]
    for n in CHECKS:
        p=m.predict_proba(xva,ntree_end=n)[:,1]
        rows.append({"iterations":n,"brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p)})
        print(rows[-1],flush=True)
    res=pd.DataFrame(rows); RES.mkdir(exist_ok=True)
    res.to_csv(RES/"exp33_catboost_iteration_curve.csv",index=False,encoding="utf-8-sig")
    cur=res[res.iterations==300].brier.iloc[0]; best=res.loc[res.brier.idxmin()]
    print(f"\n현재 설정 300회: {cur:.6f}")
    print(f"최적 {int(best.iterations)}회: {best.brier:.6f}  ({(cur-best.brier)/1e-5:+.1f}e-5)")
    print(f"total={time.time()-start:.1f}s")

if __name__=="__main__": main()

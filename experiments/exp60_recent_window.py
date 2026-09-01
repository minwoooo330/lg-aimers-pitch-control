# -*- coding: utf-8 -*-
"""실험 60: 최근 체제 전용 모델(최근 N시즌만 학습)을 앙상블 멤버 후보로 추가. 3-fold.
   exp11(최근 시즌 가중)과 다름: 가중이 아니라 '창(window)'을 잘라 만든 별도 멤버."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
WINDOWS=[2,3]

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig"); rows=[]
    store={w:[] for w in WINDOWS}
    for year in [2022,2023,2024]:
        va=df[df.season==year].reset_index(drop=True)
        yva=va[TARGET].to_numpy(np.int8)
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        xb0=va[cols].copy()
        for w in WINDOWS:
            tr=df[(df.season<year)&(df.season>=year-w)].reset_index(drop=True)
            ytr=tr[TARGET].to_numpy(np.int8)
            xa,xb=tr[cols].copy(),xb0.copy()
            for c in CATS:
                vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
                xa[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
                xb[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
            xa=pd.concat([xa,add_features(tr)],axis=1); xb=pd.concat([xb,add_features(va)],axis=1)
            mdl=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                categorical_features=[c in CATS for c in xa.columns],random_state=42).fit(xa,ytr)
            p=mdl.predict_proba(xb)[:,1]
            rows.append({"fold":year,"window":w,"n_train":len(tr),
                         "brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p)})
            print(rows[-1],flush=True)
            store[w].append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,
                                          TARGET:yva,"prediction":p}))
            del xa,xb,mdl,p,tr; gc.collect()
    for w in WINDOWS:
        pd.concat(store[w],ignore_index=True).to_csv(RES/f"exp60_recent{w}_oof.csv.gz",
                                                     index=False,compression="gzip")
    pd.DataFrame(rows).to_csv(RES/"exp60_recent_window.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

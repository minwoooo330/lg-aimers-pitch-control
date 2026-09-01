# -*- coding: utf-8 -*-
"""실험 62: 규제를 푼 HGB를 앙상블 '추가' 멤버로 검증 (과감한 예측 방향).

배경: 2022·2024 신뢰도 기울기가 1.08~1.13으로 현 앙상블 예측이 덜 퍼져 있다.
교체가 아니라 추가로 넣어 앙상블 예측 산포를 늘리는 것이 목적.
판정: pooled가 아니라 2024 fold 기여로 본다(2023은 퓨처스 체제 붕괴로 대표성 없음).
"""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
VARIANTS={
 "cur":  dict(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.0),
 "L1":   dict(max_iter=300,learning_rate=.06,max_leaf_nodes=63,min_samples_leaf=50,l2_regularization=0.3),
 "L2":   dict(max_iter=400,learning_rate=.06,max_leaf_nodes=127,min_samples_leaf=10,l2_regularization=0.05),
}

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig"); rows=[]
    store={v:[] for v in VARIANTS}
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
        catmask=[c in CATS for c in xa.columns]
        for name,prm in VARIANTS.items():
            tt=time.time()
            mdl=HistGradientBoostingClassifier(early_stopping=False,random_state=42,
                    categorical_features=catmask,**prm).fit(xa,ytr)
            p=mdl.predict_proba(xb)[:,1]
            rows.append({"fold":year,"variant":name,"brier":brier_score_loss(yva,p),
                         "auc":roc_auc_score(yva,p),"pred_sd":float(p.std()),
                         "sec":round(time.time()-tt,1)})
            print(rows[-1],flush=True)
            store[name].append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,
                                             TARGET:yva,"prediction":p}))
            del mdl,p; gc.collect()
        del xa,xb,tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp62_loose_hgb.csv",index=False,encoding="utf-8-sig")
    for v in VARIANTS:
        pd.concat(store[v],ignore_index=True).to_csv(RES/f"exp62_{v}_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

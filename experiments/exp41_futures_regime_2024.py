# -*- coding: utf-8 -*-
"""실험 41: 퓨처스 2019~2022 라벨 체제 변경 처리 (2024 화면)."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def encode(tr,va,cols):
    a,b=tr[cols].copy(),va[cols].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    return a,b

def fit(x,y):
    return HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
        min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
        categorical_features=[c in CATS for c in x.columns],random_state=42).fit(x,y)

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    va=df[df.season==2024]; yva=va[TARGET].to_numpy(np.int8)
    old_f=(df.game_type=="F")&(df.season<2023)
    rows=[]
    cases=[("기준(전부 사용)",df[df.season<2024],False),
           ("퓨처스 19~22 제거",df[(df.season<2024)&~old_f],False),
           ("abs_era 플래그 추가",df[df.season<2024],True)]
    for lbl,tr,flag in cases:
        tr=tr.reset_index(drop=True)
        xtr,xva=encode(tr,va,base)
        xtr=pd.concat([xtr,add_features(tr)],axis=1); xva2=pd.concat([xva,add_features(va)],axis=1)
        if flag:
            xtr["abs_era"]=((tr.game_type=="F")&(tr.season>=2023)).astype(np.int8).to_numpy()
            xva2["abs_era"]=((va.game_type=="F")).astype(np.int8).to_numpy()
        m=fit(xtr,tr[TARGET].to_numpy(np.int8)); p=m.predict_proba(xva2)[:,1]
        rows.append({"구성":lbl,"n_train":len(tr),"brier":brier_score_loss(yva,p),
                     "auc":roc_auc_score(yva,p)})
        print(rows[-1],flush=True)
        np.save(RES/f"exp41_{len(rows)}.npy",p)
        del xtr,xva2,m,p,tr; gc.collect()
    r=pd.DataFrame(rows); r["gain_e5"]=(r.brier[0]-r.brier)/1e-5
    r.to_csv(RES/"exp41_futures_regime_2024.csv",index=False,encoding="utf-8-sig")
    print("\n",r.to_string(index=False))
    print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

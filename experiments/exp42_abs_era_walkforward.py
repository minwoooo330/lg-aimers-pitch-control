# -*- coding: utf-8 -*-
"""실험 42: abs_era 플래그 3-fold 검증."""
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
    rows=[]; oof=[]
    for year in [2022,2023,2024]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        xtr,xva=encode(tr,va,base)
        xtr=pd.concat([xtr,add_features(tr)],axis=1); xva=pd.concat([xva,add_features(va)],axis=1)
        for flag in [False,True]:
            a,b=xtr.copy(),xva.copy()
            if flag:
                a["abs_era"]=((tr.game_type=="F")&(tr.season>=2023)).astype(np.int8).to_numpy()
                b["abs_era"]=((va.game_type=="F")&(va.season>=2023)).astype(np.int8).to_numpy()
            m=fit(a,ytr); p=m.predict_proba(b)[:,1]
            rows.append({"fold":year,"abs_era":flag,"brier":brier_score_loss(yva,p),
                         "auc":roc_auc_score(yva,p)})
            print(rows[-1],flush=True)
            if flag:
                oof.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,
                                         TARGET:yva,"prediction":p}))
            del a,b,m,p; gc.collect()
        del tr,va,xtr,xva; gc.collect()
    r=pd.DataFrame(rows); r.to_csv(RES/"exp42_abs_era_walkforward.csv",index=False,encoding="utf-8-sig")
    pd.concat(oof,ignore_index=True).to_csv(RES/"exp42_abs_era_oof.csv.gz",index=False,compression="gzip")
    piv=r.pivot(index="fold",columns="abs_era",values="brier")
    piv["gain_e5"]=(piv[False]-piv[True])/1e-5
    print("\n",piv.to_string())
    print(f"평균 개선 {piv.gain_e5.mean():+.2f}e-5 | 세 해 모두 개선? {bool((piv.gain_e5>0).all())}")
    print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

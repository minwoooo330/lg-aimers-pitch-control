# -*- coding: utf-8 -*-
"""실험 56: D-1(팀ID 범주형) + D-2(park_id 신설) 3-fold 검증."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
BASE_CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def prep(tr,va,cats,use_park):
    a,b=tr.copy(),va.copy()
    if use_park:
        a["park_id"]=np.where(a.top_bottom=="T",a.pitcher_team_id,a.batter_team_id)
        b["park_id"]=np.where(b.top_bottom=="T",b.pitcher_team_id,b.batter_team_id)
        cats=cats+["park_id"]
    cols=[c for c in a.columns if c not in (ID,TARGET)]
    xa,xb=a[cols].copy(),b[cols].copy()
    for c in cats:
        vals=sorted(a[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
        xa[c]=a[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xb[c]=b[c].astype(str).map(m).fillna(-1).astype(np.int16)
    xa=pd.concat([xa,add_features(a)],axis=1); xb=pd.concat([xb,add_features(b)],axis=1)
    return xa,xb,[c in cats for c in xa.columns]

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig"); rows=[]
    CFG=[("기준",BASE_CATS,False),
         ("D1 팀범주형",BASE_CATS+["pitcher_team_id","batter_team_id"],False),
         ("D2 park만",BASE_CATS,True),
         ("D1+D2",BASE_CATS+["pitcher_team_id","batter_team_id"],True)]
    for year in [2022,2023,2024]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        for lbl,cats,park in CFG:
            xa,xb,mask=prep(tr,va,cats,park)
            m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                categorical_features=mask,random_state=42).fit(xa,ytr)
            p=m.predict_proba(xb)[:,1]
            rows.append({"fold":year,"cfg":lbl,"brier":brier_score_loss(yva,p),
                         "auc":roc_auc_score(yva,p)})
            print(rows[-1],flush=True)
            if lbl!="기준":
                np.save(RES/f"exp56_{lbl.replace(' ','_')}_{year}.npy",p)
            del xa,xb,m,p; gc.collect()
        del tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp56_team_park.csv",index=False,encoding="utf-8-sig")
    r=pd.DataFrame(rows); piv=r.pivot(index="cfg",columns="fold",values="brier")
    base=piv.loc["기준"]
    for cfg in piv.index:
        if cfg=="기준": continue
        g=[(base[f]-piv.loc[cfg,f])/1e-5 for f in [2022,2023,2024]]
        print(f"{cfg}: 2022 {g[0]:+.2f} / 2023 {g[1]:+.2f} / 2024 {g[2]:+.2f} e-5")
    print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

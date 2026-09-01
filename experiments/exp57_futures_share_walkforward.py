# -*- coding: utf-8 -*-
"""실험 57: D-4 투수 퓨처스 이력 비중 (시즌 as-of) 3-fold 검증."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
BASE={2022:0.24352431071531802,2023:0.2534316109162257,2024:0.24805922339357517}

def futures_tables(df):
    g=df.groupby(["pitcher_id","season"]).agg(n=("row_id","size"),
        f=("game_type",lambda s:(s=="F").sum())).reset_index()
    g["f_pre"]=np.where(g.season<2023,g.f,0)
    g=g.sort_values(["pitcher_id","season"])
    for c in ["n","f","f_pre"]:
        g["c_"+c]=g.groupby("pitcher_id")[c].cumsum()-g[c]   # 현재 시즌 제외 누적(as-of)
    g["fut_share"]=g.c_f/g.c_n.replace(0,np.nan)
    g["fut_share_pre23"]=g.c_f_pre/g.c_n.replace(0,np.nan)
    return g[["pitcher_id","season","fut_share","fut_share_pre23"]]

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig")
    tab=futures_tables(df)
    df=df.merge(tab,on=["pitcher_id","season"],how="left")
    rows=[]
    for year in [2022,2023,2024]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        cols=[c for c in df.columns if c not in (ID,TARGET,"fut_share","fut_share_pre23")]
        xa,xb=tr[cols].copy(),va[cols].copy()
        for c in CATS:
            vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
            xa[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
            xb[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xa=pd.concat([xa,add_features(tr)],axis=1); xb=pd.concat([xb,add_features(va)],axis=1)
        xa["fut_share"]=tr.fut_share.to_numpy(np.float32); xb["fut_share"]=va.fut_share.to_numpy(np.float32)
        xa["fut_share_pre23"]=tr.fut_share_pre23.to_numpy(np.float32); xb["fut_share_pre23"]=va.fut_share_pre23.to_numpy(np.float32)
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
            min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
            categorical_features=[c in CATS for c in xa.columns],random_state=42).fit(xa,ytr)
        p=m.predict_proba(xb)[:,1]
        br=brier_score_loss(yva,p)
        rows.append({"fold":year,"brier":br,"auc":roc_auc_score(yva,p),
                     "gain_e5":(BASE[year]-br)/1e-5})
        print(rows[-1],flush=True)
        np.save(RES/f"exp57_fut_{year}.npy",p)
        del xa,xb,m,p,tr,va; gc.collect()
    pd.DataFrame(rows).to_csv(RES/"exp57_futures_share.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

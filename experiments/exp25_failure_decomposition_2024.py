# -*- coding: utf-8 -*-
"""실험 25: 실패 유형 분해를 반영한 파생피처 2024 gate."""
from pathlib import Path
import time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from features import add_features
HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def decomposition(d):
    """행 자체의 공식 asof 값만으로 계산하는 실패유형 분해 피처."""
    f=pd.DataFrame(index=d.index)
    s=d["asof_pitcher_success_rate"]; r=d["asof_pitcher_reverse_rate"]; m=d["asof_pitcher_middle_rate"]
    b=d["asof_pitcher_ball_rate"]; k=d["asof_pitcher_strike_rate"]
    fail=(1.0-s)
    f["fd_unnamed_minus_overlap"]=1.0-s-r-m      # 미분류실패 - 중복실패
    f["fd_inplay_rate"]=1.0-b-k                  # 판정축의 인플레이 비율
    f["fd_rev_share_true"]=r/fail.clip(lower=1e-4)
    f["fd_mid_share_true"]=m/fail.clip(lower=1e-4)
    f["fd_named_share"]=(r+m)/fail.clip(lower=1e-4)
    f["fd_rev_minus_mid"]=r-m
    f["fd_fail_x_logn"]=fail*np.log1p(d["asof_pitcher_n"].fillna(0))
    return f.replace([np.inf,-np.inf],np.nan).astype(np.float32)

def encode(tr,va,base):
    a=tr[base].copy(); b=va[base].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    return a,b

def main():
    start=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    tr=df[df.season<2024]; va=df[df.season==2024]
    xtr,xva=encode(tr,va,base)
    xtr=pd.concat([xtr,add_features(tr),decomposition(tr)],axis=1)
    xva=pd.concat([xva,add_features(va),decomposition(va)],axis=1)
    ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    t=time.time()
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
        min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
        categorical_features=[c in CATS for c in xtr.columns],random_state=42)
    m.fit(xtr,ytr); p=m.predict_proba(xva)[:,1]
    base_brier=0.24811007644535327
    row={"model":"domain43_plus_failure_decomp","fold":2024,"n_features":xtr.shape[1],
         "brier":brier_score_loss(yva,p),"logloss":log_loss(yva,p,labels=[0,1]),
         "roc_auc":roc_auc_score(yva,p),"baseline_brier":base_brier,
         "gain":base_brier-brier_score_loss(yva,p),"seconds":time.time()-t}
    RES.mkdir(exist_ok=True); pd.DataFrame([row]).to_csv(RES/"exp25_failure_decomposition_2024.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({ID:va[ID].to_numpy(),"season":2024,TARGET:yva,"prediction":p}).to_csv(RES/"exp25_failure_decomp_2024_oof.csv.gz",index=False,compression="gzip")
    print(row); print("PASS" if row["gain"]>=1e-5 else "STOP", f"total={time.time()-start:.1f}s")

if __name__=="__main__": main()

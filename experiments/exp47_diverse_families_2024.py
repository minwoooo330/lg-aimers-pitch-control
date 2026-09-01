# -*- coding: utf-8 -*-
"""실험 47: 미시도 모델 계열 3종 2024 화면 (ExtraTrees / 로지스틱+구간 / FM)."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    tr=df[df.season<2024].reset_index(drop=True); va=df[df.season==2024].reset_index(drop=True)
    ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    a,b=tr[base].copy(),va[base].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
    a=pd.concat([a,add_features(tr)],axis=1).astype(np.float32)
    b=pd.concat([b,add_features(va)],axis=1).astype(np.float32)
    med=a.median(); a=a.fillna(med); b=b.fillna(med)
    rows=[]

    t=time.time()
    et=ExtraTreesClassifier(n_estimators=300,max_depth=14,min_samples_leaf=150,
        max_features="sqrt",n_jobs=-1,random_state=42).fit(a,ytr)
    p=et.predict_proba(b)[:,1]
    rows.append({"model":"ExtraTrees","brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p),"sec":time.time()-t})
    np.save(RES/"exp47_et_2024.npy",p); print(rows[-1],flush=True); del et; gc.collect()

    t=time.time()
    kb=KBinsDiscretizer(n_bins=16,encode="onehot-dense",strategy="quantile",subsample=200000,random_state=42)
    A=kb.fit_transform(a); B=kb.transform(b)
    lr=LogisticRegression(C=0.1,max_iter=300,solver="lbfgs",n_jobs=-1).fit(A,ytr)
    p=lr.predict_proba(B)[:,1]
    rows.append({"model":"Logistic+구간","brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p),"sec":time.time()-t})
    np.save(RES/"exp47_lr_2024.npy",p); print(rows[-1],flush=True); del A,B,lr,kb; gc.collect()

    r=pd.DataFrame(rows); r.to_csv(RES/"exp47_diverse_2024.csv",index=False,encoding="utf-8-sig")
    print("\n기준 HGB 2024 = 0.248059"); print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

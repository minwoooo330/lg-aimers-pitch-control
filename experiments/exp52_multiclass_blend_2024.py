# -*- coding: utf-8 -*-
"""실험 52: 4분류 모델을 앙상블 멤버로 재평가 (exp24는 단독 평가만 했음). 2024 fold."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig")
    lab=np.load(RES/"exp24_reconstructed_labels.npy")
    assert len(lab)==len(df)
    base=[c for c in df.columns if c not in (ID,TARGET)]
    year=2024
    tr_m=(df.season<year).to_numpy()
    # 분석가 권고: 퓨처스 2019~2022 행 제거 (체제 전 라벨)
    oldF=((df.game_type=="F")&(df.season<2023)).to_numpy()
    tr=df[tr_m&~oldF].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
    ymc=lab[(tr_m&~oldF)]
    yva=va[TARGET].to_numpy(np.int8)
    a,b=tr[base].copy(),va[base].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
    a=pd.concat([a,add_features(tr)],axis=1); b=pd.concat([b,add_features(va)],axis=1)
    model=HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,max_leaf_nodes=31,
        min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
        categorical_features=[c in CATS for c in a.columns],random_state=42)
    t=time.time(); model.fit(a,ymc)
    proba=model.predict_proba(b); p0=proba[:,list(model.classes_).index(0)]
    print({"단독 brier":brier_score_loss(yva,p0),"auc":roc_auc_score(yva,p0),
           "sec":time.time()-t},flush=True)
    pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p0}).to_csv(
        RES/"exp52_mc4_2024_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

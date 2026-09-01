# -*- coding: utf-8 -*-
"""실험 46: 카운트x최근체제 상호작용 (2024 fold 화면)."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def extra(d,recent_from):
    f=pd.DataFrame(index=d.index)
    cid=(d.balls_before*3+d.strikes_before).astype(np.int16)
    recent=(d.season>=recent_from)
    # (a) 카운트 x 체제 명시 (범주): 최근 체제의 카운트만 구분, 과거는 단일 버킷
    f["count_recent"]=np.where(recent,cid,12).astype(np.int16)
    # (c) 카운트 축 한정 최근 정보: 최근 체제에서의 해당 카운트 성공률 (train에서 고정)
    return f

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    year=2024
    tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
    ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    a,b=tr[base].copy(),va[base].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
    a=pd.concat([a,add_features(tr)],axis=1); b=pd.concat([b,add_features(va)],axis=1)
    # 최근 체제 카운트별 성공률 표 (train 내 최근 2개 시즌으로 고정, 자기 시즌 미포함)
    rec=tr[tr.season>=year-2]
    tab=rec.groupby(rec.balls_before*3+rec.strikes_before)[TARGET].mean()
    old=tr.groupby(tr.balls_before*3+tr.strikes_before)[TARGET].mean()
    rows=[]
    for lbl,mode in [("기준",0),("count_recent 범주",1),("count_shift 수치",2),("둘다",3)]:
        x1,x2=a.copy(),b.copy()
        if mode in (1,3):
            x1["count_recent"]=np.where(tr.season>=year-2,(tr.balls_before*3+tr.strikes_before),12).astype(np.int16)
            x2["count_recent"]=(va.balls_before*3+va.strikes_before).astype(np.int16)
        if mode in (2,3):
            cid_tr=(tr.balls_before*3+tr.strikes_before); cid_va=(va.balls_before*3+va.strikes_before)
            x1["count_shift"]=(cid_tr.map(tab)-cid_tr.map(old)).astype(np.float32)
            x2["count_shift"]=(cid_va.map(tab)-cid_va.map(old)).astype(np.float32)
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
            min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
            categorical_features=[c in CATS for c in x1.columns],random_state=42).fit(x1,ytr)
        p=m.predict_proba(x2)[:,1]
        rows.append({"구성":lbl,"brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p)})
        print(rows[-1],flush=True)
        del m,p,x1,x2; gc.collect()
    r=pd.DataFrame(rows); r["gain_e5"]=(r.brier[0]-r.brier)/1e-5
    r.to_csv(RES/"exp46_count_era_2024.csv",index=False,encoding="utf-8-sig")
    print(r.to_string(index=False)); print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

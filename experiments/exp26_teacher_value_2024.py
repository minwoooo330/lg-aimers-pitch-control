# -*- coding: utf-8 -*-
"""실험 26: 실제 Trackman 측정값(특권정보)이 얼마나 더 맞히는지 2024 사전확인."""
from pathlib import Path
import time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from features import add_features
from trackman_features import match_exact_games

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
PRIV=["rel_speed","spin_rate","induced_vert_break","horz_break","extension",
      "rel_height","rel_side","zone_speed"]
TM_COLS=["trackman_game_id","pitch_no","inning","top_bottom","balls_before","strikes_before",
         "outs_before","pitcher_trackman_id","pitcher_hand","batter_hand","pitch_type_group"]+PRIV

def encode(tr,va,cols):
    a,b=tr[cols].copy(),va[cols].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    return a,b

def fit(x,y,mask):
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
        min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
        categorical_features=mask,random_state=42)
    return m.fit(x,y)

def main():
    start=time.time()
    df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
    mg,ts,matches=match_exact_games(df,tm)
    mi=mg.groupby("_game_idx",sort=False).indices; ti=ts.groupby("trackman_game_id",sort=False).indices
    keep_tm=PRIV+["pitch_type_group"]
    parts=[]
    for row in matches.itertuples(index=False):
        a=mg.iloc[mi[row.main_game_idx]].reset_index(drop=True)
        b=ts.iloc[ti[row.trackman_game_id]][keep_tm].reset_index(drop=True)
        if len(a)==len(b): parts.append(pd.concat([a,b],axis=1))
    d=pd.concat(parts,ignore_index=True)
    print("정렬된 행:",len(d))

    base=[c for c in df.columns if c not in (ID,TARGET)]
    tr=d[d.season<2024]; va=d[d.season==2024]
    ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    xtr,xva=encode(tr,va,base)
    xtr=pd.concat([xtr,add_features(tr)],axis=1); xva=pd.concat([xva,add_features(va)],axis=1)

    rows=[]
    m1=fit(xtr,ytr,[c in CATS for c in xtr.columns]); p1=m1.predict_proba(xva)[:,1]
    rows.append({"model":"student_only_prepitch","brier":brier_score_loss(yva,p1),
                 "auc":roc_auc_score(yva,p1),"logloss":log_loss(yva,p1,labels=[0,1])})
    print(rows[-1])

    # 특권정보 추가: 그 투구의 실제 측정값 + 실제 구종군
    ptr=xtr.copy(); pva=xva.copy()
    for c in PRIV:
        ptr[c]=tr[c].to_numpy(float); pva[c]=va[c].to_numpy(float)
    groups=sorted(tr.pitch_type_group.dropna().astype(str).unique())
    gm={v:i for i,v in enumerate(groups)}
    ptr["pitch_type_group"]=tr.pitch_type_group.astype(str).map(gm).fillna(-1).astype(np.int16)
    pva["pitch_type_group"]=va.pitch_type_group.astype(str).map(gm).fillna(-1).astype(np.int16)
    mask=[c in CATS+["pitch_type_group"] for c in ptr.columns]
    m2=fit(ptr,ytr,mask); p2=m2.predict_proba(pva)[:,1]
    rows.append({"model":"teacher_with_privileged","brier":brier_score_loss(yva,p2),
                 "auc":roc_auc_score(yva,p2),"logloss":log_loss(yva,p2,labels=[0,1])})
    print(rows[-1])

    res=pd.DataFrame(rows)
    gap=float(res.iloc[0].brier-res.iloc[1].brier)
    res["teacher_gain"]=[0.0,gap]
    RES.mkdir(exist_ok=True)
    res.to_csv(RES/"exp26_teacher_value_2024.csv",index=False,encoding="utf-8-sig")
    print(f"\n교사 우위 Brier {gap:.6f}, AUC {res.iloc[1].auc-res.iloc[0].auc:+.4f}")
    print("증류 가치 있음" if gap>=0.002 else "증류 가치 낮음", f"| total={time.time()-start:.1f}s")

if __name__=="__main__": main()

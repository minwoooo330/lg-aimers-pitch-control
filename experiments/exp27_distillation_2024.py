# -*- coding: utf-8 -*-
"""실험 27: 특권정보 교사의 soft target으로 학생 모델을 학습하는 2024 gate."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
ALPHAS=[0.0,0.3,0.7]
BASELINE=0.24811007644535327

def main():
    start=time.time()
    df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
    mg,ts,matches=match_exact_games(df,tm)
    mi=mg.groupby("_game_idx",sort=False).indices; ti=ts.groupby("trackman_game_id",sort=False).indices
    keep=[ID,"season"]+PRIV+["pitch_type_group"]
    parts=[]
    for row in matches.itertuples(index=False):
        a=mg.iloc[mi[row.main_game_idx]].reset_index(drop=True)
        b=ts.iloc[ti[row.trackman_game_id]][PRIV+["pitch_type_group"]].reset_index(drop=True)
        if len(a)==len(b):
            parts.append(pd.concat([a[[ID,"season",TARGET]],b],axis=1))
    aligned=pd.concat(parts,ignore_index=True)
    del parts, mg, ts; gc.collect()
    print("정렬된 행:",len(aligned))

    base=[c for c in df.columns if c not in (ID,TARGET)]
    # 전체 학습행 인코딩 (범주 매핑은 2024 제외 구간에서만)
    tr=df[df.season<2024]; va=df[df.season==2024]
    xtr=tr[base].copy(); xva=va[base].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        xtr[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        xva[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    xtr=pd.concat([xtr,add_features(tr)],axis=1); xva=pd.concat([xva,add_features(va)],axis=1)
    ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    cat_mask=[c in CATS for c in xtr.columns]

    # 교사용: 정렬행만, 학습구간 내에서 시즌 단위 cross-fit (자기 시즌 라벨 안 봄)
    al=aligned[aligned.season<2024].reset_index(drop=True)
    pos=pd.Series(np.arange(len(tr)),index=tr[ID].to_numpy())
    al_pos=pos.reindex(al[ID].to_numpy()).to_numpy()
    groups=sorted(al.pitch_type_group.dropna().astype(str).unique()); gm={v:i for i,v in enumerate(groups)}

    xt=xtr.iloc[al_pos].reset_index(drop=True)
    for c in PRIV: xt[c]=al[c].to_numpy(float)
    xt["pitch_type_group"]=al.pitch_type_group.astype(str).map(gm).fillna(-1).astype(np.int16)
    yt=al[TARGET].to_numpy(np.int8)
    tmask=[c in CATS+["pitch_type_group"] for c in xt.columns]

    soft=np.full(len(al),np.nan)
    for s in sorted(al.season.unique()):
        fit_idx=(al.season!=s).to_numpy(); app_idx=~fit_idx
        t0=time.time()
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
            min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
            categorical_features=tmask,random_state=42)
        m.fit(xt[fit_idx],yt[fit_idx])
        soft[app_idx]=m.predict_proba(xt[app_idx])[:,1]
        print(f"teacher {s}: {app_idx.sum():,}행 soft target, {time.time()-t0:.0f}s")
        del m; gc.collect()
    del xt; gc.collect()

    print("교사 soft target 상관(실제라벨):",round(float(np.corrcoef(soft,yt)[0,1]),4))

    rows=[]
    for a in ALPHAS:
        target=ytr.astype(float).copy()
        target[al_pos]=(1-a)*yt + a*soft
        t0=time.time()
        st=HistGradientBoostingRegressor(loss="squared_error",max_iter=200,learning_rate=.06,
            max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
            categorical_features=cat_mask,random_state=42)
        st.fit(xtr,target)
        p=np.clip(st.predict(xva),1e-6,1-1e-6)
        row={"alpha":a,"brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p),
             "logloss":log_loss(yva,p,labels=[0,1]),"pred_mean":float(p.mean()),
             "gain_vs_baseline":BASELINE-brier_score_loss(yva,p),"seconds":time.time()-t0}
        rows.append(row); print(row)
        del st,p,target; gc.collect()

    res=pd.DataFrame(rows); RES.mkdir(exist_ok=True)
    res.to_csv(RES/"exp27_distillation_2024.csv",index=False,encoding="utf-8-sig")
    best=res.loc[res.brier.idxmin()]
    print("\n최고:",dict(best))
    print("PASS" if best.gain_vs_baseline>=1e-5 else "STOP", f"total={time.time()-start:.1f}s")

if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""실험 28: 평가지표와 동일한 제곱오차 손실로 HGB 3종 재학습 시간검증."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from features import add_features
from trackman_features import (match_exact_games, build_pitcher_mapping,
                               build_trackman_profile, build_trackman_profile_by_league)

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
TM_COLS=["trackman_id","season","trackman_game_id","pitch_no","inning","top_bottom","balls_before",
         "strikes_before","outs_before","pitcher_trackman_id","pitcher_hand","pitcher_team",
         "pitch_type_group","rel_speed","spin_rate","induced_vert_break","horz_break","extension",
         "rel_height","rel_side","zone_speed","batter_hand"]
MATCH=["season","inning","top_bottom","balls_before","strikes_before","outs_before",
       "pitcher_id","pitcher_hand","batter_hand"]

def encode(tr,va,cols):
    a,b=tr[cols].copy(),va[cols].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    return a,b

def attach_overall(frame,profile,cols):
    b=frame[["pitcher_id"]].join(profile[cols],on="pitcher_id").drop(columns="pitcher_id")
    b["tm_profile_available"]=b.notna().any(axis=1).astype(np.int8)
    return b.astype(np.float32)

def attach_league(frame,profile,cols):
    b=(frame[["pitcher_id","game_type"]].reset_index()
       .merge(profile[cols].reset_index(),on=["pitcher_id","game_type"],how="left",sort=False)
       .set_index("index").reindex(frame.index).drop(columns=["pitcher_id","game_type"]))
    b["tm_league_profile_available"]=b.notna().any(axis=1).astype(np.int8)
    return b.astype(np.float32)

def fit_reg(x,y):
    m=HistGradientBoostingRegressor(loss="squared_error",max_iter=200,learning_rate=.06,
        max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
        categorical_features=[c in CATS for c in x.columns],random_state=42)
    return m.fit(x,y)

def main():
    start=time.time()
    df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
    mg,ts,matches=match_exact_games(df[MATCH],tm)
    base=[c for c in df.columns if c not in (ID,TARGET)]
    names=["reg_domain","reg_tm_mean","reg_league_role"]
    oof={k:[] for k in names}; rows=[]

    for year in [2022,2023,2024]:
        mapping=build_pitcher_mapping(mg,ts,matches,year)
        prof=build_trackman_profile(tm,mapping,year)
        prof_lg=build_trackman_profile_by_league(tm,mapping,year)
        mean_cols=[c for c in prof.columns if "_mean_" in c or "pitch_group_n" in c
                   or "pitch_share" in c or c=="tm_total_n"]
        role_cols=[c for c in prof_lg.columns if any(k in c for k in
                   ["appearances","starter_share","long_relief_share","short_relief_share",
                    "starter_like_share","avg_pitches","avg_innings"])]

        tr=df[df.season<year]; va=df[df.season==year]
        ytr=tr[TARGET].to_numpy(float); yva=va[TARGET].to_numpy(np.int8)
        xtr,xva=encode(tr,va,base)
        xtr=pd.concat([xtr,add_features(tr)],axis=1); xva=pd.concat([xva,add_features(va)],axis=1)

        for name in names:
            if name=="reg_domain":
                a,b=xtr,xva
            elif name=="reg_tm_mean":
                a=pd.concat([xtr,attach_overall(tr,prof,mean_cols)],axis=1)
                b=pd.concat([xva,attach_overall(va,prof,mean_cols)],axis=1)
            else:
                a=pd.concat([xtr,attach_league(tr,prof_lg,role_cols)],axis=1)
                b=pd.concat([xva,attach_league(va,prof_lg,role_cols)],axis=1)
            t0=time.time()
            m=fit_reg(a,ytr); p=np.clip(m.predict(b),1e-6,1-1e-6)
            rows.append({"model":name,"fold":year,"brier":brier_score_loss(yva,p),
                         "logloss":log_loss(yva,p,labels=[0,1]),"roc_auc":roc_auc_score(yva,p),
                         "pred_mean":float(p.mean()),"seconds":time.time()-t0})
            oof[name].append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,
                                           TARGET:yva,"prediction":p}))
            print(rows[-1])
            if name!="reg_domain": del a,b
            del m,p; gc.collect()
        del tr,va,xtr,xva,prof,prof_lg,mapping; gc.collect()

    res=pd.DataFrame(rows)
    summ=res.groupby("model",as_index=False).agg(brier=("brier","mean"),logloss=("logloss","mean"),
                                                 roc_auc=("roc_auc","mean"),seconds=("seconds","sum"))
    summ["fold"]="mean_2022_2024"
    for c in res.columns:
        if c not in summ: summ[c]=np.nan
    res=pd.concat([res,summ[res.columns]],ignore_index=True)
    RES.mkdir(exist_ok=True)
    res.to_csv(RES/"exp28_squared_loss_walkforward.csv",index=False,encoding="utf-8-sig")
    for k,v in oof.items():
        pd.concat(v,ignore_index=True).to_csv(RES/f"exp28_{k}_oof.csv.gz",index=False,compression="gzip")
    print(res[res.fold=="mean_2022_2024"].to_string(index=False))
    print(f"total={time.time()-start:.1f}s")

if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""실험 08: Trackman 피처 그룹별 HGB 제거 실험."""
from pathlib import Path
import gc,time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from features import add_features
from trackman_features import match_exact_games,build_pitcher_mapping,build_trackman_profile

HERE=Path(__file__).resolve().parent;DATA=HERE/"data";RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
FOLDS=[2022,2023,2024]
TM_COLS=["trackman_id","season","trackman_game_id","pitch_no","inning","top_bottom",
 "balls_before","strikes_before","outs_before","pitcher_trackman_id","pitcher_hand",
 "pitcher_team","pitch_type_group","rel_speed","spin_rate","induced_vert_break",
 "horz_break","extension","rel_height","rel_side","zone_speed","batter_hand"]
MATCH=["season","inning","top_bottom","balls_before","strikes_before","outs_before",
       "pitcher_id","pitcher_hand","batter_hand"]

def enc(tr,va,cols):
 a=tr[cols].copy();b=va[cols].copy()
 for c in CAT:
  vals=sorted(tr[c].dropna().astype(str).unique());mp={v:i for i,v in enumerate(vals)}
  a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
  b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
 return a,b

def block(frame,profile,cols):
 b=frame[["pitcher_id"]].join(profile[cols],on="pitcher_id").drop(columns="pitcher_id")
 b["tm_profile_available"]=(b.notna().any(axis=1)).astype(np.int8)
 return b.astype(np.float32)

def groups(profile):
 cols=list(profile.columns)
 role=[c for c in cols if c.startswith("tm_") and any(k in c for k in
       ["appearances","starter_share","long_relief_share","short_relief_share","avg_pitches","avg_innings"])]
 support=[c for c in cols if ("pitch_group_n" in c or "pitch_share" in c or c=="tm_total_n")]
 means=[c for c in cols if "_mean_" in c]
 stds=[c for c in cols if "_std_" in c]
 aligned=[c for c in cols if any(k in c for k in
          ["rel_side_aligned","horz_break_aligned","rel_height"])]
 return {"tm_role":role,
         "tm_pitch_means":support+means,
         "tm_pitch_stds":support+stds,
         "tm_expert_core":list(dict.fromkeys(support+role+aligned))}

def main():
 start=time.time();df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
 tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
 mg,ts,matches=match_exact_games(df[MATCH],tm);base=[c for c in df.columns if c not in(ID,TARGET)]
 rows=[];oofs={}
 for year in FOLDS:
  mapping=build_pitcher_mapping(mg,ts,matches,year);profile=build_trackman_profile(tm,mapping,year)
  tr=df[df.season<year];va=df[df.season==year];ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8)
  xbtr,xbva=enc(tr,va,base);base_tr=pd.concat([xbtr,add_features(tr)],axis=1);base_va=pd.concat([xbva,add_features(va)],axis=1)
  for name,cols in groups(profile).items():
   fit=time.time();xtr=pd.concat([base_tr,block(tr,profile,cols)],axis=1);xva=pd.concat([base_va,block(va,profile,cols)],axis=1)
   mask=[c in CAT for c in xtr.columns]
   m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
      min_samples_leaf=200,l2_regularization=1.,early_stopping=False,categorical_features=mask,random_state=42)
   m.fit(xtr,ytr);p=m.predict_proba(xva)[:,1];sec=time.time()-fit
   row={"model":"hgb","features":name,"fold":str(year),"n_features":xtr.shape[1],
        "mapped_pitchers":len(mapping),"valid_coverage":va.pitcher_id.isin(mapping.pitcher_id).mean(),
        "brier":brier_score_loss(yva,p),"logloss":log_loss(yva,p,labels=[0,1]),
        "roc_auc":roc_auc_score(yva,p),"seconds":sec};rows.append(row)
   oofs.setdefault(name,[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}))
   print(f"{year} {name}: Brier={row['brier']:.6f} AUC={row['roc_auc']:.6f} {sec:.1f}초")
   del xtr,xva,m,p;gc.collect()
  del mapping,profile,tr,va,xbtr,xbva,base_tr,base_va,ytr,yva;gc.collect()
 result=pd.DataFrame(rows);summary=(result.groupby("features",as_index=False).agg(
  brier=("brier","mean"),logloss=("logloss","mean"),roc_auc=("roc_auc","mean"),seconds=("seconds","sum")))
 summary["model"]="hgb";summary["fold"]="mean_2022_2024"
 for c in result.columns:
  if c not in summary:summary[c]=np.nan
 result=pd.concat([result,summary[result.columns]],ignore_index=True);RES.mkdir(exist_ok=True)
 result.to_csv(RES/"exp08_hgb_trackman_groups.csv",index=False,encoding="utf-8-sig")
 for name,parts in oofs.items():pd.concat(parts,ignore_index=True).to_csv(RES/f"exp08_{name}_oof.csv.gz",index=False,compression="gzip")
 print(result[result.fold.eq("mean_2022_2024")][["features","brier","logloss","roc_auc","seconds"]].to_string(index=False))
 print(f"총 {time.time()-start:.1f}초")
if __name__=="__main__":main()

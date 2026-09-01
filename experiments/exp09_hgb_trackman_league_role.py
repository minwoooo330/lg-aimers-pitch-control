# -*- coding: utf-8 -*-
"""실험 09: 1군/퓨처스 분리 Trackman 평균과 선발형 역할 피처."""
from pathlib import Path
import gc,time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from features import add_features
from trackman_features import match_exact_games,build_pitcher_mapping,build_trackman_profile_by_league

HERE=Path(__file__).resolve().parent;DATA=HERE/"data";RES=HERE/"results"
ID,TARGET="row_id","control_success";CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
FOLDS=[2022,2023,2024]
TM_COLS=["trackman_id","season","trackman_game_id","pitch_no","inning","top_bottom","balls_before","strikes_before","outs_before","pitcher_trackman_id","pitcher_hand","pitcher_team","pitch_type_group","rel_speed","spin_rate","induced_vert_break","horz_break","extension","rel_height","rel_side","zone_speed","batter_hand"]
MATCH=["season","inning","top_bottom","balls_before","strikes_before","outs_before","pitcher_id","pitcher_hand","batter_hand"]

def encode(tr,va,cols):
 a=tr[cols].copy();b=va[cols].copy()
 for c in CAT:
  vals=sorted(tr[c].dropna().astype(str).unique());mp={v:i for i,v in enumerate(vals)}
  a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16);b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
 return a,b

def attach(frame,profile,cols):
 tmp=frame[["pitcher_id","game_type"]].reset_index().merge(profile[cols].reset_index(),on=["pitcher_id","game_type"],how="left",sort=False).set_index("index")
 tmp=tmp.reindex(frame.index).drop(columns=["pitcher_id","game_type"]);tmp["tm_league_profile_available"]=(tmp.notna().any(axis=1)).astype(np.int8)
 return tmp.astype(np.float32)

def colgroups(profile):
 cols=list(profile.columns);role=[c for c in cols if any(k in c for k in ["appearances","starter_share","long_relief_share","short_relief_share","starter_like_share","avg_pitches","avg_innings"])]
 pitch=[c for c in cols if c not in role]
 return {"tm_league_pitch_means":pitch,"tm_league_role":role,"tm_league_combined":pitch+role}

def main():
 start=time.time();df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig");tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
 mg,ts,matches=match_exact_games(df[MATCH],tm);base=[c for c in df.columns if c not in(ID,TARGET)];rows=[];oofs={}
 for year in FOLDS:
  mapping=build_pitcher_mapping(mg,ts,matches,year);profile=build_trackman_profile_by_league(tm,mapping,year)
  tr=df[df.season<year];va=df[df.season==year];ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8)
  xbtr,xbva=encode(tr,va,base);base_tr=pd.concat([xbtr,add_features(tr)],axis=1);base_va=pd.concat([xbva,add_features(va)],axis=1)
  for name,cols in colgroups(profile).items():
   begin=time.time();xtr=pd.concat([base_tr,attach(tr,profile,cols)],axis=1);xva=pd.concat([base_va,attach(va,profile,cols)],axis=1)
   m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.,early_stopping=False,categorical_features=[c in CAT for c in xtr.columns],random_state=42)
   m.fit(xtr,ytr);pred=m.predict_proba(xva)[:,1];sec=time.time()-begin
   row={"model":"hgb","features":name,"fold":str(year),"n_features":xtr.shape[1],"brier":brier_score_loss(yva,pred),"logloss":log_loss(yva,pred,labels=[0,1]),"roc_auc":roc_auc_score(yva,pred),"seconds":sec,"valid_profile_coverage":xva["tm_league_profile_available"].mean()};rows.append(row)
   oofs.setdefault(name,[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":pred}))
   print(f"{year} {name}: Brier={row['brier']:.6f}, AUC={row['roc_auc']:.6f}, coverage={row['valid_profile_coverage']:.1%}, {sec:.1f}초")
   del xtr,xva,m,pred;gc.collect()
  del mapping,profile,tr,va,xbtr,xbva,base_tr,base_va,ytr,yva;gc.collect()
 result=pd.DataFrame(rows);summary=result.groupby("features",as_index=False).agg(brier=("brier","mean"),logloss=("logloss","mean"),roc_auc=("roc_auc","mean"),seconds=("seconds","sum"),valid_profile_coverage=("valid_profile_coverage","mean"));summary["model"]="hgb";summary["fold"]="mean_2022_2024"
 for c in result.columns:
  if c not in summary:summary[c]=np.nan
 result=pd.concat([result,summary[result.columns]],ignore_index=True);RES.mkdir(exist_ok=True);result.to_csv(RES/"exp09_hgb_trackman_league_role.csv",index=False,encoding="utf-8-sig")
 for name,parts in oofs.items():pd.concat(parts,ignore_index=True).to_csv(RES/f"exp09_{name}_oof.csv.gz",index=False,compression="gzip")
 print(result[result.fold.eq("mean_2022_2024")][["features","brier","logloss","roc_auc","valid_profile_coverage","seconds"]].to_string(index=False));print(f"총 {time.time()-start:.1f}초")
if __name__=="__main__":main()

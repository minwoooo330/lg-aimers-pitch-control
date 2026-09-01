# -*- coding: utf-8 -*-
"""seoyeon_v6 비중복 교차 피처의 3-fold 시간검증."""
from pathlib import Path
import time
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from features import add_features
HERE=Path(__file__).resolve().parent;DATA=HERE/"data"/"train.csv";RES=HERE/"results";ID,TARGET="row_id","control_success";CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
def encode(tr,va,base):
 a=tr[base].copy();b=va[base].copy()
 for c in CATS:
  vals=sorted(tr[c].dropna().astype(str).unique());mp={v:i for i,v in enumerate(vals)};a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16);b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
 return a,b
def extra(d):
 f=pd.DataFrame(index=d.index);f["sy_pitcher_x_balls"]=d.asof_pitcher_success_rate*d.balls_before;f["sy_pitcher_x_strikes"]=d.asof_pitcher_success_rate*d.strikes_before;f["sy_pressure"]=d.num_runners_on*d.outs_before;f["sy_li_pitcher"]=d.li*d.asof_pitcher_success_rate;f["sy_reverse_x_base"]=d.asof_pitcher_reverse_rate*d.num_runners_on;return f.astype(np.float32)
def main():
 df=pd.read_csv(DATA,encoding="utf-8-sig");base=[c for c in df.columns if c not in(ID,TARGET)];rows=[];parts=[]
 for year in [2022,2023,2024]:
  tr=df[df.season<year];va=df[df.season==year];a,b=encode(tr,va,base);a=pd.concat([a,add_features(tr),extra(tr)],axis=1);b=pd.concat([b,add_features(va),extra(va)],axis=1);ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8);m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.,early_stopping=False,categorical_features=[c in CATS for c in a.columns],random_state=42);t=time.time();m.fit(a,ytr);p=m.predict_proba(b)[:,1];row={"features":"domain43_plus_seoyeon5","fold":year,"brier":brier_score_loss(yva,p),"logloss":log_loss(yva,p),"roc_auc":roc_auc_score(yva,p),"seconds":time.time()-t};rows.append(row);parts.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}));print(row)
 result=pd.DataFrame(rows);mean={"features":"domain43_plus_seoyeon5","fold":"mean_2022_2024","brier":result.brier.mean(),"logloss":result.logloss.mean(),"roc_auc":result.roc_auc.mean(),"seconds":result.seconds.sum()};result=pd.concat([result,pd.DataFrame([mean])],ignore_index=True);RES.mkdir(exist_ok=True);result.to_csv(RES/"exp22_seoyeon_interactions_walkforward.csv",index=False,encoding="utf-8-sig");pd.concat(parts,ignore_index=True).to_csv(RES/"exp22_seoyeon_interactions_oof.csv.gz",index=False,compression="gzip");print(result.to_string(index=False))
if __name__=="__main__":main()

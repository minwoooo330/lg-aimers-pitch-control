# -*- coding: utf-8 -*-
"""실험 15: Chronological CatBoost 2022/2023 추가 시간검증 후 2024와 결합."""
from pathlib import Path
import gc,time
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from features import add_features
HERE=Path(__file__).resolve().parent;DATA=HERE/"data"/"train.csv";RES=HERE/"results";ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand","pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
def main():
 start=time.time();df=pd.read_csv(DATA,encoding="utf-8-sig");base=[c for c in df.columns if c not in(ID,TARGET)];rows=[];parts=[]
 for year in [2022,2023]:
  tr=df[df.season<year];va=df[df.season==year];xtr=pd.concat([tr[base].copy(),add_features(tr)],axis=1);xva=pd.concat([va[base].copy(),add_features(va)],axis=1)
  for c in CATS:xtr[c]=xtr[c].where(xtr[c].notna(),"__MISSING__").astype(str);xva[c]=xva[c].where(xva[c].notna(),"__MISSING__").astype(str)
  ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8);catidx=[xtr.columns.get_loc(c) for c in CATS]
  model=CatBoostClassifier(iterations=300,learning_rate=.06,depth=8,loss_function="Logloss",l2_leaf_reg=3.,random_seed=42,thread_count=-1,allow_writing_files=False,verbose=100,has_time=True)
  fit=time.time();model.fit(xtr,ytr,cat_features=catidx);pred=model.predict_proba(xva)[:,1];sec=time.time()-fit
  row={"model":"catboost","features":"domain43_has_time","fold":year,"brier":brier_score_loss(yva,pred),"logloss":log_loss(yva,pred,labels=[0,1]),"roc_auc":roc_auc_score(yva,pred),"pred_mean":pred.mean(),"target_mean":yva.mean(),"seconds":sec};rows.append(row);parts.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":pred}));print(row)
  del tr,va,xtr,xva,ytr,yva,model,pred;gc.collect()
 old=pd.read_csv(RES/"exp14_catboost_chronological_2024.csv");rows.extend(old.to_dict("records"));parts.append(pd.read_csv(RES/"exp14_catboost_chronological_2024_oof.csv.gz"))
 result=pd.DataFrame(rows);summary={"model":"catboost","features":"domain43_has_time","fold":"mean_2022_2024","brier":result.brier.mean(),"logloss":result.logloss.mean(),"roc_auc":result.roc_auc.mean(),"seconds":result.seconds.sum()};result=pd.concat([result,pd.DataFrame([summary])],ignore_index=True)
 result.to_csv(RES/"exp15_catboost_chronological_walkforward.csv",index=False,encoding="utf-8-sig");pd.concat(parts,ignore_index=True).to_csv(RES/"exp15_catboost_chronological_oof.csv.gz",index=False,compression="gzip");print(result.to_string(index=False));print(f"total {time.time()-start:.1f}s")
if __name__=="__main__":main()

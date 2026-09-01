# -*- coding: utf-8 -*-
"""실험 12: 고카디널리티 ID를 범주형으로 처리하는 XGBoost 2024 화면."""
from pathlib import Path
import time,gc
import numpy as np,pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from features import add_features
HERE=Path(__file__).resolve().parent;DATA=HERE/"data"/"train.csv";RES=HERE/"results"
ID,TARGET="row_id","control_success";CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand","pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
def main():
 start=time.time();df=pd.read_csv(DATA,encoding="utf-8-sig");tr=df[df.season<2024];va=df[df.season==2024];base=[c for c in df.columns if c not in(ID,TARGET)]
 xtr=pd.concat([tr[base].copy(),add_features(tr)],axis=1);xva=pd.concat([va[base].copy(),add_features(va)],axis=1)
 for c in CATS:
  categories=pd.Index(tr[c].dropna().astype(str).unique());xtr[c]=pd.Categorical(tr[c].astype(str),categories=categories);xva[c]=pd.Categorical(va[c].astype(str),categories=categories)
 ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8)
 configs={"xgb_d6_400":dict(n_estimators=400,max_depth=6,min_child_weight=50),"xgb_d8_400":dict(n_estimators=400,max_depth=8,min_child_weight=80)}
 rows=[]
 for name,special in configs.items():
  begin=time.time();m=XGBClassifier(objective="binary:logistic",learning_rate=.03,subsample=.9,colsample_bytree=.9,reg_lambda=5.,reg_alpha=.1,max_bin=256,tree_method="hist",enable_categorical=True,max_cat_to_onehot=4,max_cat_threshold=64,n_jobs=-1,random_state=42,eval_metric="logloss",**special)
  m.fit(xtr,ytr,verbose=False);p=m.predict_proba(xva)[:,1];sec=time.time()-begin
  row={"features":name,"fold":"2024","brier":brier_score_loss(yva,p),"logloss":log_loss(yva,p),"roc_auc":roc_auc_score(yva,p),"pred_mean":p.mean(),"target_mean":yva.mean(),"seconds":sec};rows.append(row)
  pd.DataFrame({ID:va[ID].to_numpy(),"season":2024,TARGET:yva,"prediction":p}).to_csv(RES/f"exp12_{name}_2024.csv.gz",index=False,compression="gzip")
  print(row);del m,p;gc.collect()
 RES.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(RES/"exp12_xgboost_2024_screen.csv",index=False,encoding="utf-8-sig");print(f"total {time.time()-start:.1f}s")
if __name__=="__main__":main()

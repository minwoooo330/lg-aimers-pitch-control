# -*- coding: utf-8 -*-
"""실험 14: 입력 시간 순서를 보존한 CatBoost 2024 선별 검증."""
from pathlib import Path
import time
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from features import add_features
HERE=Path(__file__).resolve().parent;DATA=HERE/"data"/"train.csv";RES=HERE/"results";ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand","pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
def main():
 start=time.time();df=pd.read_csv(DATA,encoding="utf-8-sig");tr=df[df.season<2024];va=df[df.season==2024];base=[c for c in df.columns if c not in(ID,TARGET)]
 xtr=pd.concat([tr[base].copy(),add_features(tr)],axis=1);xva=pd.concat([va[base].copy(),add_features(va)],axis=1)
 for c in CATS:xtr[c]=xtr[c].where(xtr[c].notna(),"__MISSING__").astype(str);xva[c]=xva[c].where(xva[c].notna(),"__MISSING__").astype(str)
 ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8);catidx=[xtr.columns.get_loc(c) for c in CATS]
 model=CatBoostClassifier(iterations=300,learning_rate=.06,depth=8,loss_function="Logloss",l2_leaf_reg=3.,random_seed=42,thread_count=-1,allow_writing_files=False,verbose=50,has_time=True)
 fit=time.time();model.fit(xtr,ytr,cat_features=catidx);pred=model.predict_proba(xva)[:,1];sec=time.time()-fit
 row={"model":"catboost","features":"domain43_has_time","fold":2024,"brier":brier_score_loss(yva,pred),"logloss":log_loss(yva,pred,labels=[0,1]),"roc_auc":roc_auc_score(yva,pred),"pred_mean":pred.mean(),"target_mean":yva.mean(),"seconds":sec}
 RES.mkdir(exist_ok=True);pd.DataFrame([row]).to_csv(RES/"exp14_catboost_chronological_2024.csv",index=False,encoding="utf-8-sig");pd.DataFrame({ID:va[ID].to_numpy(),"season":2024,TARGET:yva,"prediction":pred}).to_csv(RES/"exp14_catboost_chronological_2024_oof.csv.gz",index=False,compression="gzip")
 print(row);print(f"total {time.time()-start:.1f}s")
if __name__=="__main__":main()

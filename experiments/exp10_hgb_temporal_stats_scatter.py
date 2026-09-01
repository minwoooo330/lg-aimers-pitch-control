# -*- coding: utf-8 -*-
"""실험 10: 과거 시즌 전용 평활 범주 통계와 최근 비가운데 실패 proxy."""
from pathlib import Path
import gc,time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from features import add_features
from temporal_target_features import make_temporal_target_features,add_scatter_proxy
HERE=Path(__file__).resolve().parent;DATA=HERE/"data"/"train.csv";RES=HERE/"results"
ID,TARGET="row_id","control_success";CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def encode(tr,va,cols):
 a=tr[cols].copy();b=va[cols].copy()
 for c in CATS:
  vals=sorted(tr[c].dropna().astype(str).unique());mp={v:i for i,v in enumerate(vals)}
  a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16);b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
 return a,b

def main():
 start=time.time();df=pd.read_csv(DATA,encoding="utf-8-sig");base=[c for c in df.columns if c not in(ID,TARGET)];rows=[];oofs={}
 for year in [2022,2023,2024]:
  tr=df[df.season<year];va=df[df.season==year];ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8)
  xbtr,xbva=encode(tr,va,base);domain_tr=add_features(tr);domain_va=add_features(va);scatter_tr=add_scatter_proxy(tr);scatter_va=add_scatter_proxy(va)
  te_start=time.time();te_tr,te_va=make_temporal_target_features(tr,va);print(f"{year} temporal stats {time.time()-te_start:.1f}초")
  configs={"domain_scatter":(scatter_tr,scatter_va),"domain_temporal_te":(te_tr,te_va),"domain_te_scatter":(pd.concat([te_tr,scatter_tr],axis=1),pd.concat([te_va,scatter_va],axis=1))}
  for name,(extra_tr,extra_va) in configs.items():
   begin=time.time();xtr=pd.concat([xbtr,domain_tr,extra_tr],axis=1);xva=pd.concat([xbva,domain_va,extra_va],axis=1)
   model=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.,early_stopping=False,categorical_features=[c in CATS for c in xtr.columns],random_state=42)
   model.fit(xtr,ytr);pred=model.predict_proba(xva)[:,1];sec=time.time()-begin
   row={"model":"hgb","features":name,"fold":str(year),"n_features":xtr.shape[1],"brier":brier_score_loss(yva,pred),"logloss":log_loss(yva,pred,labels=[0,1]),"roc_auc":roc_auc_score(yva,pred),"pred_mean":pred.mean(),"target_mean":yva.mean(),"seconds":sec};rows.append(row)
   oofs.setdefault(name,[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":pred}))
   print(f"{year} {name}: Brier={row['brier']:.6f}, AUC={row['roc_auc']:.6f}, {sec:.1f}초")
   del xtr,xva,model,pred;gc.collect()
  del tr,va,xbtr,xbva,domain_tr,domain_va,scatter_tr,scatter_va,te_tr,te_va,ytr,yva,configs;gc.collect()
 result=pd.DataFrame(rows);summary=result.groupby("features",as_index=False).agg(brier=("brier","mean"),logloss=("logloss","mean"),roc_auc=("roc_auc","mean"),seconds=("seconds","sum"));summary["model"]="hgb";summary["fold"]="mean_2022_2024"
 for c in result.columns:
  if c not in summary:summary[c]=np.nan
 result=pd.concat([result,summary[result.columns]],ignore_index=True);RES.mkdir(exist_ok=True);result.to_csv(RES/"exp10_hgb_temporal_stats_scatter.csv",index=False,encoding="utf-8-sig")
 for name,parts in oofs.items():pd.concat(parts,ignore_index=True).to_csv(RES/f"exp10_{name}_oof.csv.gz",index=False,compression="gzip")
 print(result[result.fold.eq("mean_2022_2024")][["features","brier","logloss","roc_auc","seconds"]].to_string(index=False));print(f"총 {time.time()-start:.1f}초")
if __name__=="__main__":main()

# -*- coding: utf-8 -*-
"""실험 11: 시즌 하락 드리프트에 대한 완만한 최근 시즌 가중 시간검증."""
from pathlib import Path
import gc,time
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from features import add_features
HERE=Path(__file__).resolve().parent;DATA=HERE/"data"/"train.csv";RES=HERE/"results";ID,TARGET="row_id","control_success";CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
CONFIGS={"decay_1.05":1.05,"decay_1.10":1.10,"decay_1.15":1.15,"decay_1.20":1.20,"decay_1.30":1.30}
def encode(tr,va,cols):
 a=tr[cols].copy();b=va[cols].copy()
 for c in CATS:
  vals=sorted(tr[c].dropna().astype(str).unique());mp={v:i for i,v in enumerate(vals)}
  a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16);b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
 return a,b

def main():
 start=time.time();df=pd.read_csv(DATA,encoding="utf-8-sig");base=[c for c in df.columns if c not in(ID,TARGET)];rows=[];oofs={}
 for year in [2022,2023,2024]:
  tr=df[df.season<year];va=df[df.season==year];xtr,xva=encode(tr,va,base);xtr=pd.concat([xtr,add_features(tr)],axis=1);xva=pd.concat([xva,add_features(va)],axis=1);ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8)
  for name,decay in CONFIGS.items():
   sw=np.power(decay,tr.season.to_numpy(float)-(year-1));sw=sw/sw.mean();begin=time.time()
   m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.,early_stopping=False,categorical_features=[c in CATS for c in xtr.columns],random_state=42)
   m.fit(xtr,ytr,sample_weight=sw);pred=m.predict_proba(xva)[:,1];sec=time.time()-begin
   row={"model":"hgb","features":name,"fold":str(year),"brier":brier_score_loss(yva,pred),"logloss":log_loss(yva,pred,labels=[0,1]),"roc_auc":roc_auc_score(yva,pred),"pred_mean":pred.mean(),"target_mean":yva.mean(),"seconds":sec};rows.append(row)
   oofs.setdefault(name,[]).append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":pred}));print(f"{year} {name}: Brier={row['brier']:.6f}, mean={row['pred_mean']:.5f}/{row['target_mean']:.5f}, {sec:.1f}초")
   del m,pred,sw;gc.collect()
  del tr,va,xtr,xva,ytr,yva;gc.collect()
 result=pd.DataFrame(rows);summary=result.groupby("features",as_index=False).agg(brier=("brier","mean"),logloss=("logloss","mean"),roc_auc=("roc_auc","mean"),seconds=("seconds","sum"));summary["model"]="hgb";summary["fold"]="mean_2022_2024"
 for c in result.columns:
  if c not in summary:summary[c]=np.nan
 result=pd.concat([result,summary[result.columns]],ignore_index=True);RES.mkdir(exist_ok=True);result.to_csv(RES/"exp11_hgb_recency.csv",index=False,encoding="utf-8-sig")
 for name,parts in oofs.items():pd.concat(parts,ignore_index=True).to_csv(RES/f"exp11_{name}_oof.csv.gz",index=False,compression="gzip")
 print(result[result.fold.eq("mean_2022_2024")][["features","brier","logloss","roc_auc","seconds"]].sort_values("brier").to_string(index=False));print(f"총 {time.time()-start:.1f}초")
if __name__=="__main__":main()

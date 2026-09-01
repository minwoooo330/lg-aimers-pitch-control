# -*- coding: utf-8 -*-
"""실험 85: LGBM 멤버에 hf 56피처 주입 — GBDT 6멤버 중 유일하게 hf가 없던 슬롯 완성."""
from pathlib import Path
import gc,time
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from hfeatures import add_hfeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
HERE=Path(__file__).resolve().parent;DATA=HERE/"data"/"train.csv";RES=HERE/"results";ID,TARGET="row_id","control_success";CATS=["top_bottom","game_type","base_state"]
def add_interactions(df):
 f=df.copy();f["inter_pitcher_x_balls"]=f.asof_pitcher_success_rate*f.balls_before;f["inter_pitcher_x_strikes"]=f.asof_pitcher_success_rate*f.strikes_before;f["inter_matchup_diff"]=f.asof_pitcher_success_rate-f.asof_batter_success_rate;f["inter_platoon_same"]=(f.pitcher_hand==f.batter_hand).astype(int);f["inter_pressure"]=f.num_runners_on*f.outs_before;f["inter_li_pitcher"]=f.li*f.asof_pitcher_success_rate;f["inter_count_diff"]=f.balls_before-f.strikes_before;f["inter_reverse_x_base"]=f.asof_pitcher_reverse_rate*f.num_runners_on;return f
def main():
 start=time.time();df=pd.read_csv(DATA,encoding="utf-8-sig");base=[c for c in df.columns if c not in(ID,TARGET)];rows=[];parts=[]
 for year in [2022,2023,2024]:
  tr=df[df.season<year];va=df[df.season==year];xtr=pd.concat([add_interactions(tr[base]),add_hfeatures(tr)],axis=1);xva=pd.concat([add_interactions(va[base]),add_hfeatures(va)],axis=1);nums=[c for c in xtr.columns if c not in CATS]
  pre=ColumnTransformer([("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),CATS),("num",SimpleImputer(strategy="median"),nums)])
  clf=LGBMClassifier(n_estimators=153,learning_rate=0.03331120682555046,num_leaves=61,max_depth=8,min_child_samples=247,reg_alpha=0.013389279116695478,reg_lambda=0.0022976624140824994,feature_fraction=0.905269907953647,bagging_fraction=0.8320457481218804,bagging_freq=1,random_state=42,n_jobs=-1,verbose=-1)
  model=Pipeline([("pre",pre),("clf",clf)]);ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8);t=time.time();model.fit(xtr,ytr);p=model.predict_proba(xva)[:,1];sec=time.time()-t;row={"model":"lgbm_hf","fold":year,"brier":brier_score_loss(yva,p),"logloss":log_loss(yva,p),"roc_auc":roc_auc_score(yva,p),"pred_mean":p.mean(),"target_mean":yva.mean(),"seconds":sec};rows.append(row);parts.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}));print(row);del tr,va,xtr,xva,model,pre,clf,p;gc.collect()
 result=pd.DataFrame(rows);mean={"model":"lgbm_hf","fold":"mean_2022_2024","brier":result.brier.mean(),"logloss":result.logloss.mean(),"roc_auc":result.roc_auc.mean(),"seconds":result.seconds.sum()};result=pd.concat([result,pd.DataFrame([mean])],ignore_index=True);RES.mkdir(exist_ok=True);result.to_csv(RES/"exp85_lgbm_hf.csv",index=False,encoding="utf-8-sig");pd.concat(parts,ignore_index=True).to_csv(RES/"exp85_lgbm_hf_oof.csv.gz",index=False,compression="gzip");print(result.to_string(index=False));print(f"total={time.time()-start:.1f}s")
if __name__=="__main__":main()

# -*- coding: utf-8 -*-
"""실험 19: 과거시즌 구종확률×투수별 구종 제구율 피처 하나의 2024 gate."""
from pathlib import Path
import gc,time
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss,roc_auc_score,log_loss
from features import add_features
from trackman_features import match_exact_games
HERE=Path(__file__).resolve().parent;DATA=HERE/"data";RES=HERE/"results";ID,TARGET="row_id","control_success"
GROUPS=["fastball","breaking","offspeed"]
QF=["season","game_month","inning","top_bottom","game_type","base_state","balls_before","strikes_before","outs_before","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","li","pitcher_hand","batter_hand","asof_pitcher_n","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
QC=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
MATCH=["season","inning","top_bottom","balls_before","strikes_before","outs_before","pitcher_id","pitcher_hand","batter_hand"]
TM_COLS=["trackman_game_id","pitch_no","inning","top_bottom","balls_before","strikes_before","outs_before","pitcher_trackman_id","pitcher_hand","batter_hand","pitch_type_group"]
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
def qmatrix(d):
 p=d[["asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]].to_numpy(float).copy();p[~np.isfinite(p)]=1/3;p=np.clip(p,1e-6,None);return p/p.sum(1,keepdims=True)
def qx(train,apply):
 a=train[QF].copy();b=apply[QF].copy()
 for c in QC:
  vals=sorted(train[c].dropna().astype(str).unique());mp={v:i for i,v in enumerate(vals)};a[c]=train[c].astype(str).map(mp).fillna(-1).astype(np.int16);b[c]=apply[c].astype(str).map(mp).fillna(-1).astype(np.int16)
 for x,s in [(a,train),(b,apply)]:
  x["count_id"]=(s.balls_before*3+s.strikes_before).astype(np.int8);x["scoring_pos"]=((s.runner_on_2b==1)|(s.runner_on_3b==1)).astype(np.int8);x["high_li"]=(s.li>=1.5).astype(np.int8);x["same_hand"]=(s.pitcher_hand.astype(str)==s.batter_hand.astype(str)).astype(np.int8)
 return a,b
def predict_q(history,apply):
 if len(history)<1000:return qmatrix(apply)
 a,b=qx(history,apply);m=HistGradientBoostingClassifier(loss="log_loss",max_iter=150,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=2.,early_stopping=False,categorical_features=[c in QC for c in a.columns],random_state=42);m.fit(a,history.pitch_type_group);p0=m.predict_proba(b);p=np.column_stack([p0[:,list(m.classes_).index(g)] for g in GROUPS]);del a,b,m;gc.collect();return p
def make_latent(history,apply,q,k=100.):
 prior=float(history.control_success.mean()) if len(history) else .52;theta=np.zeros((len(apply),3),float)
 for j,g in enumerate(GROUPS):
  hg=history[history.pitch_type_group==g];parent=float(hg.control_success.mean()) if len(hg) else prior
  st=hg.groupby("pitcher_id").control_success.agg(["sum","count"]);hit=st.reindex(pd.Index(apply.pitcher_id.to_numpy(),name="pitcher_id"));n=hit["count"].fillna(0).to_numpy(float);s=hit["sum"].fillna(0).to_numpy(float);theta[:,j]=(s+k*parent)/(n+k)
 return (q*theta).sum(1).astype(np.float32)
def encode_base(tr,va,base):
 a=tr[base].copy();b=va[base].copy()
 for c in CATS:
  vals=sorted(tr[c].dropna().astype(str).unique());mp={v:i for i,v in enumerate(vals)};a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16);b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
 return a,b
def main():
 start=time.time();df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig");tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
 mg,ts,matches=match_exact_games(df,tm);mi=mg.groupby("_game_idx",sort=False).indices;ti=ts.groupby("trackman_game_id",sort=False).indices;parts=[]
 keep=list(dict.fromkeys([ID,"season","pitcher_id",TARGET,*QF]))
 for row in matches.itertuples(index=False):
  a=mg.iloc[mi[row.main_game_idx]][keep].reset_index(drop=True);b=ts.iloc[ti[row.trackman_game_id]][["pitch_type_group"]].reset_index(drop=True)
  if len(a)==len(b):parts.append(pd.concat([a,b],axis=1))
 aligned=pd.concat(parts,ignore_index=True);aligned=aligned[aligned.pitch_type_group.isin(GROUPS)]
 latent=pd.Series(index=df.index,dtype=np.float32)
 for year in [2019,2020,2021,2022,2023,2024]:
  app=df[df.season==year];hist=aligned[aligned.season<year];q=predict_q(hist,app);latent.loc[app.index]=make_latent(hist,app,q);print(f"latent {year}: history={len(hist):,}, apply={len(app):,}")
 tr=df[df.season<2024];va=df[df.season==2024];base=[c for c in df.columns if c not in(ID,TARGET)];xbtr,xbva=encode_base(tr,va,base);domtr=add_features(tr);domva=add_features(va);ytr=tr[TARGET].to_numpy(np.int8);yva=va[TARGET].to_numpy(np.int8);rows=[]
 for name,use in [("domain43",False),("domain43_latent_pitch_command",True)]:
  xtr=pd.concat([xbtr,domtr],axis=1);xva=pd.concat([xbva,domva],axis=1)
  if use:xtr["latent_pitch_command"]=latent.loc[tr.index].to_numpy();xva["latent_pitch_command"]=latent.loc[va.index].to_numpy()
  model=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.,early_stopping=False,categorical_features=[c in CATS for c in xtr.columns],random_state=42);fit=time.time();model.fit(xtr,ytr);pred=model.predict_proba(xva)[:,1];sec=time.time()-fit
  row={"features":name,"fold":2024,"brier":brier_score_loss(yva,pred),"logloss":log_loss(yva,pred,labels=[0,1]),"roc_auc":roc_auc_score(yva,pred),"pred_mean":pred.mean(),"seconds":sec};rows.append(row);print(row);del xtr,xva,model,pred;gc.collect()
 result=pd.DataFrame(rows);gain=float(result.iloc[0].brier-result.iloc[1].brier);passed=gain>=1e-5;result["brier_gain_vs_base"]=[0,gain];result["pass_threshold"]=passed;RES.mkdir(exist_ok=True);result.to_csv(RES/"exp19_latent_pitch_command_2024.csv",index=False,encoding="utf-8-sig");print("PASS" if passed else "STOP",f"gain={gain:.8f}, total={time.time()-start:.1f}s")
if __name__=="__main__":main()

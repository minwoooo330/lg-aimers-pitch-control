# -*- coding: utf-8 -*-
"""실험 18: 공식 사전정보만으로 3구종 확률을 예측하는 2024 저비용 화면."""
from pathlib import Path
import time
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss,accuracy_score
from trackman_features import match_exact_games
HERE=Path(__file__).resolve().parent;DATA=HERE/"data";RES=HERE/"results"
CLASSES=["fastball","breaking","offspeed"]
MATCH=["season","inning","top_bottom","balls_before","strikes_before","outs_before","pitcher_id","pitcher_hand","batter_hand"]
FEATURES=["season","game_month","inning","top_bottom","game_type","base_state","balls_before","strikes_before","outs_before","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","li","pitcher_hand","batter_hand","asof_pitcher_n","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
MAIN_COLS=list(dict.fromkeys(["row_id","control_success",*MATCH,*FEATURES]))
TM_COLS=["trackman_game_id","pitch_no","inning","top_bottom","balls_before","strikes_before","outs_before","pitcher_trackman_id","pitcher_hand","batter_hand","pitch_type_group"]
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
def main():
 start=time.time();main=pd.read_csv(DATA/"train.csv",usecols=MAIN_COLS,encoding="utf-8-sig");tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
 mg,ts,matches=match_exact_games(main,tm);mi=mg.groupby("_game_idx",sort=False).indices;ti=ts.groupby("trackman_game_id",sort=False).indices;parts=[]
 for row in matches.itertuples(index=False):
  a=mg.iloc[mi[row.main_game_idx]].reset_index(drop=True);b=ts.iloc[ti[row.trackman_game_id]][["pitch_type_group"]].reset_index(drop=True)
  if len(a)==len(b):parts.append(pd.concat([a,b],axis=1))
 d=pd.concat(parts,ignore_index=True);d=d[d.pitch_type_group.isin(CLASSES)].copy();tr=d[d.season<2024];va=d[d.season==2024]
 xtr=tr[FEATURES].copy();xva=va[FEATURES].copy()
 for c in CATS:
  vals=sorted(tr[c].dropna().astype(str).unique());mp={v:i for i,v in enumerate(vals)};xtr[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16);xva[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
 # 행 단독 상황 상호작용
 for x,src in [(xtr,tr),(xva,va)]:
  x["count_id"]=(src.balls_before*3+src.strikes_before).astype(np.int8);x["scoring_pos"]=((src.runner_on_2b==1)|(src.runner_on_3b==1)).astype(np.int8);x["high_li"]=(src.li>=1.5).astype(np.int8);x["same_hand"]=(src.pitcher_hand.astype(str)==src.batter_hand.astype(str)).astype(np.int8)
 ytr=tr.pitch_type_group.to_numpy();yva=va.pitch_type_group.to_numpy()
 # 공식 asof 구종비율 baseline
 ratecols=["asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
 prior=tr.pitch_type_group.value_counts(normalize=True).reindex(CLASSES).to_numpy();pb=va[ratecols].to_numpy(float).copy();bad=~np.isfinite(pb);pb[bad]=np.take(prior,np.where(bad)[1]);pb=np.clip(pb,1e-6,None);pb/=pb.sum(axis=1,keepdims=True)
 yi=np.array([CLASSES.index(v) for v in yva]);base_ll=-float(np.mean(np.log(pb[np.arange(len(yva)),yi])));base_acc=accuracy_score(yva,np.array(CLASSES)[pb.argmax(1)])
 model=HistGradientBoostingClassifier(loss="log_loss",max_iter=150,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=2.,early_stopping=False,categorical_features=[c in CATS for c in xtr.columns],random_state=42)
 fit=time.time();model.fit(xtr,ytr);pm0=model.predict_proba(xva);pm=np.column_stack([pm0[:,list(model.classes_).index(c)] for c in CLASSES]);sec=time.time()-fit
 model_ll=-float(np.mean(np.log(pm[np.arange(len(yva)),yi])));model_acc=accuracy_score(yva,np.array(CLASSES)[pm.argmax(1)]);gain=base_ll-model_ll;passed=gain>=.005
 row={"train_rows":len(tr),"valid_rows":len(va),"exact_games":len(matches),"asof_mix_logloss":base_ll,"model_logloss":model_ll,"logloss_gain":gain,"asof_mix_accuracy":base_acc,"model_accuracy":model_acc,"fit_seconds":sec,"pass_threshold":passed}
 RES.mkdir(exist_ok=True);pd.DataFrame([row]).to_csv(RES/"exp18_pitch_family_2024_screen.csv",index=False,encoding="utf-8-sig");pd.DataFrame({"row_id":va.row_id.to_numpy(),"season":2024,"pitch_type_group":yva,"p_fastball":pm[:,0],"p_breaking":pm[:,1],"p_offspeed":pm[:,2]}).to_csv(RES/"exp18_pitch_family_2024_oof.csv.gz",index=False,compression="gzip")
 print(row);print("PASS" if passed else "STOP",f"total={time.time()-start:.1f}s")
if __name__=="__main__":main()

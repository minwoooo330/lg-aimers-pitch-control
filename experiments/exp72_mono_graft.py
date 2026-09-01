# -*- coding: utf-8 -*-
"""실험 72: 단조 제약(monotonic_cst)을 우리 HGB 멤버 3종에 이식해 3-fold 재검.

에이전트 exp71은 '퓨처스 2019~22 제외' 학습본에서 monoA가 2024 +8.79e-5임을 보였다.
그 모델을 그대로 우리 체인에 넣으면 2024 기여 0이므로(이득이 퓨처스 처리 차이에서 옴),
같은 제약을 '우리 학습 조건'의 HGB 3종에 붙여 실제 전달 여부를 확인한다.
"""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features
from trackman_features import (match_exact_games, build_pitcher_mapping,
                               build_trackman_profile, build_trackman_profile_by_league)

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
MONO={"asof_pitcher_success_rate":1,"asof_pitcher_strike_rate":1,"asof_batter_success_rate":1,
 "asof_pitcher_prev1_game_success_rate":1,"asof_pitcher_prev3_game_success_rate":1,
 "asof_pitcher_prev5_game_success_rate":1,"asof_pitcher_middle_rate":-1,
 "asof_pitcher_reverse_rate":-1,"asof_pitcher_ball_rate":-1,"asof_batter_middle_rate":-1,
 "asof_pitcher_prev3_game_middle_rate":-1,"asof_pitcher_prev5_game_middle_rate":-1}
PRM=dict(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,
         l2_regularization=1.,early_stopping=False,random_state=42)

def main():
    t0=time.time()
    df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(DATA/"trackman_history.csv",encoding="utf-8-sig")
    main_g,tm_s,matches=match_exact_games(df,tm)
    rows=[]; store={}
    for year in [2022,2023,2024]:
        mp=build_pitcher_mapping(main_g,tm_s,matches,year)
        prof=build_trackman_profile(tm,mp,year)
        profL=build_trackman_profile_by_league(tm,mp,year)
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        def enc(d):
            x=d[cols].copy()
            for c in CATS:
                vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
                x[c]=d[c].astype(str).map(m).fillna(-1).astype(np.int16)
            return pd.concat([x,add_features(d)],axis=1)
        xa0,xb0=enc(tr),enc(va)
        variants={"domain":(xa0,xb0)}
        # tm_mean: 전체리그 구종평균 프로필 결합
        pa=tr[["pitcher_id"]].join(prof,on="pitcher_id"); pb=va[["pitcher_id"]].join(prof,on="pitcher_id")
        variants["tm_mean"]=(pd.concat([xa0,pa.drop(columns=["pitcher_id"]).reset_index(drop=True)],axis=1),
                             pd.concat([xb0,pb.drop(columns=["pitcher_id"]).reset_index(drop=True)],axis=1))
        # league_role: R/F 역할 프로필 결합
        def joinL(d,x):
            idx=pd.MultiIndex.from_arrays([d.pitcher_id.to_numpy(),d.game_type.to_numpy()])
            r=profL.reindex(idx).reset_index(drop=True)
            keep=[c for c in r.columns if "share" in c or "role" in c or "starter" in c]
            return pd.concat([x,r[keep] if keep else r],axis=1)
        variants["league_role"]=(joinL(tr,xa0),joinL(va,xb0))
        for name,(xa,xb) in variants.items():
            for mono in [False,True]:
                cst=None
                if mono:
                    cst=np.array([MONO.get(c,0) for c in xa.columns],dtype=int)
                m=HistGradientBoostingClassifier(**PRM,
                     categorical_features=[c in CATS for c in xa.columns],
                     monotonic_cst=cst).fit(xa,ytr)
                p=m.predict_proba(xb)[:,1]
                key=f"{name}{'_mono' if mono else ''}"
                rows.append({"fold":year,"variant":key,"brier":brier_score_loss(yva,p),
                             "auc":roc_auc_score(yva,p)})
                print(rows[-1],flush=True)
                store.setdefault(key,[]).append(pd.DataFrame({ID:va[ID].to_numpy(),
                    "season":year,TARGET:yva,"prediction":p}))
                del m,p; gc.collect()
        del xa0,xb0,variants,tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp72_mono_graft.csv",index=False,encoding="utf-8-sig")
    for k,v in store.items():
        pd.concat(v,ignore_index=True).to_csv(RES/f"exp72_{k}_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

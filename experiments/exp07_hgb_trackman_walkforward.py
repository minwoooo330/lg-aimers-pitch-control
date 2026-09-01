# -*- coding: utf-8 -*-
"""실험 07: HGB 도메인 모델에 과거 Trackman 투수 프로필 추가."""
from pathlib import Path
import gc
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from features import add_features
from trackman_features import match_exact_games, build_pitcher_mapping, build_trackman_profile

HERE=Path(__file__).resolve().parent
TRAIN_PATH=HERE/"data"/"train.csv"
TM_PATH=HERE/"data"/"trackman_history.csv"
RESULT_DIR=HERE/"results"
RESULT_PATH=RESULT_DIR/"exp07_hgb_trackman_walkforward.csv"
OOF_PATH=RESULT_DIR/"exp07_hgb_trackman_oof.csv.gz"
ID,TARGET="row_id","control_success"
CAT_COLS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
FOLDS=[("2022",2022),("2023",2023),("2024",2024)]
TM_USECOLS=["trackman_id","season","trackman_game_id","pitch_no","inning","top_bottom",
 "balls_before","strikes_before","outs_before","pitcher_trackman_id","pitcher_hand",
 "pitcher_team","pitch_type_group","rel_speed","spin_rate","induced_vert_break",
 "horz_break","extension","rel_height","rel_side","zone_speed","batter_hand"]
MATCH_COLS=["season","inning","top_bottom","balls_before","strikes_before","outs_before",
            "pitcher_id","pitcher_hand","batter_hand"]


def encode_base(train,valid,base_cols):
    xtr=train[base_cols].copy();xva=valid[base_cols].copy()
    for col in CAT_COLS:
        vals=sorted(train[col].dropna().astype(str).unique())
        mapping={v:i for i,v in enumerate(vals)}
        xtr[col]=train[col].astype(str).map(mapping).fillna(-1).astype(np.int16)
        xva[col]=valid[col].astype(str).map(mapping).fillna(-1).astype(np.int16)
    return xtr,xva


def attach_profile(frame,profile):
    block=frame[["pitcher_id"]].join(profile,on="pitcher_id").drop(columns="pitcher_id")
    block["tm_profile_available"]=(block["tm_mapping_votes"].notna()).astype(np.int8)
    return block.astype(np.float32)


def main():
    started=time.time()
    print("메인 데이터 로드")
    df=pd.read_csv(TRAIN_PATH,encoding="utf-8-sig")
    print("Trackman 로드")
    tm=pd.read_csv(TM_PATH,encoding="utf-8-sig",usecols=TM_USECOLS)
    print("동일 경기의 완전 일치 투구열 탐색")
    main_games,tm_sorted,matches=match_exact_games(df[MATCH_COLS],tm)
    print(f"정확히 연결된 경기: {len(matches):,}개")
    base_cols=[c for c in df.columns if c not in (ID,TARGET)]
    rows=[];oof=[]

    for fold_name,year in FOLDS:
        fold_started=time.time()
        mapping=build_pitcher_mapping(main_games,tm_sorted,matches,year,min_votes=20,min_purity=0.99)
        profile=build_trackman_profile(tm,mapping,year)
        train=df[df.season<year];valid=df[df.season==year]
        train_cov=train.pitcher_id.isin(mapping.pitcher_id).mean()
        valid_cov=valid.pitcher_id.isin(mapping.pitcher_id).mean()
        print(f"\n[{year}] 투수 대응 {len(mapping)}명, 학습행 커버 {train_cov:.1%}, 시험행 커버 {valid_cov:.1%}")
        xtr,xva=encode_base(train,valid,base_cols)
        xtr=pd.concat([xtr,add_features(train),attach_profile(train,profile)],axis=1)
        xva=pd.concat([xva,add_features(valid),attach_profile(valid,profile)],axis=1)
        ytr=train[TARGET].to_numpy(np.int8);yva=valid[TARGET].to_numpy(np.int8)
        catmask=[c in CAT_COLS for c in xtr.columns]
        model=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.06,max_leaf_nodes=31,
            min_samples_leaf=200,l2_regularization=1.0,early_stopping=False,
            categorical_features=catmask,random_state=42)
        fit=time.time();model.fit(xtr,ytr);pred=model.predict_proba(xva)[:,1];seconds=time.time()-fit
        row={"model":"hgb","features":"raw47_domain43_trackman65","fold":fold_name,
             "mapped_pitchers":len(mapping),"train_mapping_coverage":train_cov,
             "valid_mapping_coverage":valid_cov,"n_train":len(train),"n_valid":len(valid),
             "pred_mean":float(pred.mean()),"valid_target_mean":float(yva.mean()),
             "brier":brier_score_loss(yva,pred),"logloss":log_loss(yva,pred,labels=[0,1]),
             "roc_auc":roc_auc_score(yva,pred),"seconds":seconds,"n_features":xtr.shape[1]}
        rows.append(row)
        oof.append(pd.DataFrame({ID:valid[ID].to_numpy(),"season":year,TARGET:yva,"prediction":pred}))
        print(f"Brier={row['brier']:.6f}, LogLoss={row['logloss']:.6f}, AUC={row['roc_auc']:.6f}, 시간={seconds:.1f}초")
        del mapping,profile,train,valid,xtr,xva,ytr,yva,model,pred
        gc.collect();print(f"fold 전체={time.time()-fold_started:.1f}초")

    result=pd.DataFrame(rows)
    mean={"model":"hgb","features":"raw47_domain43_trackman65","fold":"mean_2022_2024",
          "brier":result.brier.mean(),"logloss":result.logloss.mean(),
          "roc_auc":result.roc_auc.mean(),"seconds":result.seconds.sum(),
          "valid_mapping_coverage":result.valid_mapping_coverage.mean(),"n_features":result.n_features.max()}
    result=pd.concat([result,pd.DataFrame([mean])],ignore_index=True)
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    result.to_csv(RESULT_PATH,index=False,encoding="utf-8-sig")
    pd.concat(oof,ignore_index=True).to_csv(OOF_PATH,index=False,encoding="utf-8",compression="gzip")
    print("\n",result[["fold","brier","logloss","roc_auc","valid_mapping_coverage","seconds"]].to_string(index=False))
    print(f"결과: {RESULT_PATH}")
    print(f"총 소요={time.time()-started:.1f}초")


if __name__=="__main__":main()

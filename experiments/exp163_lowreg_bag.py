# -*- coding: utf-8 -*-
"""exp163 — v7과 동일 피처, 정규화 철학만 정반대(l2=1, min_data=2, subsample=0.5,
   early stopping)로 5-배깅. 다양성 멤버 추가 목적. 교체 아님."""
"""exp148 — 팀원 v7 파이프라인(1099.47)을 우리 fold로 실행해 OOF 생성.

  v7 설정 재현: min_season=2020, halflife=2, legacy-norm, platoon(투수측만),
                trackman, mono, l2=1000, rsm=0.6. 시드는 42 하나(스크린 목적).
  fold마다 학습은 검증연도 미만, 피처는 그쪽 방식 그대로 시즌별 out-of-time.
  검증연도 행의 피처도 같은 방식(hist = 검증연도 미만 전체)으로 만든다."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
sys.path.insert(0,str(HERE))
import v7_pipeline as V7
from catboost import Pool
from trackman_features import match_exact_games, build_pitcher_mapping

# v7 = legacy-norm
V7.SF_LEAGUE_KEYS=[]; V7.MRC_LEAGUE_KEYS=["season"]
ID,TARGET="row_id","control_success"
L2,RSM=1000.0,0.6; SEED=42; MIN_SEASON=2020

def main():
    t0=time.time()
    full=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(HERE/"data"/"trackman_history.csv",encoding="utf-8-sig")
    mg,ts,matches=match_exact_games(full,tm)
    mp=build_pitcher_mapping(mg,ts,matches,int(full.season.max())+1)
    mp=mp.sort_values("votes",ascending=False)
    mp=mp[~mp.pitcher_trackman_id.duplicated()]
    pid2tid=dict(zip(mp.pitcher_id,mp.pitcher_trackman_id))
    print(f"트랙맨 대응 {len(pid2tid)}명  ({time.time()-t0:.0f}s)",flush=True)
    label_cols=["season","game_type","pitcher_id",TARGET]
    plt_cols=["season","game_type","pitcher_id","batter_id","pitcher_hand",
              "batter_hand","balls_before","strikes_before",TARGET]
    parts_oof=[]
    for year in [2024,2022]:
        tt=time.time()
        df=full[(full.season>=MIN_SEASON)&(full.season<year)].reset_index(drop=True)
        va=full[full.season==year].reset_index(drop=True)
        # --- 학습 피처: 시즌별 out-of-time (그쪽 main 루프 그대로) ---
        parts=[]
        for s in np.sort(df.season.unique()):
            hist=full[full.season<s]
            sub=df[df.season==s]
            base=V7.build_all(sub, V7.build_carry_state(hist),
                              V7.build_projection(hist[label_cols],["pitcher_id"],int(s),reg=V7.MRC_REG) if len(hist) else
                              pd.DataFrame(columns=["mrc_dev","mrc_n"]).rename_axis("pitcher_id"))
            base=pd.concat([base, V7.tm_attach(sub, V7.build_profile(tm[tm.season<s]), pid2tid)],axis=1)
            base=pd.concat([base, V7.plt_attach(sub, V7.build_state(hist[plt_cols]), False)],axis=1)
            parts.append(base)
        X=pd.concat(parts).reindex(df.index)
        # --- 검증 피처: hist = 검증연도 미만 전체 ---
        hist=full[full.season<year]
        bv=V7.build_all(va, V7.build_carry_state(hist),
                        V7.build_projection(hist[label_cols],["pitcher_id"],int(year),reg=V7.MRC_REG))
        bv=pd.concat([bv, V7.tm_attach(va, V7.build_profile(tm[tm.season<year]), pid2tid)],axis=1)
        bv=pd.concat([bv, V7.plt_attach(va, V7.build_state(hist[plt_cols]), False)],axis=1)
        Xv=bv.reindex(columns=X.columns)
        y=df[TARGET].values.astype("float32"); yv=va[TARGET].values.astype("int8")
        cats=[c for c in V7.CATEGORICAL if c in X.columns]
        for c in cats:
            X[c]=X[c].astype("int32").astype("str"); Xv[c]=Xv[c].astype("int32").astype("str")
        mono=[V7.MONO.get(c,0) for c in X.columns]
        age=(year-1)-df.season.values
        w=np.power(2.0,-age/2.0); w=w/w.mean()
        print(f"[{year}] 학습 {len(X):,} x {X.shape[1]}  검증 {len(Xv):,}  조립 {time.time()-tt:.0f}s",flush=True)
        pool=Pool(X,y,weight=w,cat_features=cats)
        # 저규제 + 배깅 + early stopping (LG Aimers 4기 수상팀 스타일)
        # 학습 구간 내부에서만 80/20 분할하므로 검증 연도 정보는 쓰지 않는다.
        from catboost import CatBoostClassifier
        from sklearn.model_selection import StratifiedKFold
        preds=[]
        ytr_=y.to_numpy() if hasattr(y,"to_numpy") else np.asarray(y)
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        for kf,(ti,vi) in enumerate(skf.split(X,ytr_)):
            t1=time.time()
            ptr=Pool(X.iloc[ti],ytr_[ti],weight=w[ti],cat_features=cats)
            pes=Pool(X.iloc[vi],ytr_[vi],weight=w[vi],cat_features=cats)
            m=CatBoostClassifier(learning_rate=0.05,n_estimators=3000,early_stopping_rounds=50,
                                 max_depth=6,l2_leaf_reg=1,min_data_in_leaf=2,subsample=0.5,
                                 bootstrap_type="Bernoulli",loss_function="Logloss",
                                 random_seed=42+kf,verbose=0,thread_count=8,
                                 monotone_constraints=mono)
            m.fit(ptr,eval_set=pes,use_best_model=True)
            preds.append(m.predict_proba(Xv)[:,1])
            print(f"  bag{kf} best_iter={m.get_best_iteration()} ({time.time()-t1:.0f}s)",flush=True)
            del m,ptr,pes; gc.collect()
        p=np.mean(preds,axis=0)
        rel=((va.game_type=="R")|(year>=2023)).to_numpy()
        z=p[rel]+(yv[rel].mean()-p[rel].mean())
        print(f"[{year}] v7 재현 Brier(rel,정렬) {float(np.mean((z-yv[rel])**2)):.8f}  누적 {time.time()-t0:.0f}s",flush=True)
        parts_oof.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yv,"prediction":p}))
        pd.concat(parts_oof,ignore_index=True).to_csv(RES/"exp163_lowreg_bag_oof.csv.gz",index=False,compression="gzip")
        del X,Xv,pool,parts; gc.collect()
    print(f"완료 {time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()

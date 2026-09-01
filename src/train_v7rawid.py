# -*- coding: utf-8 -*-
"""v7 + raw 선수ID 2컬럼 배포 학습 (exp168 게이트: 2024 +1.41 / 2022 +1.71 e-5).
   시드 42/7/2024 x 5설정 = 15모델. v7 시드잡음이 0이라 15개면 충분."""
"""팀원 v7 전체 학습 (2020~2024, 5설정 x 3시드) + 배포 테이블 생성.
   그쪽 train_v7.py의 main()과 동일한 절차. 차이는 트랙맨 매핑을 우리 것으로 쓰는 것뿐."""
from pathlib import Path
import sys, time, gc, json
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; OUT=HERE/"v7rawid_bundle"
sys.path.insert(0,str(HERE))
import v7_pipeline as V7
from catboost import Pool
from trackman_features import match_exact_games, build_pitcher_mapping
V7.SF_LEAGUE_KEYS=[]; V7.MRC_LEAGUE_KEYS=["season"]      # legacy-norm
ID,TARGET="row_id","control_success"
L2,RSM=1000.0,0.6; SEEDS=[42,7,2024]; MIN_SEASON=2020; TARGET_YEAR=2025

def main():
    t0=time.time(); OUT.mkdir(exist_ok=True)
    full=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(HERE/"data"/"trackman_history.csv",encoding="utf-8-sig")
    mg,ts,matches=match_exact_games(full,tm)
    mp=build_pitcher_mapping(mg,ts,matches,int(full.season.max())+1)
    mp=mp.sort_values("votes",ascending=False); mp=mp[~mp.pitcher_trackman_id.duplicated()]
    pid2tid=dict(zip(mp.pitcher_id,mp.pitcher_trackman_id))
    print(f"트랙맨 대응 {len(pid2tid)}명 ({time.time()-t0:.0f}s)",flush=True)
    df=full[full.season>=MIN_SEASON].reset_index(drop=True)
    y=df[TARGET].values.astype("float32")
    label_cols=["season","game_type","pitcher_id",TARGET]
    plt_cols=["season","game_type","pitcher_id","batter_id","pitcher_hand",
              "batter_hand","balls_before","strikes_before",TARGET]
    parts=[]
    for s in np.sort(df.season.unique()):
        hist=full[full.season<s]; sub=df[df.season==s]
        base=V7.build_all(sub, V7.build_carry_state(hist),
                          V7.build_projection(hist[label_cols],["pitcher_id"],int(s),reg=V7.MRC_REG))
        base=pd.concat([base, V7.tm_attach(sub, V7.build_profile(tm[tm.season<s]), pid2tid)],axis=1)
        base=pd.concat([base, V7.plt_attach(sub, V7.build_state(hist[plt_cols]), False)],axis=1)
        parts.append(base)
    X=pd.concat(parts).reindex(df.index); del parts; gc.collect()
    X["pid_raw"]=df["pitcher_id"].to_numpy(np.float32)      # raw 선수 ID (exp168 게이트 통과)
    X["bid_raw"]=df["batter_id"].to_numpy(np.float32)
    cats=[c for c in V7.CATEGORICAL if c in X.columns]
    feat_names=list(X.columns)
    mono=[V7.MONO.get(c,0) for c in feat_names]
    age=df.season.max()-df.season.values
    w=np.power(2.0,-age/2.0); w=w/w.mean()
    for c in cats: X[c]=X[c].astype("int32").astype("str")
    print(f"피처 {len(feat_names)}개, 학습 {len(X):,}행 ({time.time()-t0:.0f}s)",flush=True)
    # 배포 테이블 (전체 기준)
    state=V7.build_carry_state(full)
    state["pitcher"].to_csv(OUT/"carry_pitcher.csv",index_label="pitcher_id")
    state["batter"].to_csv(OUT/"carry_batter.csv",index_label="batter_id")
    state["role"].to_csv(OUT/"carry_role.csv",index_label="pitcher_id")
    V7.build_projection(full[label_cols],["pitcher_id"],TARGET_YEAR,reg=V7.MRC_REG)\
        .to_csv(OUT/"marcel_pitcher.csv",index_label="pitcher_id")
    V7.save_state(V7.build_state(full[plt_cols]),str(OUT))
    V7.build_profile(tm).to_csv(OUT/"tm_profile.csv",index_label="trackman_id")
    pd.Series(pid2tid,name="trackman_id").to_csv(OUT/"pitcher_map.csv",index_label="pitcher_id")
    pool=Pool(X,y,weight=w,cat_features=cats)
    members=[]
    for spec in V7.ENSEMBLE:
        for seed in SEEDS:
            t1=time.time(); nm,dep,it,lr,_,_=spec
            m=V7.make_model((nm,dep,it,lr,L2,RSM),seed,thread_count=8)
            m.set_params(monotone_constraints=mono)
            m.fit(pool)
            name=f"cat_{nm}_{seed}"; m.save_model(str(OUT/f"{name}.cbm")); members.append(name)
            print(f"  {name} ({time.time()-t1:.0f}s)",flush=True)
            del m; gc.collect()
    rates=full.groupby("season")[TARGET].mean()
    r_hat=V7.estimate_base_rate(rates.index.values.astype(float),rates.values,TARGET_YEAR)
    json.dump({"features":feat_names,"cat_features":cats,"members":members,
               "season_rates":{int(k):float(v) for k,v in rates.items()},"r_hat":r_hat},
              open(OUT/"meta.json","w"),indent=1)
    print(f"완료 {len(members)}개 모델, r_hat={r_hat:.6f}, {time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()

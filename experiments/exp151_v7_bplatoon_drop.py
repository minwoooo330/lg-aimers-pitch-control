# -*- coding: utf-8 -*-
"""exp151 — v7측에 팀원이 만들고도 v7에서 안 쓴 두 플래그를 켠다.
   ① bplatoon: 타자x투수좌우 플래툰 9피처
   ② drop-pre-regime-f: 2022 이하 퓨처스 제외 (라벨 체제가 다른 죽은 분포)
   나머지는 v7과 동일(halflife=2, legacy-norm, trackman, mono, l2=1000, rsm=0.6, 1시드).
   판정: eval_oct.py로 병합 베이스의 v7 슬롯을 이걸로 바꿨을 때의 개선."""
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
        df=df[~((df.season<=2022)&(df.game_type=="F"))].reset_index(drop=True)  # drop-pre-regime-f
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
            base=pd.concat([base, V7.plt_attach(sub, V7.build_state(hist[plt_cols]), True)],axis=1)
            parts.append(base)
        X=pd.concat(parts).reindex(df.index)
        # --- 검증 피처: hist = 검증연도 미만 전체 ---
        hist=full[full.season<year]
        bv=V7.build_all(va, V7.build_carry_state(hist),
                        V7.build_projection(hist[label_cols],["pitcher_id"],int(year),reg=V7.MRC_REG))
        bv=pd.concat([bv, V7.tm_attach(va, V7.build_profile(tm[tm.season<year]), pid2tid)],axis=1)
        bv=pd.concat([bv, V7.plt_attach(va, V7.build_state(hist[plt_cols]), True)],axis=1)
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
        preds=[]
        for spec in V7.ENSEMBLE:
            t1=time.time()
            nm,dep,it,lr,_,_=spec
            m=V7.make_model((nm,dep,it,lr,L2,RSM),SEED,thread_count=8)
            m.set_params(monotone_constraints=mono)
            m.fit(pool)
            preds.append(m.predict_proba(Xv)[:,1])
            print(f"  cat_{nm} ({time.time()-t1:.0f}s)",flush=True)
            del m; gc.collect()
        p=np.mean(preds,axis=0)
        rel=((va.game_type=="R")|(year>=2023)).to_numpy()
        z=p[rel]+(yv[rel].mean()-p[rel].mean())
        print(f"[{year}] v7 재현 Brier(rel,정렬) {float(np.mean((z-yv[rel])**2)):.8f}  누적 {time.time()-t0:.0f}s",flush=True)
        parts_oof.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yv,"prediction":p}))
        pd.concat(parts_oof,ignore_index=True).to_csv(RES/"exp151_v7bd_oof.csv.gz",index=False,compression="gzip")
        del X,Xv,pool,parts; gc.collect()
    print(f"완료 {time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()

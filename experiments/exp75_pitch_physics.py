# -*- coding: utf-8 -*-
"""실험 75: 투구 물리 soft-feature — 특권교사(exp26 +12.7e-4)의 예측 가능 몫 회수.

exp18/19는 구종군 3분류(이산)까지만 하고 접었다. 구속·무브먼트의 '연속값'을
투구 직전 정보만으로 예측해 soft feature로 주입하는 것은 미시도다.

1단계(교사 학습): 정렬 775K행에서 fold 이전 시즌만 써서
   rel_speed, total_move = hypot(ivb, hb) 를 회귀로 예측.
   보고 지표는 전체 R²와 '투수 내(within-pitcher) R²' 둘 다.
   투수 평균 구속은 이미 pitcher_id/asof가 담고 있으므로, 새 정보는 within 몫뿐이다.

2단계(주입): 교사는 투구 직전 정보만 쓰므로 정렬되지 않은 행에도 적용 가능하다.
   전 행에 p_speed, p_move와 투수 기준선 대비 편차를 붙여 HGB 2024 fold 비교.

게이트: 2024 Brier 개선 >= 1e-5 여야 3 fold로 확장한다.
"""
from pathlib import Path
import gc, time, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features
from trackman_features import match_exact_games

sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent; DATA=HERE/"data"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
PRM=dict(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,
         l2_regularization=1.,early_stopping=False,random_state=42)
RPRM=dict(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,
          l2_regularization=1.,early_stopping=False,random_state=42)
TMC=["trackman_id","season","trackman_game_id","pitch_no","inning","top_bottom",
     "balls_before","strikes_before","outs_before","pitcher_trackman_id",
     "pitcher_hand","batter_hand","rel_speed","induced_vert_break","horz_break",
     "spin_rate","extension"]
FOLD=2024

def align(main_g, tm_s, matches):
    mi=main_g.groupby("_game_idx",sort=False).indices
    ti=tm_s.groupby("trackman_game_id",sort=False).indices
    ids=[]; sp=[]; iv=[]; hb=[]
    for row in matches.itertuples(index=False):
        a,b=mi[row.main_game_idx], ti[row.trackman_game_id]
        if len(a)!=len(b): continue
        ids.append(main_g[ID].to_numpy()[a])
        sp.append(tm_s["rel_speed"].to_numpy()[b])
        iv.append(tm_s["induced_vert_break"].to_numpy()[b])
        hb.append(tm_s["horz_break"].to_numpy()[b])
    ids=np.concatenate(ids); sp=np.concatenate(sp)
    iv=np.concatenate(iv); hb=np.concatenate(hb)
    return pd.DataFrame({ID:ids,"rel_speed":sp,"move":np.hypot(iv,hb)})

def main():
    t0=time.time()
    df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TMC,encoding="utf-8-sig")
    main_g,tm_s,matches=match_exact_games(df,tm)
    del tm; gc.collect()
    al=align(main_g,tm_s,matches)
    del main_g,tm_s,matches; gc.collect()
    print(f"정렬 {len(al)}행  ({time.time()-t0:.0f}s)",flush=True)

    tr=df[df.season<FOLD].reset_index(drop=True)
    va=df[df.season==FOLD].reset_index(drop=True)
    cols=[c for c in df.columns if c not in (ID,TARGET)]
    maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CATS}
    def enc(d):
        x=d[cols].copy()
        for c in CATS: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
        return pd.concat([x,add_features(d)],axis=1)
    Xtr,Xva=enc(tr),enc(va)
    cat_mask=[c in CATS for c in Xtr.columns]
    print(f"피처 {Xtr.shape[1]}개  ({time.time()-t0:.0f}s)",flush=True)

    # ---- 1단계: 물리 교사 ----
    a_tr=al.merge(tr[[ID,"pitcher_id"]],on=ID,how="inner")
    a_va=al.merge(va[[ID,"pitcher_id"]],on=ID,how="inner")
    itr=pd.Series(np.arange(len(tr)),index=tr[ID]).reindex(a_tr[ID]).to_numpy()
    iva=pd.Series(np.arange(len(va)),index=va[ID]).reindex(a_va[ID]).to_numpy()
    print(f"교사 학습 {len(a_tr)}행 / 평가 {len(a_va)}행",flush=True)
    preds={}
    for tgt in ["rel_speed","move"]:
        mt=np.isfinite(a_tr[tgt].to_numpy()); mv=np.isfinite(a_va[tgt].to_numpy())
        print(f"[{tgt}] 결측제거 후 학습 {int(mt.sum())} / 평가 {int(mv.sum())}",flush=True)
        m=HistGradientBoostingRegressor(**RPRM,categorical_features=cat_mask)
        m.fit(Xtr.iloc[itr[mt]],a_tr[tgt].to_numpy()[mt])
        a_va=a_va[mv].reset_index(drop=True); iva=iva[mv]
        pv=m.predict(Xva.iloc[iva]); yv=a_va[tgt].to_numpy()
        ss=1-np.mean((yv-pv)**2)/np.var(yv)
        g=a_va.groupby("pitcher_id")
        ym=g[tgt].transform("mean").to_numpy()
        pm=pd.Series(pv,index=a_va.index).groupby(a_va.pitcher_id).transform("mean").to_numpy()
        wr=np.corrcoef(yv-ym,pv-pm)[0,1]
        wss=1-np.mean(((yv-ym)-(pv-pm))**2)/np.var(yv-ym)
        print(f"[교사 {tgt}] 전체R2={ss:.4f}  투수내 상관={wr:.4f}  투수내R2={wss:.4f}  "
              f"({time.time()-t0:.0f}s)",flush=True)
        preds[tgt]=(m,)
    # 전 행 예측
    ex_tr=pd.DataFrame(index=Xtr.index); ex_va=pd.DataFrame(index=Xva.index)
    for tgt in ["rel_speed","move"]:
        m=preds[tgt][0]
        ptr=m.predict(Xtr); pva=m.predict(Xva)
        ex_tr[f"phys_{tgt}"]=ptr; ex_va[f"phys_{tgt}"]=pva
        base=pd.Series(ptr).groupby(tr.pitcher_id.to_numpy()).mean()
        ex_tr[f"physdev_{tgt}"]=ptr-tr.pitcher_id.map(base).to_numpy()
        ex_va[f"physdev_{tgt}"]=pva-va.pitcher_id.map(base).to_numpy()
    gc.collect()
    print(f"전 행 물리 예측 완료  ({time.time()-t0:.0f}s)",flush=True)

    # ---- 2단계: 주입 후 2024 게이트 ----
    ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    rows=[]
    for name,(xa,xb) in {"base":(Xtr,Xva),
                         "phys":(pd.concat([Xtr,ex_tr],axis=1),pd.concat([Xva,ex_va],axis=1))}.items():
        m=HistGradientBoostingClassifier(**PRM,
            categorical_features=[c in CATS for c in xa.columns]).fit(xa,ytr)
        p=m.predict_proba(xb)[:,1]
        rows.append({"variant":name,"n_feat":xa.shape[1],"brier":brier_score_loss(yva,p),
                     "auc":roc_auc_score(yva,p)})
        print(rows[-1],flush=True)
        pd.DataFrame({ID:va[ID].to_numpy(),"season":FOLD,TARGET:yva,"prediction":p}).to_csv(
            RES/f"exp75_{name}_2024_oof.csv.gz",index=False,compression="gzip")
        del m,p; gc.collect()
    r=pd.DataFrame(rows); r.to_csv(RES/"exp75_pitch_physics.csv",index=False,encoding="utf-8-sig")
    d=(r.brier.iloc[0]-r.brier.iloc[1])*1e5
    print(f"\n게이트: 2024 개선 {d:+.3f}e-5  -> {'통과' if d>=1.0 else '기각'}")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

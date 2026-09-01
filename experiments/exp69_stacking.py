# -*- coding: utf-8 -*-
"""실험 69: 스태킹/게이팅 메타모델 (미탐색 방향 C).

상수 가중치 대신 행별 문맥(투수 표본수, 리그, 손잡이, 카운트 우열)에 따라
멤버 가중을 바꾸는 메타모델을 시간 안전 프로토콜로 검증한다.
  - 2024 평가: 메타를 2022+2023 OOF로 학습
  - 2023 평가: 메타를 2022 OOF로 학습
비교 대상: 현 챔피언(고정 가중 체인). 채점은 2025 유형 행(1군 전부 + 2023이후 F).
학습 없음(저장된 OOF만 사용), 소요 몇 분.
"""
from pathlib import Path
import time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
F={"hgb_domain":"exp04_hgb_domain_oof.csv.gz","cat_domain":"exp05_catboost_domain_oof.csv.gz",
 "tm_mean":"exp08_tm_pitch_means_oof.csv.gz","league_role":"exp09_tm_league_role_oof.csv.gz",
 "cat_time":"exp15_catboost_chronological_oof.csv.gz","lgbm":"exp23_seoyeon_lgbm_oof.csv.gz",
 "nn3":"exp37_nn_seedavg_oof.csv.gz","hand3":"exp55_hand_seedavg_oof.csv.gz",
 "pbhand3":"exp59_pbhand_seedavg_oof.csv.gz","hc3":"exp63_handcnt_oof.csv.gz"}
def main():
    t0=time.time(); b=None
    for k,v in F.items():
        d=pd.read_csv(RES/v)
        if b is None: b=d[["row_id","season","control_success"]].copy()
        b[k]=b.row_id.map(d.set_index("row_id")["prediction"])
    s5=pd.read_csv(RES/"exp58_hand_seeds5_oof.csv.gz"); sc5=[c for c in s5.columns if c.startswith("p")]
    b["hand5"]=b.row_id.map(s5.set_index("row_id")[sc5].mean(axis=1))
    b["hand8"]=(3*b.hand3+5*b.hand5)/8
    mm=pd.read_csv(HERE/"data"/"train.csv",usecols=["row_id","game_type","asof_pitcher_n",
        "pitcher_hand","batter_hand","balls_before","strikes_before"],encoding="utf-8-sig")
    b=b.merge(mm,on="row_id",how="left")
    y=b.control_success.to_numpy(); season=b.season.to_numpy()
    rel=(b.game_type=="R").to_numpy()|(season>=2023)
    W=[0.06726666,0.11952455,0.15582993,0.17546554,0.23856908,0.24334425]
    c6=["hgb_domain","cat_domain","tm_mean","league_role","cat_time","lgbm"]
    e7=0.85*sum(w*b[c].to_numpy() for w,c in zip(W,c6))+0.15*b.nn3.to_numpy()
    e=0.65*e7+0.25*b.hand8.to_numpy()+0.10*b.pbhand3.to_numpy()
    champ=0.90*e+0.10*b.hc3.to_numpy()
    members=["hgb_domain","cat_domain","tm_mean","league_role","cat_time","lgbm",
             "nn3","hand8","pbhand3","hc3"]
    P=b[members].to_numpy()
    ctx=np.c_[np.log1p(b.asof_pitcher_n.fillna(0)),
              (b.game_type=="F").astype(float),
              (b.pitcher_hand==b.batter_hand).astype(float),
              np.sign(b.strikes_before-b.balls_before)]
    def wf_shift(p):
        bi={yr:(y[season==yr].mean()-p[season==yr].mean()) for yr in [2022,2023,2024]}
        q=p.copy(); q[season==2023]+=bi[2022]; q[season==2024]+=(bi[2022]+bi[2023])/2
        return q
    qch=wf_shift(champ)
    def br(p,yr): m=(season==yr)&rel; return np.mean((p[m]-y[m])**2)
    print("기준(챔피언): 2023 %.6f | 2024 %.6f"%(br(qch,2023),br(qch,2024)),flush=True)
    for name in ["logit","hgb"]:
        preds=champ.copy()
        for target in [2023,2024]:
            tr=season<target; te=season==target
            Xtr=np.c_[P[tr],ctx[tr]]; Xte=np.c_[P[te],ctx[te]]
            if name=="logit":
                mdl=LogisticRegression(max_iter=1000,C=1.0).fit(Xtr,y[tr])
            else:
                mdl=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.05,
                    max_leaf_nodes=15,min_samples_leaf=500,l2_regularization=10.0,
                    early_stopping=False,random_state=42).fit(Xtr,y[tr])
            preds[te]=mdl.predict_proba(Xte)[:,1]
        qm=wf_shift(preds)
        print(f"[{name} meta] 2023 {br(qm,2023):.6f} (Δ {(br(qch,2023)-br(qm,2023))*1e5:+.2f}e-5) | "
              f"2024 {br(qm,2024):.6f} (Δ {(br(qch,2024)-br(qm,2024))*1e5:+.2f}e-5)",flush=True)
    # 절충: 메타 출력과 챔피언의 50:50
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

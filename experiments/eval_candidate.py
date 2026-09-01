# -*- coding: utf-8 -*-
"""후보 OOF를 현 챔피언(hand8pb+hc) 체인에 10% 추가했을 때의 기여를 자동 채점.
   채점 규칙: 2025에 존재하는 유형 행만(1군 전부 + 2023년 이후 퓨처스), 판정은 2024 fold.
   사용법: python eval_candidate.py <oof파일명> [예측컬럼명]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
F={"hgb_domain":"exp04_hgb_domain_oof.csv.gz","cat_domain":"exp05_catboost_domain_oof.csv.gz",
 "tm_mean":"exp08_tm_pitch_means_oof.csv.gz","league_role":"exp09_tm_league_role_oof.csv.gz",
 "cat_time":"exp15_catboost_chronological_oof.csv.gz","lgbm":"exp23_seoyeon_lgbm_oof.csv.gz",
 "nn3":"exp37_nn_seedavg_oof.csv.gz","hand3":"exp55_hand_seedavg_oof.csv.gz",
 "pbhand3":"exp59_pbhand_seedavg_oof.csv.gz","hc3":"exp63_handcnt_oof.csv.gz"}
def load():
    b=None
    for k,v in F.items():
        d=pd.read_csv(RES/v)
        if b is None: b=d[["row_id","season","control_success"]].copy()
        b[k]=b.row_id.map(d.set_index("row_id")["prediction"])
    s5=pd.read_csv(RES/"exp58_hand_seeds5_oof.csv.gz"); sc=[c for c in s5.columns if c.startswith("p")]
    b["hand5"]=b.row_id.map(s5.set_index("row_id")[sc].mean(axis=1))
    b["hand8"]=(3*b.hand3+5*b.hand5)/8
    gt=b.row_id.map(pd.read_csv(HERE/"data"/"train.csv",usecols=["row_id","game_type"],
        encoding="utf-8-sig").set_index("row_id")["game_type"]).to_numpy()
    return b,gt
def champ(b):
    W=[0.06726666,0.11952455,0.15582993,0.17546554,0.23856908,0.24334425]
    c6=["hgb_domain","cat_domain","tm_mean","league_role","cat_time","lgbm"]
    e7=0.85*sum(w*b[c].to_numpy() for w,c in zip(W,c6))+0.15*b.nn3.to_numpy()
    e=0.65*e7+0.25*b.hand8.to_numpy()+0.10*b.pbhand3.to_numpy()
    return 0.90*e+0.10*b.hc3.to_numpy()
def score(p,y,season,rel):
    bi={yr:(y[season==yr].mean()-p[season==yr].mean()) for yr in [2022,2023,2024]}
    q=p.copy(); q[season==2023]+=bi[2022]; q[season==2024]+=(bi[2022]+bi[2023])/2
    return {yr:np.mean((q[(season==yr)&rel]-y[(season==yr)&rel])**2) for yr in [2022,2023,2024]}, float(np.mean(list(bi.values())))
def main(fname,col="prediction"):
    b,gt=load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
    rel=(gt=="R")|(season>=2023)
    d=pd.read_csv(RES/fname)
    if col not in d.columns:
        cand=[c for c in d.columns if c not in ("row_id","season","control_success")]
        col=cand[0]
    b["cand"]=b.row_id.map(d.set_index("row_id")[col])
    if b.cand.isna().any():
        print(f"  ! 결측 {b.cand.isna().sum()}행 -> 채점 불가"); return
    base=champ(b); s0,_=score(base,y,season,rel)
    print(f"[{fname}] 상관(챔피언) {np.corrcoef(base,b.cand)[0,1]:.4f}")
    for f in [0.05,0.10,0.15]:
        s,sh=score((1-f)*base+f*b.cand.to_numpy(),y,season,rel)
        print(f"   {int(f*100):>3}%  Δ2024 {(s0[2024]-s[2024])*1e5:+7.3f}  "
              f"Δ2022 {(s0[2022]-s[2022])*1e5:+7.3f}  Δ2023 {(s0[2023]-s[2023])*1e5:+7.3f}  shift {sh:.12f}")
if __name__=="__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "prediction")

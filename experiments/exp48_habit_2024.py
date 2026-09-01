# -*- coding: utf-8 -*-
"""실험 48: v18 위기상황 습관 피처 5개 검증 (2024/2023 fold)."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features
from trackman_features import match_exact_games, build_pitcher_mapping

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
HAB=["tm_fb_rate_2strike","tm_fb_rate_3ball","tm_fb_rate_normal",
     "tm_fb_shift_2strike","tm_fb_shift_3ball"]
TM_COLS=["season","trackman_game_id","pitch_no","inning","top_bottom","balls_before",
         "strikes_before","outs_before","pitcher_trackman_id","pitcher_hand","batter_hand",
         "pitch_type_group"]
MATCH=["season","inning","top_bottom","balls_before","strikes_before","outs_before",
       "pitcher_id","pitcher_hand","batter_hand"]

def habit(tm,mapping,cutoff):
    t=tm[tm.season<cutoff].copy()
    t["s2"]=t.strikes_before==2; t["b3"]=t.balls_before==3
    t["fb"]=t.pitch_type_group=="fastball"
    def rate(mask):
        sub=t[mask]
        return sub.groupby("pitcher_trackman_id").fb.mean()
    p=pd.DataFrame({"tm_fb_rate_2strike":rate(t.s2),
                    "tm_fb_rate_3ball":rate(t.b3),
                    "tm_fb_rate_normal":rate(~t.s2&~t.b3)})
    p["tm_fb_shift_2strike"]=p.tm_fb_rate_2strike-p.tm_fb_rate_normal
    p["tm_fb_shift_3ball"]=p.tm_fb_rate_3ball-p.tm_fb_rate_normal
    return p.join(mapping.set_index("pitcher_trackman_id")["pitcher_id"],how="inner").set_index("pitcher_id")

def main():
    t0=time.time()
    df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
    mg,ts,matches=match_exact_games(df[MATCH],tm)
    base=[c for c in df.columns if c not in (ID,TARGET)]
    rows=[]
    for year in [2024,2023]:
        mp=build_pitcher_mapping(mg,ts,matches,year); prof=habit(tm,mp,year)
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        a,b=tr[base].copy(),va[base].copy()
        for c in CATS:
            v=sorted(tr[c].dropna().astype(str).unique()); m={x:i for i,x in enumerate(v)}
            a[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
            b[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
        a=pd.concat([a,add_features(tr)],axis=1); b=pd.concat([b,add_features(va)],axis=1)
        for lbl,use in [("기준",False),("습관5개",True)]:
            x1,x2=a.copy(),b.copy()
            if use:
                for f in HAB:
                    x1[f]=tr[["pitcher_id"]].join(prof[f],on="pitcher_id")[f].to_numpy(np.float32)
                    x2[f]=va[["pitcher_id"]].join(prof[f],on="pitcher_id")[f].to_numpy(np.float32)
            m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                categorical_features=[c in CATS for c in x1.columns],random_state=42).fit(x1,ytr)
            p=m.predict_proba(x2)[:,1]
            rows.append({"fold":year,"구성":lbl,"brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p)})
            print(rows[-1],flush=True); del m,p,x1,x2; gc.collect()
        del tr,va,a,b,prof; gc.collect()
    r=pd.DataFrame(rows); r.to_csv(RES/"exp48_habit_2024.csv",index=False,encoding="utf-8-sig")
    for yr in [2024,2023]:
        s=r[r.fold==yr]
        print(f"{yr}: {s.brier.iloc[0]:.6f} -> {s.brier.iloc[1]:.6f} ({(s.brier.iloc[0]-s.brier.iloc[1])/1e-5:+.2f}e-5)")
    print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

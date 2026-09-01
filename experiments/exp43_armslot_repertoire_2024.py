# -*- coding: utf-8 -*-
"""실험 43: 팔각도x좌우매치업(A) + 세분구종 7개(B) + 구종난이도편차(C) 묶음 평가."""
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
PT7=["Fastball","Sinker","Cutter","Slider","Curveball","ChangeUp","Splitter"]
TM_COLS=["season","trackman_game_id","pitch_no","inning","top_bottom","balls_before","strikes_before",
         "outs_before","pitcher_trackman_id","pitcher_hand","batter_hand","tagged_pitch_type",
         "pitch_type_group","rel_height","rel_side"]
MATCH=["season","inning","top_bottom","balls_before","strikes_before","outs_before",
       "pitcher_id","pitcher_hand","batter_hand"]

def build_profile(tm,mapping,cutoff):
    t=tm[tm.season<cutoff]
    fb=t[t.pitch_type_group=="fastball"]
    g=fb.groupby("pitcher_trackman_id").agg(rel_h=("rel_height","mean"),rel_s=("rel_side","mean"),
                                            hand=("pitcher_hand","first"),n=("rel_height","size"))
    g=g[g.n>=30]
    g["rel_s_aligned"]=g.rel_s*g.hand.map({"Right":1.0,"Left":-1.0})
    g["rel_h_z"]=(g.rel_h-g.rel_h.mean())/g.rel_h.std()
    g["rel_s_z"]=(g.rel_s_aligned-g.rel_s_aligned.mean())/g.rel_s_aligned.std()
    # 세분 구종 비중
    tt=t[t.tagged_pitch_type.isin(PT7)]
    mix=(tt.groupby(["pitcher_trackman_id","tagged_pitch_type"]).size()
           .unstack(fill_value=0))
    mix=mix.reindex(columns=PT7,fill_value=0)
    mix=mix.div(mix.sum(axis=1).replace(0,np.nan),axis=0)
    mix.columns=[f"pt_{c.lower()}" for c in PT7]
    prof=g[["rel_h_z","rel_s_z"]].join(mix,how="outer")
    # C) 구종 난이도 편차용: 구종군별 리그 제구율은 메인에서 계산 불가 -> 구종 다양성 대용
    return prof.join(mapping.set_index("pitcher_trackman_id")["pitcher_id"],how="inner").set_index("pitcher_id")

def attach(frame,prof):
    b=frame[["pitcher_id"]].join(prof,on="pitcher_id").drop(columns="pitcher_id")
    same=(frame.pitcher_hand.to_numpy()==frame.batter_hand.to_numpy())
    out=pd.DataFrame(index=frame.index)
    out["armslot_vs_opposite"]=b.rel_h_z.to_numpy()*(~same)
    out["armslot_vs_same"]=b.rel_h_z.to_numpy()*same
    out["relside_vs_opposite"]=b.rel_s_z.to_numpy()*(~same)
    out["relside_vs_same"]=b.rel_s_z.to_numpy()*same
    for c in [f"pt_{p.lower()}" for p in PT7]:
        out[c]=b[c].to_numpy()
    out["tm_prof_avail"]=b.rel_h_z.notna().astype(np.int8).to_numpy()
    return out.astype(np.float32)

def main():
    t0=time.time()
    df=pd.read_csv(DATA/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(DATA/"trackman_history.csv",usecols=TM_COLS,encoding="utf-8-sig")
    mg,ts,matches=match_exact_games(df[MATCH],tm)
    base=[c for c in df.columns if c not in (ID,TARGET)]
    rows=[]
    for year in [2024,2023]:
        mp=build_pitcher_mapping(mg,ts,matches,year)
        prof=build_profile(tm,mp,year)
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        a,b=tr[base].copy(),va[base].copy()
        for c in CATS:
            vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
            a[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
            b[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
        a=pd.concat([a,add_features(tr)],axis=1); b=pd.concat([b,add_features(va)],axis=1)
        for lbl,use in [("기준",False),("A+B 추가",True)]:
            x1,x2=(pd.concat([a,attach(tr,prof)],axis=1),pd.concat([b,attach(va,prof)],axis=1)) if use else (a,b)
            m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                categorical_features=[c in CATS for c in x1.columns],random_state=42).fit(x1,ytr)
            p=m.predict_proba(x2)[:,1]
            rows.append({"fold":year,"구성":lbl,"n_feat":x1.shape[1],
                         "brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p)})
            print(rows[-1],flush=True)
            del m,p,x1,x2; gc.collect()
        del tr,va,a,b,prof,mp; gc.collect()
    r=pd.DataFrame(rows); r.to_csv(RES/"exp43_armslot_2024.csv",index=False,encoding="utf-8-sig")
    for yr in [2024,2023]:
        s=r[r.fold==yr]
        print(f"\n{yr}: {s.brier.iloc[0]:.6f} -> {s.brier.iloc[1]:.6f} ({(s.brier.iloc[0]-s.brier.iloc[1])/1e-5:+.2f}e-5)")
    print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

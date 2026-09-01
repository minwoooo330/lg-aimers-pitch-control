# -*- coding: utf-8 -*-
"""실험 34: CatBoost 반복횟수를 3개 fold 전부에서, 300회 이하 구간까지 정밀 측정.
   기본형과 시간순형 둘 다. 각 지점 예측을 저장해 나중에 앙상블 재계산에 쓴다."""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT_ALL=["top_bottom","game_type","base_state","pitcher_hand","batter_hand",
         "pitcher_id","batter_id","pitcher_team_id","batter_team_id"]
CHECKS=[25,50,75,100,125,150,200,250,300,400]
MAXIT=400

def main():
    start=time.time()
    df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    rows=[]
    for year in [2022,2023,2024]:
        tr=df[df.season<year]; va=df[df.season==year]
        xtr=pd.concat([tr[base].copy(),add_features(tr)],axis=1)
        xva=pd.concat([va[base].copy(),add_features(va)],axis=1)
        for c in CAT_ALL:
            xtr[c]=xtr[c].where(xtr[c].notna(),"__MISSING__").astype(str)
            xva[c]=xva[c].where(xva[c].notna(),"__MISSING__").astype(str)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        idx=[xtr.columns.get_loc(c) for c in CAT_ALL]
        for variant,has_time in [("cat_domain",False),("cat_time",True)]:
            t0=time.time()
            m=CatBoostClassifier(iterations=MAXIT,learning_rate=0.06,depth=8,loss_function="Logloss",
                                 l2_leaf_reg=3.0,random_seed=42,thread_count=-1,
                                 allow_writing_files=False,verbose=False,has_time=has_time)
            m.fit(xtr,ytr,cat_features=idx)
            out={ID:va[ID].to_numpy(),"season":year,TARGET:yva}
            for n in CHECKS:
                p=m.predict_proba(xva,ntree_end=n)[:,1]
                out[f"pred_{n}"]=p
                rows.append({"variant":variant,"fold":year,"iterations":n,
                             "brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p)})
            pd.DataFrame(out).to_csv(RES/f"exp34_{variant}_{year}_preds.csv.gz",
                                     index=False,compression="gzip")
            best=min([r for r in rows if r["variant"]==variant and r["fold"]==year],
                     key=lambda r:r["brier"])
            cur=[r for r in rows if r["variant"]==variant and r["fold"]==year
                 and r["iterations"]==300][0]
            print(f"[{year}] {variant}: 최적 {best['iterations']}회 {best['brier']:.6f} | "
                  f"현재300회 {cur['brier']:.6f} | 차이 {(cur['brier']-best['brier'])/1e-5:+.1f}e-5 "
                  f"({time.time()-t0:.0f}s)",flush=True)
            del m; gc.collect()
        del tr,va,xtr,xva; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp34_catboost_iters_allfolds.csv",index=False,encoding="utf-8-sig")
    res=pd.DataFrame(rows)
    print("\n=== 반복횟수별 3개 fold 평균 ===")
    for variant in ["cat_domain","cat_time"]:
        sub=res[res.variant==variant].groupby("iterations").brier.mean()
        print(f"\n{variant}")
        for n,v in sub.items():
            mark=" <- 현재" if n==300 else (" <- 최적" if n==sub.idxmin() else "")
            print(f"  {n:4d}회: {v:.6f}{mark}")
    print(f"\ntotal={time.time()-start:.1f}s")

if __name__=="__main__": main()

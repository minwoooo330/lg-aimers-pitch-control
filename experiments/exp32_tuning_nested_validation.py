# -*- coding: utf-8 -*-
"""실험 32: 튜닝 절차가 처음 보는 연도에 일반화되는지 중첩 검증.
   24개 설정을 3개 fold 전부에서 평가 -> 두 해로 고르고 나머지 해로 시험."""
from pathlib import Path
import gc, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
KEYS=["learning_rate","max_iter","max_leaf_nodes","min_samples_leaf",
      "l2_regularization","max_features","max_bins"]
INTS={"max_iter","max_leaf_nodes","min_samples_leaf","max_bins"}
DEFAULT={"learning_rate":0.06,"max_iter":200,"max_leaf_nodes":31,
         "min_samples_leaf":200,"l2_regularization":1.0,"max_bins":255}

def encode(tr,va,cols):
    a,b=tr[cols].copy(),va[cols].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    return a,b

def main():
    start=time.time()
    scr=pd.read_csv(RES/"exp29_hgb_screen_2024.csv")
    configs=[]
    for r in scr.itertuples(index=False):
        p={k:(int(getattr(r,k)) if k in INTS else float(getattr(r,k))) for k in KEYS}
        configs.append((int(r.trial),p))
    print(f"설정 {len(configs)}개 x 2022/2023 평가 시작",flush=True)

    df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    rows=[]
    for year in [2022,2023]:
        tr=df[df.season<year]; va=df[df.season==year]
        xtr,xva=encode(tr,va,base)
        xtr=pd.concat([xtr,add_features(tr)],axis=1); xva=pd.concat([xva,add_features(va)],axis=1)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        mask=[c in CATS for c in xtr.columns]
        # 기본 설정도 같은 조건에서 측정
        t0=time.time()
        m=HistGradientBoostingClassifier(early_stopping=False,categorical_features=mask,
                                         random_state=42,**DEFAULT).fit(xtr,ytr)
        rows.append({"trial":-1,"fold":year,"brier":brier_score_loss(yva,m.predict_proba(xva)[:,1])})
        print(f"  [{year}] default brier={rows[-1]['brier']:.6f} ({time.time()-t0:.0f}s)",flush=True)
        del m; gc.collect()
        for tid,p in configs:
            t0=time.time()
            m=HistGradientBoostingClassifier(early_stopping=False,categorical_features=mask,
                                             random_state=42,**p).fit(xtr,ytr)
            br=brier_score_loss(yva,m.predict_proba(xva)[:,1])
            rows.append({"trial":tid,"fold":year,"brier":br})
            print(f"  [{year}] trial {tid:2d} brier={br:.6f} ({time.time()-t0:.0f}s)",flush=True)
            del m; gc.collect()
        del tr,va,xtr,xva; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp32_tuning_all_folds.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-start:.1f}s")

if __name__=="__main__": main()

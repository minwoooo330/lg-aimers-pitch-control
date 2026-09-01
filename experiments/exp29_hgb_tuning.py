# -*- coding: utf-8 -*-
"""실험 29: HGB 하이퍼파라미터 랜덤 서치. 2024로 1차 선별 후 상위만 3 fold 확인."""
from pathlib import Path
import gc, time, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
BASELINE={2022:0.243524,2023:0.253432,2024:0.248110,"mean":0.248355}
N_SCREEN=24; N_CONFIRM=5

def encode(tr,va,cols):
    a,b=tr[cols].copy(),va[cols].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    return a,b

def sample(rng):
    return {"learning_rate":float(np.exp(rng.uniform(np.log(0.015),np.log(0.12)))),
            "max_iter":int(rng.choice([200,300,400,600,800])),
            "max_leaf_nodes":int(rng.choice([15,31,63,127])),
            "min_samples_leaf":int(np.exp(rng.uniform(np.log(50),np.log(2000)))),
            "l2_regularization":float(np.exp(rng.uniform(np.log(0.1),np.log(50)))),
            "max_features":float(rng.uniform(0.6,1.0)),
            "max_bins":int(rng.choice([128,255]))}

def run(params,xtr,ytr,xva,yva,mask):
    m=HistGradientBoostingClassifier(early_stopping=False,categorical_features=mask,
                                     random_state=42,**params)
    m.fit(xtr,ytr); p=m.predict_proba(xva)[:,1]
    out=(brier_score_loss(yva,p),roc_auc_score(yva,p))
    del m,p; gc.collect(); return out

def main():
    start=time.time(); rng=np.random.RandomState(42)
    df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    cache={}
    for year in [2022,2023,2024]:
        tr=df[df.season<year]; va=df[df.season==year]
        a,b=encode(tr,va,base)
        cache[year]=(pd.concat([a,add_features(tr)],axis=1),tr[TARGET].to_numpy(np.int8),
                     pd.concat([b,add_features(va)],axis=1),va[TARGET].to_numpy(np.int8))
        del tr,va,a,b; gc.collect()
    mask=[c in CATS for c in cache[2024][0].columns]

    # 1단계: 2024 선별
    trials=[]
    xtr,ytr,xva,yva=cache[2024]
    for i in range(N_SCREEN):
        p=sample(rng); t0=time.time()
        br,auc=run(p,xtr,ytr,xva,yva,mask)
        trials.append({"trial":i,**p,"brier_2024":br,"auc_2024":auc,"seconds":time.time()-t0})
        print(f"[{i:2d}] brier={br:.6f} auc={auc:.5f} ({time.time()-t0:.0f}s) {json.dumps(p,default=float)}",flush=True)
    scr=pd.DataFrame(trials).sort_values("brier_2024").reset_index(drop=True)
    RES.mkdir(exist_ok=True)
    scr.to_csv(RES/"exp29_hgb_screen_2024.csv",index=False,encoding="utf-8-sig")
    print("\n기준선 2024:",BASELINE[2024],"| 최고:",round(scr.brier_2024.iloc[0],6),flush=True)

    # 2단계: 상위 후보만 2022/2023 확인
    keys=["learning_rate","max_iter","max_leaf_nodes","min_samples_leaf",
          "l2_regularization","max_features","max_bins"]
    rows=[]
    for r_ in scr.head(N_CONFIRM).itertuples(index=False):
        p={k:getattr(r_,k) for k in keys}
        p={k:(int(v) if k in ("max_iter","max_leaf_nodes","min_samples_leaf","max_bins") else float(v))
           for k,v in p.items()}
        res={"trial":int(r_.trial),**p,"brier_2024":float(r_.brier_2024)}
        for year in [2022,2023]:
            a,yb,c,yd=cache[year]
            br,auc=run(p,a,yb,c,yd,mask)
            res[f"brier_{year}"]=br
        res["brier_mean"]=np.mean([res["brier_2022"],res["brier_2023"],res["brier_2024"]])
        rows.append(res)
        print(f"confirm trial {r_.trial}: 2022={res['brier_2022']:.6f} 2023={res['brier_2023']:.6f} "
              f"2024={res['brier_2024']:.6f} mean={res['brier_mean']:.6f}",flush=True)

    cf=pd.DataFrame(rows).sort_values("brier_mean")
    cf["gain_vs_baseline"]=BASELINE["mean"]-cf.brier_mean
    cf.to_csv(RES/"exp29_hgb_confirm.csv",index=False,encoding="utf-8-sig")
    print("\n기준선 평균:",BASELINE["mean"])
    print(cf[["trial","brier_2022","brier_2023","brier_2024","brier_mean","gain_vs_baseline"]].to_string(index=False))
    print(f"total={time.time()-start:.1f}s")

if __name__=="__main__": main()

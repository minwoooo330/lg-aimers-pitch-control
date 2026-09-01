# -*- coding: utf-8 -*-
"""실험 31: 같은 설정, 다른 시드 3개를 평균내면 좋아지는지 검증."""
from pathlib import Path
import gc, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
TUNED=json.load(open(HERE/"tuned_hgb_params.json"))
SEEDS=[42,7,2024]

def encode(tr,va,cols):
    a,b=tr[cols].copy(),va[cols].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    return a,b

def main():
    start=time.time()
    df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    rows=[]; oof_avg=[]
    for year in [2022,2023,2024]:
        tr=df[df.season<year]; va=df[df.season==year]
        xtr,xva=encode(tr,va,base)
        xtr=pd.concat([xtr,add_features(tr)],axis=1); xva=pd.concat([xva,add_features(va)],axis=1)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        mask=[c in CATS for c in xtr.columns]
        ps=[]
        for sd in SEEDS:
            t0=time.time()
            m=HistGradientBoostingClassifier(early_stopping=False,categorical_features=mask,
                                             random_state=sd,**TUNED).fit(xtr,ytr)
            p=m.predict_proba(xva)[:,1]; ps.append(p)
            rows.append({"fold":year,"seed":sd,"brier":brier_score_loss(yva,p),"sec":time.time()-t0})
            print(rows[-1],flush=True); del m; gc.collect()
        avg=np.mean(ps,axis=0)
        rows.append({"fold":year,"seed":"average","brier":brier_score_loss(yva,avg),"sec":0})
        print(rows[-1],flush=True)
        oof_avg.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":avg}))
        del tr,va,xtr,xva,ps; gc.collect()
    res=pd.DataFrame(rows); RES.mkdir(exist_ok=True)
    res.to_csv(RES/"exp31_seed_average.csv",index=False,encoding="utf-8-sig")
    pd.concat(oof_avg,ignore_index=True).to_csv(RES/"exp31_seedavg_domain_oof.csv.gz",index=False,compression="gzip")
    single=res[res.seed!="average"].groupby("fold").brier.mean()
    avg=res[res.seed=="average"].set_index("fold").brier
    print("\n단일 시드 평균 vs 3시드 평균")
    for f in [2022,2023,2024]:
        print(f"  {f}: {single[f]:.8f} -> {avg[f]:.8f}  ({(single[f]-avg[f])/1e-5:+.2f}e-5)")
    print(f"  전체: {single.mean():.8f} -> {avg.mean():.8f}  ({(single.mean()-avg.mean())/1e-5:+.2f}e-5)")
    print(f"total={time.time()-start:.1f}s")

if __name__=="__main__": main()

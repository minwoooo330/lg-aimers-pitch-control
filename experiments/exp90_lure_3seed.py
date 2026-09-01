# -*- coding: utf-8 -*-
"""실험 90: 유인구 컨텍스트 배치(lurefeatures 12개) 3시드 판정 — '닫지 않는다'.
exp96의 단일시드 판정(2024 +2.99 / 2022 -3.32)은 노이즈 바닥(1시그마 3.4~4.0e-5) 안이라
기각도 채택도 증명된 게 아니다. 시드 42/7/2024 평균으로 노이즈를 1/sqrt(3)로 줄여 다시 잰다.
"""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features
from hfeatures import add_hfeatures
from lurefeatures import add_lure_features
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
SEEDS=[42,7,2024]
def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,hf):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            parts=[x,add_features(d)]
            parts.append(add_hfeatures(d))
            if hf: parts.append(add_lure_features(d))
            return pd.concat(parts,axis=1)
        for hf in [False,True]:
            xa,xb=enc(tr,hf),enc(va,hf)
            cm=[c in CAT for c in xa.columns]
            ps=[]
            for seed in SEEDS:
                m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                    min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                    random_state=seed,categorical_features=cm).fit(xa,ytr)
                p=m.predict_proba(xb)[:,1]; ps.append(p)
                rows.append({"fold":year,"hf":hf,"seed":seed,
                    "brier_rel":float(np.mean((p[rel]-yva[rel])**2))})
                print(rows[-1],flush=True); del m; gc.collect()
            avg=np.mean(ps,axis=0)
            rows.append({"fold":year,"hf":hf,"seed":"avg3",
                "brier_rel":float(np.mean((avg[rel]-yva[rel])**2))})
            print(rows[-1],flush=True)
            store[(year,hf)]=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":avg})
            del xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp90_lure_3seed.csv",index=False,encoding="utf-8-sig")
    for hf in [False,True]:
        pd.concat([store[(y,hf)] for y in [2024,2022] if (y,hf) in store],ignore_index=True)\
          .to_csv(RES/f"exp90_{'lure' if hf else 'base'}_avg3_oof.csv.gz",index=False,compression="gzip")
    r=pd.DataFrame(rows)
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.seed=="avg3")&(~r.hf)].brier_rel.iloc[0]
        b=r[(r.fold==year)&(r.seed=="avg3")&(r.hf)].brier_rel.iloc[0]
        print(f"[3시드 평균] {year}: base {a:.8f} -> hf {b:.8f}  delta {(a-b)*1e5:+.3f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""실험 89: 2스트라이크 전용 모델. exp87에서 2스트라이크 구간의 모델 실력이 절반 이하로
   확인됐다(0-2 상수대비 +73.7e-5 vs 3-1 +411.6e-5). 해당 구간은 실패의 23%가 '크게 벗어남'인데
   그 유형에 대응하는 공식 피처가 없다(exp49). 통합 모델이 유인구 국면을 평균내는지 확인한다.
   손잡이 분할 GBDT와 같은 '층화 전용 모델' 접근이며 카운트 임베딩 분할(exp53/54 기각)과 다르다."""
from pathlib import Path
import gc, time, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from features import add_features
from hfeatures import add_hfeatures
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
PRM=dict(max_iter=200,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=200,
         l2_regularization=1.,early_stopping=False,random_state=42)
def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; parts=[]
    for year in [2022,2023,2024]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CATS}
        def enc(d):
            x=d[cols].copy()
            for c in CATS: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            return pd.concat([x,add_features(d),add_hfeatures(d)],axis=1)
        for tag, sel in [("2strike", lambda d: d.strikes_before==2)]:
            trm=sel(tr).to_numpy(); vam=sel(va).to_numpy()
            xa=enc(tr[trm].reset_index(drop=True)); xb=enc(va[vam].reset_index(drop=True))
            ytr=tr[TARGET].to_numpy(np.int8)[trm]; yva=va[TARGET].to_numpy(np.int8)[vam]
            m=HistGradientBoostingClassifier(**PRM,
                categorical_features=[c in CATS for c in xa.columns]).fit(xa,ytr)
            p=m.predict_proba(xb)[:,1]
            rows.append({"fold":year,"tag":tag,"n_train":int(trm.sum()),"n_val":int(vam.sum()),
                         "brier":brier_score_loss(yva,p),"sec":round(time.time()-t0)})
            print(rows[-1],flush=True)
            parts.append(pd.DataFrame({ID:va[ID].to_numpy()[vam],"season":year,
                                       TARGET:yva,"prediction":p}))
            del m,xa,xb; gc.collect()
        del tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp89_2strike.csv",index=False,encoding="utf-8-sig")
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp89_2strike_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""실험 83: XGBoost 신규 멤버 (다양성 증가형). 체인에 XGB 계열이 전무하다.
   exp12에서 한 번 실패했으나 그때는 고카디널리티 범주형을 그대로 넣은 설정이었다.
   여기서는 체인 표준 전처리(범주 정수코드 + domain + hf)로 다시 만든다."""
from pathlib import Path
import gc, time, sys
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss
from features import add_features
from hfeatures import add_hfeatures
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store=[]
    for year in [2022,2023,2024]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CATS}
        def enc(d):
            x=d[cols].copy()
            for c in CATS: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            return pd.concat([x,add_features(d),add_hfeatures(d)],axis=1)
        xa,xb=enc(tr),enc(va)
        m=xgb.XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=7,
            min_child_weight=50,subsample=0.8,colsample_bytree=0.7,reg_lambda=2.0,
            tree_method="hist",max_bin=256,n_jobs=-1,random_state=42,eval_metric="logloss")
        m.fit(xa,ytr)
        p=m.predict_proba(xb)[:,1]
        rows.append({"fold":year,"model":"xgb_hf","brier":brier_score_loss(yva,p),"sec":round(time.time()-t0)})
        print(rows[-1],flush=True)
        store.append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}))
        del m,xa,xb,tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp83_xgb.csv",index=False,encoding="utf-8-sig")
        pd.concat(store,ignore_index=True).to_csv(RES/"exp83_xgb_hf_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

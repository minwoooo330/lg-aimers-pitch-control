# -*- coding: utf-8 -*-
"""exp156 — 배깅 계열(RandomForest / ExtraTrees) + sd. 현재 베이스에 없는 세 번째 계열.

  재개장 근거 3가지:
   ① exp47 기각은 sd 채널이 없던 시절 측정(단독 0.248281)
   ② 그때도 상관 0.9222로 v7(0.9658)보다 낮았다 — 다양성 자체는 가장 컸다
   ③ 현재 베이스는 CatBoost(55%)+NN(45%)로 배깅 계열이 전무하다
  배깅은 부스팅과 편향/분산 프로파일이 달라 '오차의 종류'가 다르다. 과적합이 확인된
  지금은 분산축소형 모델이 다른 역할을 할 수 있다.
  판정: eval_oct.py (베이스 추가 / nnsd 슬롯 혼합 두 경로)."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
sys.path.insert(0,str(HERE))
from features import add_features
import importlib.util as _u
_s=_u.spec_from_file_location("e120",HERE/"exp120_nn_sd.py")
E=_u.module_from_spec(_s); _s.loader.exec_module(E)
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    store={"rf":[],"et":[]}; rows=[]
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB=E.build_tables(df[df.season<year], year)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        relm=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            return pd.concat([x,add_features(d),E.add_sd(d,TAB)],axis=1).astype(np.float32)
        xa=enc(tr); xb=enc(va)
        med=xa.median(); xa=xa.fillna(med); xb=xb.fillna(med)
        print(f"[{year}] 학습 {xa.shape}  검증 {xb.shape}  ({time.time()-t0:.0f}s)",flush=True)
        for tag,M in [("rf",RandomForestClassifier),("et",ExtraTreesClassifier)]:
            tt=time.time()
            m=M(n_estimators=300,max_depth=18,min_samples_leaf=100,max_features=0.4,
                n_jobs=8,random_state=42,bootstrap=True)
            m.fit(xa,ytr); p=m.predict_proba(xb)[:,1]
            z=p[relm]+(yva[relm].mean()-p[relm].mean())
            br=float(np.mean((z-yva[relm])**2))
            rows.append({"fold":year,"model":tag,"brier":br,"sec":round(time.time()-tt)})
            print(f"  {tag} {br:.8f}  ({time.time()-tt:.0f}s)",flush=True)
            store[tag].append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}))
            pd.concat(store[tag],ignore_index=True).to_csv(RES/f"exp156_{tag}_oof.csv.gz",index=False,compression="gzip")
            pd.DataFrame(rows).to_csv(RES/"exp156_bagging.csv",index=False,encoding="utf-8-sig")
            del m; gc.collect()
        del xa,xb; gc.collect()
    print(f"total={time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()

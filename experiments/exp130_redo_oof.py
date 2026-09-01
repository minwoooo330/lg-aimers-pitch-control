# -*- coding: utf-8 -*-
"""exp130 — exp125(다중시즌)/exp128(sd x 상황) OOF 재생성. 체인 한계기여로 재판정하기 위함.

  [자기수정] 두 실험을 '단독 Brier 변화'로 기각한 것은 우리가 확정한 판정 규칙 위반이다.
  규칙: "후보는 반드시 체인 한계기여로 판정. 단일 모델 성능은 순서 정하기용일 뿐 기각 근거로 쓰지 않음"
  (exp105를 단독 성능으로 기각했다가 되살려 +13.53점을 낸 사례로 확정된 규칙)
  당시 OOF를 저장하지 않아 재판정이 불가능했으므로 예측을 다시 만든다."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from features import add_features
from hfeatures import add_hfeatures
import importlib.util
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
spec=importlib.util.spec_from_file_location("ms",HERE/"exp125_multiseason.py")
ms=importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
spec2=importlib.util.spec_from_file_location("sx",HERE/"exp128_sd_x_context.py")
sx=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(sx)
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    store={"multi":[], "ctx":[]}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB=ms.build_tables(df[df.season<year], year)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        relm=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,kind):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            sd=ms.add_sd(d,TAB)
            parts=[x,add_features(d),add_hfeatures(d),sd]
            if kind=="multi": parts.append(ms.add_multiseason(d,TAB))
            elif kind=="ctx": parts.append(sx.add_sd_x_context(d,sd))
            return pd.concat(parts,axis=1)
        for kind in ["multi","ctx"]:
            xa=enc(tr,kind); xb=enc(va,kind)
            tt=time.time()
            m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.0,max_bins=255,random_state=42)
            m.fit(xa,ytr); p=m.predict_proba(xb)[:,1]
            z=p[relm]+(yva[relm].mean()-p[relm].mean())
            print({"fold":year,"kind":kind,"n_feat":xa.shape[1],
                   "brier_rel_aligned":float(np.mean((z-yva[relm])**2)),"sec":round(time.time()-tt)},flush=True)
            store[kind].append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}))
            pd.concat(store[kind],ignore_index=True).to_csv(RES/f"exp130_{kind}_oof.csv.gz",index=False,compression="gzip")
            del m,xa,xb; gc.collect()
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

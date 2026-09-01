
import sys, importlib.util as _u
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); sys.path.insert(0,str(HERE))
sp=_u.spec_from_file_location("f7",HERE/"v7_pipeline.py"); V7=_u.module_from_spec(sp); sp.loader.exec_module(V7)
from features import add_features
from hfeatures import add_hfeatures
sm=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",nrows=40000)
f7=V7.build_features(sm); fo=add_features(sm); fh=add_hfeatures(sm)
print(f"v7 build_features {f7.shape[1]}개 / 우리 domain {fo.shape[1]}개 / 우리 heuristic {fh.shape[1]}개")
f7n=f7.select_dtypes(include=[np.number]).fillna(0)
f7n=f7n.loc[:,f7n.std()>0]
dup=[]; uniq=[]
for src_name,src_df in [("dom",fo),("heu",fh)]:
    for c in src_df.columns:
        v=src_df[c].astype(float).fillna(0)
        if v.std()==0: continue
        mx=float(np.max(np.abs(np.corrcoef(np.vstack([v.to_numpy()],),f7n.to_numpy().T)[0,1:])))
        (dup if mx>0.999 else uniq).append((f"{src_name}:{c}",round(mx,3)))
print(f"\nv7 피처로 사실상 복원됨(상관>0.999): {len(dup)}개")
print(f"v7에 없는 고유 피처: {len(uniq)}개")
uniq.sort(key=lambda x:x[1])
print("\n가장 고유한 15개 (v7 최대상관):")
for n_,m_ in uniq[:15]: print(f"   {n_:34s} {m_}")

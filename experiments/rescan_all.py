
import numpy as np, pandas as pd, glob, os
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    p=RES/fn if isinstance(fn,str) else fn
    try: d=pd.read_csv(p)
    except Exception: return None
    if ID not in d.columns: return None
    col="prediction" if "prediction" in d.columns else None
    if col is None:
        pc=[c for c in d.columns if c.startswith("p") and c not in (ID,"season",TARGET)
            and pd.api.types.is_numeric_dtype(d[c])]
        if not pc: return None
        d["prediction"]=d[pc].astype(float).mean(axis=1)
    s=d.set_index(ID)["prediction"]
    if not pd.api.types.is_numeric_dtype(s): return None
    return b[ID].map(s).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
nn=0.5*((3*a3+5*a5)/8)+0.5*b8
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig",usecols=["row_id","game_type","season"])
d=b.merge(tr,on="row_id",how="left",suffixes=("","_x"))
rel=((d.game_type=="R")|(d.season>=2023)).to_numpy()
y=d[TARGET].to_numpy(); season=d.season.to_numpy(); v7=d.v7.to_numpy()
EPS=1e-6
def lo(p): p=np.clip(p,EPS,1-EPS); return np.log(p/(1-p))
def sg(z): return 1/(1+np.exp(-z))
def sc_(p,s):
    l=lo(p); c=np.nanmean(l[np.isfinite(l)]); return sg(s*(l-c)+c)
base=sc_(0.55*v7+0.45*nn,1.08)
def paired(p_new,yr):
    m=(season==yr)&rel&np.isfinite(p_new)&np.isfinite(base)
    yy=y[m]
    a=base[m]+(yy.mean()-base[m].mean()); c=p_new[m]+(yy.mean()-p_new[m].mean())
    di=(a-yy)**2-(c-yy)**2
    return di.mean()*1e5, di.std(ddof=1)/np.sqrt(len(di))*1e5
skip={"exp148_v7_oof.csv.gz","exp120_nn_sd_oof.csv.gz","exp133_nnsd_seeds5_oof.csv.gz",
      "exp144_nnsd_seeds8b_oof.csv.gz"}
rows=[]
for f in sorted(glob.glob(str(RES/"*oof*.csv.gz"))):
    nm=os.path.basename(f)
    if nm in skip: continue
    v=g(Path(f))
    if v is None: continue
    cov=np.isfinite(v).mean()
    if cov<0.4: continue
    best=None
    for w in (0.05,0.10,0.15,0.20):
        q=base.copy(); mm=np.isfinite(v); q[mm]=(1-w)*base[mm]+w*v[mm]
        d24,s24=paired(q,2024); d22,s22=paired(q,2022)
        if d24>0 and d22>0:
            sc=min(d24/max(s24,1e-9), d22/max(s22,1e-9))
            if best is None or (d24+d22)>(best[1]+best[2]): best=(w,d24,d22,s24,s22,sc)
    if best:
        rows.append((nm,best[0],best[1],best[2],best[5]))
rows.sort(key=lambda r:-(r[2]+r[3]))
print(f"{'OOF 파일':44s} {'비중':>5s} {'2024Δ':>8s} {'2022Δ':>8s} {'최소σ':>7s}")
print("-"*82)
for nm,w,d24,d22,sc in rows:
    print(f"{nm:44s} {int(w*100):4d}% {d24:+8.3f} {d22:+8.3f} {sc:+7.1f}")
print(f"\n양쪽 fold 양수인 후보 {len(rows)}개")

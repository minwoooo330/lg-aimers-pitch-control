
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
ID,TARGET="row_id","control_success"
tr=pd.read_csv(HERE/"data"/"train.csv",usecols=[ID,"game_type"],encoding="utf-8-sig").set_index(ID)
print("=== exp168: v7 피처 + raw 선수ID 2컬럼 — 짝지은 판정 ===\n")
for yr in (2024,2022):
    d=pd.read_csv(RES/f"exp168_rawid_{yr}.csv.gz")
    d["game_type"]=d[ID].map(tr.game_type)
    d=d[(d.game_type=="R")|(d.season>=2023)]
    y=d[TARGET].to_numpy()
    a=d["no_id"].to_numpy(); b=d["raw_id"].to_numpy()
    qa=a+(y.mean()-a.mean()); qb=b+(y.mean()-b.mean())
    dd=(y-qa)**2-(y-qb)**2      # +면 raw_id 우세
    m=dd.mean()*1e5; se=dd.std(ddof=1)/np.sqrt(len(dd))*1e5
    print(f"[{yr}] no_id {np.mean((y-qa)**2):.8f}  raw_id {np.mean((y-qb)**2):.8f}")
    print(f"      raw_id 이득 {m:+7.3f}e-5  (짝지은 SE {se:.3f}, {m/se:+.1f}σ)   예측상관 {np.corrcoef(a,b)[0,1]:.4f}\n")

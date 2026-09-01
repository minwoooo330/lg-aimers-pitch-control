# -*- coding: utf-8 -*-
"""보유 재료 전수 스캔: 각 OOF를 챔피언에 10% 추가했을 때의 한계 기여를 전부 계산.
   exp104(Caruana)는 백지에서 조합을 만들었지만, 여기서는 '현 챔피언에 무엇을 더할까'를 직접 본다."""
import numpy as np, pandas as pd, sys, importlib.util, glob, os, zipfile
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
rel=(gt=="R")|(season>=2023)
def seedavg(fn):
    d=pd.read_csv(RES/fn); sc=[c for c in d.columns if c.startswith("p") and c!="prediction"]
    sc=sc if sc else ["prediction"]
    return d.set_index("row_id")[sc].mean(axis=1)
def col(fn):
    d=pd.read_csv(RES/fn); return b.row_id.map(d.set_index("row_id")["prediction"]).to_numpy()
bb={"ms":b.row_id.map(seedavg("exp68_multisplit_oof.csv.gz")).to_numpy(),
    "hc5":b.row_id.map(seedavg("exp65_handcnt_seeds5_oof.csv.gz")).to_numpy()}
bigv=b.row_id.map(pd.concat([seedavg("exp84_hand_big_oof.csv.gz"),seedavg("exp84b_2023_oof.csv.gz")])).to_numpy()
W6=[0.06726666,0.11952455,0.15582993,0.17546554,0.23856908,0.24334425]
C6=["hgb_domain","cat_domain","tm_mean","league_role","cat_time","lgbm"]
flat={c:0.80*0.90*0.65*0.85*w for w,c in zip(W6,C6)}
flat.update({"nn3":0.80*0.90*0.65*0.15,"hand8":0.80*0.90*0.25,"pbhand3":0.80*0.90*0.10,
             "hc3":0.80*0.10,"ms":0.10,"hc5":0.10})
V={k:(bb[k] if k in bb else b[k].to_numpy()) for k in flat}
for k,fn in {"hgb_domain":"exp81_hf_hgb_domain_oof.csv.gz","tm_mean":"exp81_hf_tm_mean_oof.csv.gz",
            "league_role":"exp81_hf_league_role_oof.csv.gz","cat_domain":"exp81_hf_cat_domain_oof.csv.gz",
            "cat_time":"exp81_hf_cat_time_oof.csv.gz"}.items(): V[k]=col(fn)
s=sum(flat.values()); hfc=sum((v/s)*V[k] for k,v in flat.items())
chain=0.85*(0.90*hfc+0.10*bigv)+0.15*col("exp60_recent2_oof.csv.gz")
C=chain.mean(); champ=np.clip(C+1.05*(chain-C),0,1)
def ev(p,yr):
    m=(season==yr)&rel
    q=p[m]+(y[m].mean()-p[m].mean()); return float(np.mean((q-y[m])**2))
b24,b22=ev(champ,2024),ev(champ,2022)
print(f"챔피언 2024 {b24:.8f}  2022 {b22:.8f}\n")
# 재료 수집 (팀원 zip 포함)
mats={}
for f in sorted(glob.glob(str(RES/"*oof*.csv.gz"))):
    nm=os.path.basename(f).replace("_oof.csv.gz","").replace(".csv.gz","")
    try:
        d=pd.read_csv(f)
        if "row_id" not in d.columns: continue
        pc=[c for c in d.columns if (c=="prediction" or c.startswith("p")) and pd.api.types.is_numeric_dtype(d[c])]
        if not pc: continue
        v=b.row_id.map(d.set_index("row_id")[pc].mean(axis=1)).to_numpy(np.float64)
        if np.isfinite(v[season==2024]).mean()<0.95: continue
        mats[nm]=v
    except Exception: pass
z=zipfile.ZipFile("/mnt/c/Users/nynu0/Desktop/LG해커톤/extratrees_mse_oof.csv.zip")
fn=[n for n in z.namelist() if n.endswith(".csv") and not n.endswith("/")][0]
with z.open(fn) as fh: dd=pd.read_csv(fh)
mats["TEAM_extratrees"]=b.row_id.map(dd.set_index("row_id")["prediction"]).to_numpy(np.float64)
print(f"재료 {len(mats)}개 스캔\n")
rows=[]
for nm,v in mats.items():
    msk=np.isfinite(v)
    p=champ.copy(); p[msk]=0.90*champ[msk]+0.10*v[msk]
    d24=(b24-ev(p,2024))*1e5
    d22=(b22-ev(p,2022))*1e5 if np.isfinite(v[season==2022]).mean()>0.95 else np.nan
    rows.append((nm,d24,d22,float(np.corrcoef(champ[msk],v[msk])[0,1])))
r=pd.DataFrame(rows,columns=["재료","d2024","d2022","상관"]).sort_values("d2024",ascending=False)
r.to_csv(RES/"exp107_marginal_scan.csv",index=False,encoding="utf-8-sig")
print("=== 10% 추가 시 2024 기여 상위 15 ===")
print(r.head(15).to_string(index=False))
print("\n=== 2024·2022 둘 다 양수인 재료 ===")
ok=r[(r.d2024>0)&(r.d2022>0)]
print(ok.to_string(index=False) if len(ok) else "  없음")

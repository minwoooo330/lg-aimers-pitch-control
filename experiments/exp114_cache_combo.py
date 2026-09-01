# -*- coding: utf-8 -*-
"""① 재료 행렬 캐시 생성(이후 스캔 25분->2분)  ② cat_sd + nn8비중 결합 탐색"""
import numpy as np, pandas as pd, sys, importlib.util, glob, os, zipfile, time
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"; B=HERE.parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
rel=(gt=="R")|(season>=2023)
t0=time.time()
CACHE=RES/"material_cache.npz"
def col(fn):
    d=pd.read_csv(RES/fn); return b.row_id.map(d.set_index("row_id")["prediction"]).to_numpy()
def seedavg(fn):
    d=pd.read_csv(RES/fn); sc=[c for c in d.columns if c.startswith("p") and c!="prediction"]
    sc=sc if sc else ["prediction"]
    return d.set_index("row_id")[sc].mean(axis=1)
if not CACHE.exists():
    mats={}
    for f in sorted(glob.glob(str(RES/"*oof*.csv.gz"))):
        nm=os.path.basename(f).replace("_oof.csv.gz","").replace(".csv.gz","")
        try:
            d=pd.read_csv(f)
            if "row_id" not in d.columns: continue
            pc=[c for c in d.columns if (c=="prediction" or c.startswith("p")) and pd.api.types.is_numeric_dtype(d[c])]
            if not pc: continue
            v=b.row_id.map(d.set_index("row_id")[pc].mean(axis=1)).to_numpy(np.float32)
            if np.isfinite(v[season==2024]).mean()<0.95: continue
            mats[nm]=v
        except Exception: pass
    mats["sean_v42"]=b.row_id.map(pd.read_csv(B/"sean_v42_oof.csv.gz").set_index("row_id")["prediction"]).to_numpy(np.float32)
    z=zipfile.ZipFile(B/"extratrees_mse_oof.csv.zip")
    fn=[n for n in z.namelist() if n.endswith(".csv") and not n.endswith("/")][0]
    with z.open(fn) as fh: ex=pd.read_csv(fh)
    mats["TEAM_extratrees"]=b.row_id.map(ex.set_index("row_id")["prediction"]).to_numpy(np.float32)
    np.savez_compressed(CACHE, row_id=b.row_id.to_numpy(), **mats)
    print(f"캐시 생성 {len(mats)}개 재료  ({time.time()-t0:.0f}s)",flush=True)
Z=np.load(CACHE,allow_pickle=True)
print(f"캐시 로드 {len(Z.files)-1}개  ({time.time()-t0:.0f}s)\n",flush=True)
def M(nm): return Z[nm].astype(np.float64)
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
C=chain.mean(); sharp=np.clip(C+1.05*(chain-C),0,1)
sdb=M("exp110_sd_pit_bat"); sda=M("exp110_sd_all")
d24=pd.read_csv(RES/"exp86_B_8ep_코사인_oof.csv.gz"); d22=pd.read_csv(RES/"exp109_B_8ep_코사인_2022_oof.csv.gz")
nn8=b.row_id.map(pd.concat([d24.set_index("row_id")["prediction"],d22.set_index("row_id")["prediction"]])).to_numpy(np.float64)
cs24=pd.read_csv(RES/"exp112_cat_sd_oof.csv.gz"); cs23=pd.read_csv(RES/"exp112b_cat_sd_2023_oof.csv.gz")
cats=b.row_id.map(pd.concat([cs24.set_index("row_id")["prediction"],cs23.set_index("row_id")["prediction"]])).to_numpy(np.float64)
print("cat_sd 커버리지:", {yr: round(float(np.isfinite(cats[season==yr]).mean()),3) for yr in [2022,2023,2024]})
def ev(q,yr):
    mm=(season==yr)&rel
    z2=q[mm]+(y[mm].mean()-q[mm].mean()); return float(np.mean((z2-y[mm])**2))
def build(wsd,wn,wa,wc):
    q=sharp.copy(); m1=np.isfinite(sdb); q[m1]=(1-wsd)*sharp[m1]+wsd*sdb[m1]
    m2=np.isfinite(nn8); q[m2]=(1-wn)*q[m2]+wn*nn8[m2]
    m3=np.isfinite(sda); q[m3]=(1-wa)*q[m3]+wa*sda[m3]
    if wc>0:
        m4=np.isfinite(cats); q[m4]=(1-wc)*q[m4]+wc*cats[m4]
    return q
champ=build(0.40,0.10,0.10,0.0)
b24,b22=ev(champ,2024),ev(champ,2022)
print(f"\ntri 챔피언 2024 {b24:.8f} 2022 {b22:.8f}\n")
print(f"{'구성':30s} {'d2024':>8s} {'d2022':>8s}  shift")
for wsd,wn,wa,wc in [(0.40,0.15,0.10,0.0),(0.40,0.10,0.10,0.10),(0.40,0.15,0.10,0.10),
                     (0.45,0.15,0.10,0.10),(0.40,0.15,0.10,0.15),(0.45,0.15,0.10,0.15)]:
    q=build(wsd,wn,wa,wc); sc_,sh=ec.score(q,y,season,rel)
    tag=f"sd{int(wsd*100)}+nn{int(wn*100)}+all{int(wa*100)}"+(f"+cat{int(wc*100)}" if wc else "")
    print(f"{tag:30s} {(b24-ev(q,2024))*1e5:+8.2f} {(b22-ev(q,2022))*1e5:+8.2f}  {sh:.12f}")

# -*- coding: utf-8 -*-
"""캐시 활용 재스캔: quad_v1을 가상 챔피언으로 놓고 아직 안 쓴 재료 전수 확인."""
import numpy as np, pandas as pd, sys, importlib.util, time
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
rel=(gt=="R")|(season>=2023)
t0=time.time()
Z=np.load(RES/"material_cache.npz",allow_pickle=True)
print(f"캐시 로드 {len(Z.files)-1}개 재료  ({time.time()-t0:.1f}s)",flush=True)
def M(nm): return Z[nm].astype(np.float64)
def col(fn):
    d=pd.read_csv(RES/fn); return b.row_id.map(d.set_index("row_id")["prediction"]).to_numpy()
def seedavg(fn):
    d=pd.read_csv(RES/fn); sc=[c for c in d.columns if c.startswith("p") and c!="prediction"]
    sc=sc if sc else ["prediction"]
    return d.set_index("row_id")[sc].mean(axis=1)
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
p=sharp.copy()
m1=np.isfinite(sdb); p[m1]=0.60*sharp[m1]+0.40*sdb[m1]
m2=np.isfinite(nn8); p[m2]=0.85*p[m2]+0.15*nn8[m2]
m3=np.isfinite(sda); p[m3]=0.90*p[m3]+0.10*sda[m3]
m4=np.isfinite(cats); p[m4]=0.90*p[m4]+0.10*cats[m4]
champ=p   # quad_v1 가상 챔피언
def ev(q,yr):
    mm=(season==yr)&rel
    z=q[mm]+(y[mm].mean()-q[mm].mean()); return float(np.mean((z-y[mm])**2))
b24,b22=ev(champ,2024),ev(champ,2022)
print(f"quad 가상챔피언  2024 {b24:.8f}  2022 {b22:.8f}\n")
used={"exp105_sd","exp110_sd_pit","exp110_sd_pit_bat","exp110_sd_all",
      "exp86_B_8ep_코사인","exp109_B_8ep_코사인_2022","exp112_cat_sd","exp112b_cat_sd_2023"}
rows=[]
for nm in Z.files:
    if nm=="row_id" or nm in used: continue
    v=M(nm); mm=np.isfinite(v)
    if mm.mean()<0.5: continue
    q=champ.copy(); q[mm]=0.90*champ[mm]+0.10*v[mm]
    d1=(b24-ev(q,2024))*1e5
    d2=(b22-ev(q,2022))*1e5 if np.isfinite(v[season==2022]).mean()>0.95 else np.nan
    rows.append((nm,d1,d2,float(np.corrcoef(champ[mm],v[mm])[0,1])))
r=pd.DataFrame(rows,columns=["재료","d2024","d2022","상관"]).sort_values("d2024",ascending=False)
r.to_csv(RES/"exp115_rescan_quad.csv",index=False,encoding="utf-8-sig")
print(f"스캔 시간 {time.time()-t0:.1f}s\n")
print("=== 둘 다 양수 (제출 가치 판단용) ===")
ok=r[(r.d2024>0)&(r.d2022>0)]
print(ok.head(15).to_string(index=False) if len(ok) else "  없음")
print(f"\n최대 2024 기여: {r.d2024.max():.2f}e-5 (이론 {r.d2024.max()*4:.1f}점)")

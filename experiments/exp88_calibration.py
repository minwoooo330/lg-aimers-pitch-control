# -*- coding: utf-8 -*-
"""실험 88: 보정 곡선 walk-forward 비교. 과거 fold(2022+2023)로만 fit → 2024로 정직 평가.
   후보: shift만 / 대칭 샤프닝 / 비대칭 샤프닝 / Isotonic / logit 구간선형."""
import numpy as np, pandas as pd, sys, importlib.util
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
rel=((gt=="R")|(season>=2023))
def seedavg(f):
    d=pd.read_csv(RES/f); sc=[c for c in d.columns if c.startswith("p") and c!="prediction"]
    sc=sc if sc else ["prediction"]
    return d.set_index("row_id")[sc].mean(axis=1)
def col(f):
    d=pd.read_csv(RES/f); return b.row_id.map(d.set_index("row_id")["prediction"]).to_numpy()
b["ms"]=b.row_id.map(seedavg("exp68_multisplit_oof.csv.gz")).to_numpy()
b["hc5"]=b.row_id.map(seedavg("exp65_handcnt_seeds5_oof.csv.gz")).to_numpy()
bigv=b.row_id.map(pd.concat([seedavg("exp84_hand_big_oof.csv.gz"),seedavg("exp84b_2023_oof.csv.gz")])).to_numpy()
W6=[0.06726666,0.11952455,0.15582993,0.17546554,0.23856908,0.24334425]
C6=["hgb_domain","cat_domain","tm_mean","league_role","cat_time","lgbm"]
flat={c:0.80*0.90*0.65*0.85*w for w,c in zip(W6,C6)}
flat.update({"nn3":0.80*0.90*0.65*0.15,"hand8":0.80*0.90*0.25,"pbhand3":0.80*0.90*0.10,
             "hc3":0.80*0.10,"ms":0.10,"hc5":0.10})
V={k:b[k].to_numpy() for k in flat}
for k,f in {"hgb_domain":"exp81_hf_hgb_domain_oof.csv.gz","tm_mean":"exp81_hf_tm_mean_oof.csv.gz",
            "league_role":"exp81_hf_league_role_oof.csv.gz","cat_domain":"exp81_hf_cat_domain_oof.csv.gz",
            "cat_time":"exp81_hf_cat_time_oof.csv.gz"}.items(): V[k]=col(f)
s=sum(flat.values()); hfc=sum((v/s)*V[k] for k,v in flat.items())
chain=0.85*(0.90*hfc+0.10*bigv)+0.15*col("exp60_recent2_oof.csv.gz")
fit=(np.isin(season,[2022,2023]))&rel     # 과거 fold만으로 학습
ev =(season==2024)&rel                     # 2024로 평가
pf,yf=chain[fit],y[fit]; pe,ye=chain[ev],y[ev]
def score(p):  # 평균정렬 후 Brier (shift는 별도 축이므로 제거)
    q=p+(ye.mean()-p.mean()); return float(np.mean((q-ye)**2))
base=score(pe); print(f"기준(shift만) 2024 Brier {base:.8f}\n")
Cf=pf.mean()
print("--- 대칭 샤프닝 (k는 과거 fold에서 최적화, 2024로 평가) ---")
ks=np.arange(0.95,1.31,0.01)
bf=[np.mean(((Cf+k*(pf-Cf))+(yf.mean()-(Cf+k*(pf-Cf)).mean())-yf)**2) for k in ks]
kbest=ks[int(np.argmin(bf))]
for k in [1.00,1.05,1.08,1.10,kbest]:
    p=Cf+k*(pe-Cf); print(f"  k={k:.2f}{'  <-과거fold 최적' if k==kbest else '':16s} 2024 {(base-score(p))*1e5:+7.3f}e-5")
print("\n--- 비대칭 샤프닝 (중앙 위/아래 다른 k, 과거 fold 최적화) ---")
best=None
for kl in np.arange(0.95,1.26,0.05):
    for kh in np.arange(0.95,1.36,0.05):
        q=np.where(pf>=Cf, Cf+kh*(pf-Cf), Cf+kl*(pf-Cf))
        v=np.mean((q+(yf.mean()-q.mean())-yf)**2)
        if best is None or v<best[0]: best=(v,kl,kh)
_,kl,kh=best
q=np.where(pe>=Cf, Cf+kh*(pe-Cf), Cf+kl*(pe-Cf))
print(f"  과거fold 최적 k_low={kl:.2f} k_high={kh:.2f}  → 2024 {(base-score(q))*1e5:+7.3f}e-5")
print("\n--- Isotonic (과거 fold fit) ---")
iso=IsotonicRegression(out_of_bounds="clip").fit(pf,yf)
print(f"  2024 {(base-score(iso.predict(pe)))*1e5:+7.3f}e-5")
print("\n--- logit 구간선형 (과거 fold fit, 10분위 노드) ---")
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
edges=np.quantile(pf,np.linspace(0,1,11))
xf,xe=lg(pf),lg(pe); nodes=lg(np.clip(edges,1e-6,1-1e-6))
tgt=[yf[(pf>=edges[i])&(pf<=edges[i+1])].mean() for i in range(10)]
mid=[(nodes[i]+nodes[i+1])/2 for i in range(10)]
pl=np.interp(xe,mid,lg(np.array(tgt)))
print(f"  2024 {(base-score(1/(1+np.exp(-pl))))*1e5:+7.3f}e-5")

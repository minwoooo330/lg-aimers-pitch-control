# -*- coding: utf-8 -*-
"""실험 104: 앙상블 조합 전면 재탐색.
   지금까지 체인은 '좋아 보이는 걸 순서대로 얹은' 탐욕적 누적이었고 조합 자체를 탐색한 적이 없다.
   기각된 모델도 다른 조합에서는 살아날 수 있으므로, 보유한 모든 OOF를 재료로 놓고
   정직한 교차 검증(2022로 조합 학습 -> 2024 평가, 그 반대)으로 현 체인과 비교한다."""
import numpy as np, pandas as pd, sys, importlib.util, glob, os
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
rel=(gt=="R")|(season>=2023)
rid=b.row_id.to_numpy()
idx=pd.Series(np.arange(len(b)),index=rid)

# ---------- 재료 수집 ----------
lib={}
for f in sorted(glob.glob(str(RES/"*oof*.csv.gz"))):
    nm=os.path.basename(f).replace("_oof.csv.gz","").replace(".csv.gz","")
    try:
        d=pd.read_csv(f)
        if "row_id" not in d.columns: continue
        pcols=[c for c in d.columns if c=="prediction" or (c.startswith("p") and c not in ("prediction",) and d[c].dtype!=bool)]
        pcols=[c for c in pcols if pd.api.types.is_numeric_dtype(d[c])]
        if not pcols: continue
        s=d.set_index("row_id")[pcols].mean(axis=1)
        v=b.row_id.map(s).to_numpy(np.float64)
        cov24=np.isfinite(v[season==2024]).mean(); cov22=np.isfinite(v[season==2022]).mean()
        if cov24<0.95: continue
        lib[nm]=(v,cov22)
    except Exception: pass
both=[k for k,(v,c22) in lib.items() if c22>0.95]
only24=[k for k,(v,c22) in lib.items() if c22<=0.95]
print(f"재료: 2024 커버 {len(lib)}개 / 그중 2022도 커버 {len(both)}개")

def score(p,yr):
    m=(season==yr)&rel
    q=p[m]+(y[m].mean()-p[m].mean()); return float(np.mean((q-y[m])**2))

# ---------- 현 체인 (기준) ----------
def seedavg(fn):
    d=pd.read_csv(RES/fn); sc=[c for c in d.columns if c.startswith("p") and c!="prediction"]
    sc=sc if sc else ["prediction"]
    return d.set_index("row_id")[sc].mean(axis=1)
def col(fn):
    d=pd.read_csv(RES/fn); return b.row_id.map(d.set_index("row_id")["prediction"]).to_numpy()
bb={}
bb["ms"]=b.row_id.map(seedavg("exp68_multisplit_oof.csv.gz")).to_numpy()
bb["hc5"]=b.row_id.map(seedavg("exp65_handcnt_seeds5_oof.csv.gz")).to_numpy()
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
print(f"\n현 챔피언  2022 {score(champ,2022):.8f}  2024 {score(champ,2024):.8f}")

# ---------- Caruana 탐욕 선택 (복원 추출) ----------
def caruana(names, fit_yr, n_iter=60):
    m=(season==fit_yr)&rel; yy=y[m]
    P={k:lib[k][0][m] for k in names}
    P={k:v for k,v in P.items() if np.isfinite(v).all()}
    cur=np.zeros(m.sum()); picked=[]
    best_hist=[]
    for it in range(n_iter):
        bestk=None; bestv=None
        for k,v in P.items():
            cand=(cur*len(picked)+v)/(len(picked)+1)
            q=cand+(yy.mean()-cand.mean())
            sc_=np.mean((q-yy)**2)
            if bestv is None or sc_<bestv: bestv, bestk = sc_, k
        picked.append(bestk); cur=(cur*(len(picked)-1)+P[bestk])/len(picked)
        best_hist.append(bestv)
    return picked, best_hist

for fit_yr, ev_yr in [(2022,2024),(2024,2022)]:
    print(f"\n{'='*70}\n조합 학습={fit_yr} → 평가={ev_yr}")
    picked,hist=caruana(both, fit_yr, 60)
    from collections import Counter
    cnt=Counter(picked)
    print(f"  선택된 멤버 {len(cnt)}종:", ", ".join(f"{k}×{v}" for k,v in cnt.most_common(10)))
    w={k:v/len(picked) for k,v in cnt.items()}
    ens=sum(wt*lib[k][0] for k,wt in w.items())
    print(f"  학습 fold({fit_yr}) Brier {hist[-1]:.8f}")
    print(f"  평가 fold({ev_yr}) Brier {score(ens,ev_yr):.8f}   챔피언 {score(champ,ev_yr):.8f}"
          f"   차이 {(score(champ,ev_yr)-score(ens,ev_yr))*1e5:+.3f}e-5")
    eq=np.mean([lib[k][0] for k in both],axis=0)
    print(f"  [참고] 전체 균등평균({len(both)}개) 평가 Brier {score(eq,ev_yr):.8f}"
          f"   차이 {(score(champ,ev_yr)-score(eq,ev_yr))*1e5:+.3f}e-5")

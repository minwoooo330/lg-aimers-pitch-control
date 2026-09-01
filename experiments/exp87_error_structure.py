# -*- coding: utf-8 -*-
"""실험 87: 오차 구조 분석 — "무엇을 추가할까"가 아니라 "어디서 지고 있나".
   2024 fold에서 챔피언 체인의 손실을 구간별로 분해하고, 상수예측 대비 어디서 이득/손실인지 본다."""
import numpy as np, pandas as pd, sys, importlib.util
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
rel=(gt=="R")|(season>=2023)
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
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
tr=tr[tr.row_id.isin(b.row_id)].set_index("row_id").loc[b.row_id].reset_index()
m=(season==2024)&rel
p=chain[m]; yy=y[m]; p=p+(yy.mean()-p.mean())   # 평균정렬
const=yy.mean()
sub=tr[m].reset_index(drop=True)
print(f"2024 채점대상 {m.sum():,}행 | 모델 Brier {np.mean((p-yy)**2):.8f} | 상수 {np.mean((const-yy)**2):.8f}")
print(f"전체 이득 {(np.mean((const-yy)**2)-np.mean((p-yy)**2))*1e5:+.2f}e-5\n")
def seg(name, keys):
    rows=[]
    for k,mask in keys:
        if mask.sum()<2000: continue
        gain=(np.mean((const-yy[mask])**2)-np.mean((p[mask]-yy[mask])**2))*1e5
        rows.append((k,int(mask.sum()),round(float(yy[mask].mean()),4),round(float(p[mask].mean()),4),round(float(gain),2)))
    d=pd.DataFrame(rows,columns=["구간","n","실제","예측","이득e-5"]).sort_values("이득e-5")
    print(f"--- {name} (이득 낮은 순) ---"); print(d.to_string(index=False)); print()
cnt=sub.balls_before.astype(str)+"-"+sub.strikes_before.astype(str)
seg("카운트",[(c,(cnt==c).to_numpy()) for c in sorted(cnt.unique())])
n=sub.asof_pitcher_n.fillna(0)
seg("투수 표본수",[("<=200",(n<=200).to_numpy()),("200-1k",((n>200)&(n<=1000)).to_numpy()),
                 ("1k-5k",((n>1000)&(n<=5000)).to_numpy()),(">5k",(n>5000).to_numpy())])
seg("리그",[("R",(sub.game_type=="R").to_numpy()),("F",(sub.game_type=="F").to_numpy())])
seg("이닝",[(f"{i}회",(sub.inning==i).to_numpy()) for i in range(1,10)])
seg("월",[(f"{i}월",(sub.game_month==i).to_numpy()) for i in sorted(sub.game_month.unique())] if "game_month" in sub.columns else [])
q=pd.qcut(p,10,labels=False,duplicates="drop")
seg("예측 십분위",[(f"D{i}",(q==i)) for i in range(10)])

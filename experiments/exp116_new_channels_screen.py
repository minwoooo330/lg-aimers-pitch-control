# -*- coding: utf-8 -*-
"""새 채널 후보 저비용 스크리닝: 월별 당해 성적 / 팀 당해 성적.
   챔피언(quad 가상) 잔차와의 상관만 본다 — 문턱 통과분만 모델 투입."""
import numpy as np, pandas as pd, sys, importlib.util
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
Z=np.load(RES/"material_cache.npz",allow_pickle=True)
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
sdb=M("exp110_sd_pit_bat")
p=sharp.copy(); m1=np.isfinite(sdb); p[m1]=0.60*sharp[m1]+0.40*sdb[m1]
champ=p
m24=(season==2024)
resid=(y-champ)[m24]
df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
d24=df[df.season==2024].reset_index(drop=True)
assert len(d24)==int(m24.sum())

print("=== 후보 ① 팀(투수팀) 당해 성적 — 팀 단위 조회표 ===")
idx=df.groupby("pitcher_team_id")["asof_pitcher_n"].idxmax()   # 근사: 팀 전체 n 대신 대표 투수
# 팀 단위는 asof_pitcher_n이 개인 값이라 직접 합산 불가 -> 팀×시즌 평균 성공률로 근사
prev_team=df[df.season==2023].groupby("pitcher_team_id")["asof_pitcher_success_rate"].mean()
cur_team=d24.groupby("pitcher_team_id")["asof_pitcher_success_rate"].transform("mean")
team_prev=d24.pitcher_team_id.map(prev_team)
team_sd=(cur_team.to_numpy()-team_prev.to_numpy())  # 팀 평균 통산 성공률의 연도간 변화(근사)
ok=np.isfinite(team_sd)
print(f"  유효 {ok.sum():,} / {len(d24):,}  상관 {np.corrcoef(team_sd[ok],resid[ok])[0,1]:+.5f}")

print("\n=== 후보 ② 월별 당해 성적 (이번 달만) ===")
prev_month=df[(df.season==2024)].copy()
prev_month["cumsucc"]=prev_month.asof_pitcher_success_rate*prev_month.asof_pitcher_n
tab={}
for mo in sorted(d24.game_month.unique()):
    prevm=prev_month[prev_month.game_month<mo]
    if len(prevm)==0:
        tab[mo]=None; continue
    idxm=prevm.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    lastm=prevm.loc[idxm]
    tab[mo]={"n":pd.Series(lastm.asof_pitcher_n.to_numpy(),index=lastm.pitcher_id.to_numpy()),
             "cs":pd.Series(lastm.cumsucc.to_numpy(),index=lastm.pitcher_id.to_numpy())}
n=d24.asof_pitcher_n.to_numpy(np.float64); pid=d24.pitcher_id.to_numpy()
cur_cs=d24.asof_pitcher_success_rate.to_numpy()*n
n0=np.full(len(d24),np.nan); cs0=np.full(len(d24),np.nan)
for mo,tb in tab.items():
    if tb is None: continue
    mm=(d24.game_month==mo).to_numpy()
    sub=pd.Series(pid[mm])
    n0[mm]=sub.map(tb["n"]).to_numpy(np.float64)
    cs0[mm]=sub.map(tb["cs"]).to_numpy(np.float64)
dn=n-n0; valid=np.isfinite(dn)&(dn>=10)
month_rate=np.where(valid,(cur_cs-cs0)/np.maximum(dn,1),np.nan)
ok2=np.isfinite(month_rate)
print(f"  유효 {ok2.sum():,} / {len(d24):,} ({ok2.mean()*100:.1f}%)  상관 {np.corrcoef(month_rate[ok2],resid[ok2])[0,1]:+.5f}")
print(f"  (문턱 참고: 채택 0.0105 이상, 기각 0.0073 이하)")

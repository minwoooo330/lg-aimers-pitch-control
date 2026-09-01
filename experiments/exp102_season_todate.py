# -*- coding: utf-8 -*-
"""실험 102: '시즌 당해 성적' 복원 피처.
   test 한 행의 (asof_n, asof_rate)와 학습데이터로 만든 '직전 시즌 말 상태' 조회표를 차분하면
   그 투수의 당해 시즌 성적이 나온다. 다른 test 행을 쓰지 않으므로 규정 허용.
   우리 모델은 지금까지 통산 누적만 봐서 '올해 폼'을 구분하지 못했다."""
import numpy as np, pandas as pd, sys, importlib.util
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
sys.path.insert(0,str(HERE))
df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
RATES=["success_rate","middle_rate","reverse_rate","ball_rate","strike_rate"]
def season_end_state(d, upto):
    """upto 시즌 말 각 투수의 (n, 각 rate) — 학습 데이터로만 만드는 조회표"""
    s=d[d.season<=upto]
    idx=s.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    last=s.loc[idx]
    out=last[["pitcher_id","asof_pitcher_n"]].copy()
    for r in RATES: out["r_"+r]=last["asof_pitcher_"+r].to_numpy()
    return out.set_index("pitcher_id")
def add_season_todate(d, tbl):
    """현재 행 + 조회표 → 당해 시즌 성적 (행 독립)"""
    n=d.asof_pitcher_n.to_numpy(np.float64)
    n0=d.pitcher_id.map(tbl.asof_pitcher_n).to_numpy(np.float64)
    f=pd.DataFrame(index=d.index)
    dn=n-n0
    f["sd_n"]=dn
    f["sd_isnew"]=(~np.isfinite(n0)).astype(np.int8)
    for r in RATES:
        cur=d["asof_pitcher_"+r].to_numpy(np.float64)
        prev=d.pitcher_id.map(tbl["r_"+r]).to_numpy(np.float64)
        succ=cur*n - prev*n0
        rate=np.where(dn>0, succ/np.maximum(dn,1), np.nan)
        f["sd_"+r]=rate
        f["sd_delta_"+r]=rate-cur          # 당해 성적이 통산 대비 얼마나 다른가
    # 신뢰도 가중 (표본 적으면 통산으로 수축)
    K=100.0
    w=dn/(dn+K)
    f["sd_w"]=w
    f["sd_shrunk_success"]=np.where(np.isfinite(f.sd_success_rate),
        w*f.sd_success_rate+(1-w)*d.asof_pitcher_success_rate.to_numpy(), d.asof_pitcher_success_rate.to_numpy())
    return f
print("=== 2024 fold에서 '당해 시즌 성적' 복원 검증 ===")
tbl=season_end_state(df,2023)
va=df[df.season==2024].reset_index(drop=True)
f=add_season_todate(va,tbl)
print(f"2024 행 {len(va):,} | 신규 투수(조회표 없음) {int(f.sd_isnew.sum()):,} ({f.sd_isnew.mean()*100:.1f}%)")
ok=np.isfinite(f.sd_success_rate.to_numpy())
print(f"당해 성적 복원 가능 {ok.sum():,} ({ok.mean()*100:.1f}%)")
print(f"  당해 시즌 투구수 중앙값 {np.nanmedian(f.sd_n[ok]):.0f}")
print(f"  당해 성공률 평균 {np.nanmean(f.sd_success_rate[ok]):.4f} (실제 2024 {va.control_success.mean():.4f})")
print(f"  통산 성공률 평균 {va.asof_pitcher_success_rate.mean():.4f}  ← 모델이 지금 보는 값")
print(f"  당해-통산 차이 표준편차 {np.nanstd(f.sd_delta_success_rate[ok]):.4f}")
# 잔차 상관
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
def seedavg(fn):
    d2=pd.read_csv(RES/fn); sc=[c for c in d2.columns if c.startswith("p") and c!="prediction"]
    sc=sc if sc else ["prediction"]
    return d2.set_index("row_id")[sc].mean(axis=1)
def col(fn):
    d2=pd.read_csv(RES/fn); return b.row_id.map(d2.set_index("row_id")["prediction"]).to_numpy()
b["ms"]=b.row_id.map(seedavg("exp68_multisplit_oof.csv.gz")).to_numpy()
b["hc5"]=b.row_id.map(seedavg("exp65_handcnt_seeds5_oof.csv.gz")).to_numpy()
bigv=b.row_id.map(pd.concat([seedavg("exp84_hand_big_oof.csv.gz"),seedavg("exp84b_2023_oof.csv.gz")])).to_numpy()
W6=[0.06726666,0.11952455,0.15582993,0.17546554,0.23856908,0.24334425]
C6=["hgb_domain","cat_domain","tm_mean","league_role","cat_time","lgbm"]
flat={c:0.80*0.90*0.65*0.85*w for w,c in zip(W6,C6)}
flat.update({"nn3":0.80*0.90*0.65*0.15,"hand8":0.80*0.90*0.25,"pbhand3":0.80*0.90*0.10,
             "hc3":0.80*0.10,"ms":0.10,"hc5":0.10})
V={k:b[k].to_numpy() for k in flat}
for k,fn in {"hgb_domain":"exp81_hf_hgb_domain_oof.csv.gz","tm_mean":"exp81_hf_tm_mean_oof.csv.gz",
            "league_role":"exp81_hf_league_role_oof.csv.gz","cat_domain":"exp81_hf_cat_domain_oof.csv.gz",
            "cat_time":"exp81_hf_cat_time_oof.csv.gz"}.items(): V[k]=col(fn)
s=sum(flat.values()); hfc=sum((v/s)*V[k] for k,v in flat.items())
chain=0.85*(0.90*hfc+0.10*bigv)+0.15*col("exp60_recent2_oof.csv.gz")
m24=(season==2024)
resid=(y-chain)[m24]
ids24=b.row_id[m24].to_numpy()
fv=f.set_index(va.row_id.to_numpy()).reindex(ids24)
print("\n=== 잔차 상관 (문턱: 채택된 손잡이축 0.0105 / 기각된 카운트축 0.0073) ===")
for c in ["sd_success_rate","sd_delta_success_rate","sd_shrunk_success","sd_middle_rate",
          "sd_reverse_rate","sd_delta_middle_rate","sd_n","sd_w"]:
    v=fv[c].to_numpy(np.float64); msk=np.isfinite(v)&np.isfinite(resid)
    if msk.sum()<5000: continue
    print(f"  {c:26s} n={int(msk.sum()):>7,}  상관 {np.corrcoef(v[msk],resid[msk])[0,1]:+.5f}")

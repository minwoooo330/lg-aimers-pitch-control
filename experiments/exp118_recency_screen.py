# -*- coding: utf-8 -*-
"""새 방향 스크리닝(합법):
   ① '최근경기(공식 prev1) vs 시즌당해(sd, 합법)' 괴리 — '지금 뜨거운가'를 안전하게 근사
   ② 최근경기 자체의 파생(prev1-prev5 추세, 최근 변동성)
   전부 현재 행 자신의 공식 컬럼 + 학습데이터 조회표만 사용. 다른 test 행 미참조."""
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
sd24=b.row_id.map(pd.DataFrame({"row_id":b.row_id[m24]}).assign(v=0)["v"])  # placeholder
sd_oof=pd.read_csv(RES/"exp105_sd_oof.csv.gz")
sd_map=sd_oof.set_index("row_id")["prediction"]
sd_val=d24.row_id.map(sd_map).to_numpy()  # sd 모델 예측(참고용, 여기선 raw 피처가 필요)

# raw 피처로 직접 계산 (모델 예측이 아니라 원본 asof 컬럼 차분)
n=d24.asof_pitcher_n.to_numpy(np.float64)
sr=d24.asof_pitcher_success_rate.to_numpy(np.float64)
p1=d24.asof_pitcher_prev1_game_success_rate.to_numpy(np.float64)
p3=d24.asof_pitcher_prev3_game_success_rate.to_numpy(np.float64)
p5=d24.asof_pitcher_prev5_game_success_rate.to_numpy(np.float64)

print("=== 후보 A: 최근1경기 vs 통산 괴리 (이미 raw로 존재하나 재확인) ===")
g=p1-sr; ok=np.isfinite(g)
print(f"  상관 {np.corrcoef(g[ok],resid[ok])[0,1]:+.5f}  유효 {ok.mean()*100:.0f}%")

print("\n=== 후보 B: 최근1 vs 최근5 추세(모멘텀) ===")
g=p1-p5; ok=np.isfinite(g)
print(f"  상관 {np.corrcoef(g[ok],resid[ok])[0,1]:+.5f}  유효 {ok.mean()*100:.0f}%")

print("\n=== 후보 C: 최근1,3,5의 변동성(표준편차) ===")
stack=np.vstack([p1,p3,p5])
volat=np.nanstd(stack,axis=0); ok=np.isfinite(volat)
print(f"  상관 {np.corrcoef(volat[ok],resid[ok])[0,1]:+.5f}  유효 {ok.mean()*100:.0f}%")

print("\n=== 후보 D: (최근1 - 통산) x (표본충분여부) — 저표본 노이즈 억제 ===")
w=np.clip(n/(n+200),0,1)
g=(p1-sr)*w; ok=np.isfinite(g)
print(f"  상관 {np.corrcoef(g[ok],resid[ok])[0,1]:+.5f}")

print("\n=== 후보 E: 최근1 vs 통산, 최근5 vs 통산 둘 다 (2차원 조합 근사: 곱) ===")
g=(p1-sr)*(p5-sr); ok=np.isfinite(g)
print(f"  상관 {np.corrcoef(g[ok],resid[ok])[0,1]:+.5f}")
print("\n(문턱 참고: 채택 0.0105 / 기각 0.0073, 단 오늘 교훈상 이건 순서 정하기용일 뿐)")


import numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"; sys.path.insert(0,str(HERE))
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
b["nn"]=0.5*((3*a3+5*a5)/8)+0.5*b8
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
d=b.merge(tr,on=ID,how="left",suffixes=("","_x"))
d=d[(d.game_type=="R")|(d.season>=2023)]
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
CTX=["balls_before","strikes_before","outs_before","inning","li","num_runners_on",
     "score_diff_pitcher_team","game_month","asof_pitcher_n","asof_pitcher_success_rate",
     "asof_batter_n","asof_batter_success_rate"]
CTX=[c for c in CTX if c in d.columns]
def prep(s):
    z=pd.DataFrame({"lv":lg(s.v7.to_numpy()),"ln":lg(s.nn.to_numpy())})
    z["same"]=(s.pitcher_hand.to_numpy()==s.batter_hand.to_numpy()).astype(float)
    for c in CTX: z[c]=s[c].to_numpy(dtype=np.float64)
    return z
s22=d[d.season==2022].reset_index(drop=True); s24=d[d.season==2024].reset_index(drop=True)
y22=s22[TARGET].to_numpy(); y24=s24[TARGET].to_numpy()
X22,X24=prep(s22),prep(s24)
med=X22.median()                      # 결측 대치는 학습 fold(2022)에서만 산출
X22=X22.fillna(med); X24=X24.fillna(med)
def sc(p,y):
    p=p+(y.mean()-p.mean()); return float(np.mean((y-p)**2))
# 현행 베이스 (고정 0.55/0.45 + scale 1.08)
def cur(s):
    p=0.55*s.v7.to_numpy()+0.45*s.nn.to_numpy()
    c=-0.027330; z=c+1.08*(lg(p)-c); return 1/(1+np.exp(-z))
base24=sc(cur(s24),y24); base22=sc(cur(s22),y22)
print("=== 스태킹 재개장 (v7 병합 후) — 시간안전: 2022로 메타학습 → 2024 평가 ===")
print(f"  현행 베이스   2024 {base24:.8f}   2022 {base22:.8f}\n")
res={}
# (1) Ridge on logits only
for tag,cols in [("Ridge 로짓2개",["lv","ln"]),("Ridge 로짓+문맥",list(X22.columns))]:
    m=Ridge(alpha=10.0).fit(X22[cols],lg(y22*0.998+0.001))
    q=1/(1+np.exp(-m.predict(X24[cols]))); res[tag]=sc(q,y24)
# (2) Logistic on logits
for tag,cols in [("Logistic 로짓2개",["lv","ln"]),("Logistic 로짓+문맥",list(X22.columns))]:
    m=LogisticRegression(C=1.0,max_iter=300).fit(X22[cols],y22)
    q=m.predict_proba(X24[cols])[:,1]; res[tag]=sc(q,y24)
# (3) HGB meta (gating)
m=HistGradientBoostingRegressor(max_iter=100,learning_rate=0.05,max_leaf_nodes=15,
    min_samples_leaf=2000,l2_regularization=10.0,random_state=42).fit(X22,y22)
res["HGB 메타(게이팅)"]=sc(np.clip(m.predict(X24),0,1),y24)
# (4) 문맥의존 가중치: w(x) 학습
m=HistGradientBoostingRegressor(max_iter=100,learning_rate=0.05,max_leaf_nodes=15,
    min_samples_leaf=2000,l2_regularization=10.0,random_state=42)
resid=(y22-s22.nn.to_numpy()); denom=(s22.v7.to_numpy()-s22.nn.to_numpy())
ok=np.abs(denom)>1e-4
m.fit(X22[ok],np.clip(resid[ok]/denom[ok],-1,2))
w=np.clip(m.predict(X24),0,1)
q=w*s24.v7.to_numpy()+(1-w)*s24.nn.to_numpy()
res["문맥의존 가중치 w(x)"]=sc(q,y24)
print(f"  {'방법':26s} {'2024 Brier':>13s}  {'현행대비':>10s}")
for k,v in res.items():
    print(f"  {k:26s} {v:.8f}  {(base24-v)*1e5:+8.2f}e-5")
print("\n  (+면 개선, -면 악화)")

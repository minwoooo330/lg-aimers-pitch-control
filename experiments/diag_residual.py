
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
import sys
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"; sys.path.insert(0,str(HERE))
from features import add_features
from hfeatures import add_hfeatures
ID,TARGET="row_id","control_success"
b=pd.read_csv(RES/"exp148_v7_oof.csv.gz").rename(columns={"prediction":"v7"})
def g(fn):
    d=pd.read_csv(RES/fn).set_index(ID)["prediction"]; return b[ID].map(d).to_numpy(np.float64)
a3=g("exp120_nn_sd_oof.csv.gz"); a5=g("exp133_nnsd_seeds5_oof.csv.gz"); b8=g("exp144_nnsd_seeds8b_oof.csv.gz")
b["nn"]=0.5*((3*a3+5*a5)/8)+0.5*b8
b["base"]=0.55*b["v7"]+0.45*b["nn"]
tr=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
d=b.merge(tr,on=ID,how="left",suffixes=("","_x"))
d=d[(d.game_type=="R")|(d.season>=2023)].copy()
d24=d[d.season==2024].copy(); d22=d[d.season==2022].copy()
for s in (d24,d22):
    p=s.base.to_numpy(); yy=s[TARGET].to_numpy()
    s["p"]=p+(yy.mean()-p.mean()); s["res"]=yy-s["p"]; s["se"]=s["res"]**2

print("="*74); print("[A] 요청하신 분석: 두 모델이 공통으로 가장 크게 틀린 상위 1%"); print("="*74)
top=d24.nlargest(int(len(d24)*0.01),"se")
rest=d24.drop(top.index)
print(f"  상위1% {len(top):,}행   평균 se {top.se.mean():.4f}   나머지 {rest.se.mean():.4f}")
print(f"  상위1%의 실제 성공률 {top[TARGET].mean():.4f}   예측평균 {top.p.mean():.4f}")
print(f"  나머지  실제 {rest[TARGET].mean():.4f}   예측 {rest.p.mean():.4f}")
print(f"\n  예측 극단성: 상위1% |p-0.5| 평균 {np.abs(top.p-0.5).mean():.4f} / 나머지 {np.abs(rest.p-0.5).mean():.4f}")
print(f"  상위1% 중 (예측낮은데 성공) 비율 {(top[TARGET]==1).mean():.3f}")
print("\n  [해석] 상위1%는 '예측이 극단적이었는데 라벨이 반대로 나온 행'이다.")
print("  이진 라벨에서는 se=(p-y)^2이라 p가 0.5에서 멀수록 자동으로 커진다.")
print("  즉 이 집합의 특징은 '모델이 놓친 패턴'이 아니라 '확신했다가 진 행'이다.")

print("\n"+"="*74); print("[B] 제대로 된 검사: 잔차에 학습 가능한 구조가 남아 있는가"); print("="*74)
print("  2022 fold 잔차로 모델을 학습해 2024 잔차를 예측한다.")
print("  구조가 있다면 2024에서 잔차가 줄고 Brier가 개선된다(out-of-time 검증).")
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
cols=[c for c in tr.columns if c not in (ID,TARGET)]
maps={c:{v:i for i,v in enumerate(sorted(d22[c].dropna().astype(str).unique()))} for c in CAT}
def enc(s):
    x=s[cols].copy()
    for c in CAT: x[c]=s[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
    return pd.concat([x.reset_index(drop=True),add_features(s).reset_index(drop=True),
                      add_hfeatures(s).reset_index(drop=True)],axis=1)
X22=enc(d22); X24=enc(d24)
m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_leaf_nodes=31,
                                min_samples_leaf=500,l2_regularization=1.0,random_state=42)
m.fit(X22, d22.res.to_numpy())
pred_res=m.predict(X24)
print(f"\n  예측된 잔차의 SD {pred_res.std():.6f}   실제 잔차 SD {d24.res.std():.6f}")
print(f"  상관(예측잔차, 실제잔차) {np.corrcoef(pred_res,d24.res)[0,1]:+.5f}")
base_b=float(np.mean(d24.res**2))
for a in (0.25,0.5,1.0):
    q=np.clip(d24.p.to_numpy()+a*pred_res,0,1)
    nb=float(np.mean((d24[TARGET].to_numpy()-q)**2))
    print(f"  보정계수 {a:.2f}: Brier {base_b:.8f} -> {nb:.8f}   {(base_b-nb)*1e5:+.2f}e-5")

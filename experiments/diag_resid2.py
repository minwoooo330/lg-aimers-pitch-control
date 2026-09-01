
import numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
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
d=d[(d.game_type=="R")|(d.season>=2023)]
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
cols=[c for c in tr.columns if c not in (ID,TARGET)]
print("=== 2024 내부 5-fold 교차적합: 잔차에 구조가 있는가 ===")
print("  (같은 시즌 안에서 학습/예측하므로 드리프트 없음 = 검정력 최대)")
print("  주의: 이건 '상한 탐지'용이지 배포 가능한 방법이 아니다.\n")
for yr in (2024,2022):
    s=d[d.season==yr].reset_index(drop=True)
    p=s.base.to_numpy(); yy=s[TARGET].to_numpy()
    p=p+(yy.mean()-p.mean()); res=yy-p
    maps={c:{v:i for i,v in enumerate(sorted(s[c].dropna().astype(str).unique()))} for c in CAT}
    x=s[cols].copy()
    for c in CAT: x[c]=s[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
    X=pd.concat([x,add_features(s),add_hfeatures(s)],axis=1)
    oof=np.zeros(len(s))
    for trn,val in KFold(5,shuffle=True,random_state=0).split(X):
        m=HistGradientBoostingRegressor(max_iter=150,learning_rate=0.05,max_leaf_nodes=31,
            min_samples_leaf=1000,l2_regularization=10.0,random_state=42)
        m.fit(X.iloc[trn],res[trn]); oof[val]=m.predict(X.iloc[val])
    base_b=float(np.mean(res**2))
    print(f"[{yr}] 예측잔차 SD {oof.std():.6f}   상관 {np.corrcoef(oof,res)[0,1]:+.5f}")
    best=None
    for a in (0.1,0.25,0.5,0.75,1.0):
        q=np.clip(p+a*oof,0,1); nb=float(np.mean((yy-q)**2))
        gain=(base_b-nb)*1e5
        if best is None or gain>best[1]: best=(a,gain)
        print(f"     계수 {a:.2f}: {gain:+8.2f}e-5")
    print(f"  → 최대 {best[1]:+.2f}e-5 (계수 {best[0]})  = 이 피처들로 도달 가능한 상한\n")

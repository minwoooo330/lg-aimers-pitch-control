
import numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
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
s=d[d.season==2024].reset_index(drop=True)
p=s.base.to_numpy(); yy=s[TARGET].to_numpy(); p=p+(yy.mean()-p.mean()); res=yy-p
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
cols=[c for c in tr.columns if c not in (ID,TARGET)]
maps={c:{v:i for i,v in enumerate(sorted(s[c].dropna().astype(str).unique()))} for c in CAT}
x=s[cols].copy()
for c in CAT: x[c]=s[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
X=pd.concat([x,add_features(s),add_hfeatures(s)],axis=1)

print("=== 잔차 구조의 정체: 피처군을 하나씩 빼면서 상한이 어떻게 변하는가 ===")
def run(Xs,tag):
    oof=np.zeros(len(s))
    for trn,val in KFold(3,shuffle=True,random_state=0).split(Xs):
        m=HistGradientBoostingRegressor(max_iter=100,learning_rate=0.05,max_leaf_nodes=31,
            min_samples_leaf=1000,l2_regularization=10.0,random_state=42)
        m.fit(Xs.iloc[trn],res[trn]); oof[val]=m.predict(Xs.iloc[val])
    q=np.clip(p+oof,0,1)
    gain=(np.mean(res**2)-np.mean((yy-q)**2))*1e5
    print(f"  {tag:34s} 상관 {np.corrcoef(oof,res)[0,1]:+.4f}   상한 {gain:+7.2f}e-5")
    return gain
full=run(X,"전체 피처")
run(X.drop(columns=["pitcher_id","batter_id"]),"− 선수ID 2개")
run(X[["pitcher_id","batter_id"]],"선수ID 2개만")

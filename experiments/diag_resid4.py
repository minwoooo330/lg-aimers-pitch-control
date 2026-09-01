
import numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
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
def run(Xs,tag,groups=None):
    oof=np.zeros(len(s))
    if groups is None:
        sp=KFold(3,shuffle=True,random_state=0).split(Xs)
    else:
        sp=GroupKFold(3).split(Xs,groups=groups)
    for trn,val in sp:
        m=HistGradientBoostingRegressor(max_iter=100,learning_rate=0.05,max_leaf_nodes=31,
            min_samples_leaf=1000,l2_regularization=10.0,random_state=42)
        m.fit(Xs.iloc[trn],res[trn]); oof[val]=m.predict(Xs.iloc[val])
    q=np.clip(p+oof,0,1)
    gain=(np.mean(res**2)-np.mean((yy-q)**2))*1e5
    print(f"  {tag:38s} 상관 {np.corrcoef(oof,res)[0,1]:+.4f}   상한 {gain:+7.2f}e-5")
# 순수 상황 피처만 (선수 식별 정보 완전 제거)
SIT=["balls_before","strikes_before","outs_before","inning","top_bottom","game_month",
     "game_dayofweek","num_runners_on","runner_on_1b","runner_on_2b","runner_on_3b",
     "score_diff_pitcher_team","li","base_state","pitcher_hand","batter_hand","game_type"]
SIT=[c for c in SIT if c in X.columns]
print("=== 결정적 판정: 투수 단위 분할 (같은 투수가 학습/검증에 안 겹침) ===")
print("  랜덤분할에서 신호가 보였던 이유가 '투수별 시즌 편차 암기'라면 여기서 사라진다.\n")
run(X.drop(columns=["pitcher_id","batter_id"]),"랜덤 3-fold (참고)")
run(X.drop(columns=["pitcher_id","batter_id"]),"투수 그룹 3-fold (판정)",groups=s.pitcher_id.values)

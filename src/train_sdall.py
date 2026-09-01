# -*- coding: utf-8 -*-
"""kimminwoo_sdall_v1: 챔피언 sd25(1022.2183)의 sd 슬롯을 투수+타자 채널판으로 교체 + 시즌당해 성적 HGB(3시드) 20% 추가.
   exp107 전수 스캔에서 109개 재료 중 1위 (2024 +2.79 / 2022 +4.26 @20%, 상관 0.9459로 최저).
   핵심: 이 피처는 단일 모델 성능은 못 올리지만 모델이 '다른 실수'를 하게 만들어 앙상블에서 값이 된다.
   조회표(2024년 말 투수 상태)는 학습 데이터로만 만들어 zip에 넣고, test 각 행이 조회만 한다(행 독립)."""
from pathlib import Path
import time, zipfile, sys
import joblib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from features import add_features
from hfeatures import add_hfeatures
DATA=HERE/"data"/"train.csv"; OUT=HERE/"submits_common"
BASE=OUT/"kimminwoo_rcsharp_v1.zip"; ZIP=OUT/"kimminwoo_sdall_v1.zip"; PKL=OUT/"sdall_model.pkl"
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
SEEDS=[42,7,2024]; RATES=["success_rate","middle_rate","reverse_rate"]
BRATES=["success_rate","middle_rate"]
PBS=["ball_rate","strike_rate"]
MIX=["fastball_rate","breaking_rate","offspeed_rate"]

SD_CODE = """
def add_sd(d, tbl):
    f=pd.DataFrame(index=d.index)
    n=d["asof_pitcher_n"].to_numpy(np.float64)
    pid=d["pitcher_id"].to_numpy()
    n0=pd.Series(pid).map(tbl["n"]).to_numpy(np.float64)
    dn=n-n0
    valid=np.isfinite(dn)&(dn>=20)
    f["sd_logn"]=np.where(valid,np.log1p(np.maximum(dn,0)),np.nan)
    f["sd_isnew"]=(~np.isfinite(n0)).astype(np.int8)
    for r in ["success_rate","middle_rate","reverse_rate"]:
        cur=d["asof_pitcher_"+r].to_numpy(np.float64)
        prev=pd.Series(pid).map(tbl[r]).to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*n-prev*n0)/dn
        rate=np.where(valid,np.clip(rate,0.0,1.0),np.nan)
        f["sd_"+r]=rate
        f["sd_delta_"+r]=np.where(valid,rate-cur,np.nan)
    bn=d["asof_batter_n"].to_numpy(np.float64)
    bid=d["batter_id"].to_numpy()
    bn0=pd.Series(bid).map(tbl["b_n"]).to_numpy(np.float64)
    bdn=bn-bn0; bvalid=np.isfinite(bdn)&(bdn>=20)
    f["bat_logn"]=np.where(bvalid,np.log1p(np.maximum(bdn,0)),np.nan)
    f["bat_isnew"]=(~np.isfinite(bn0)).astype(np.int8)
    for r in ["success_rate","middle_rate"]:
        cur=d["asof_batter_"+r].to_numpy(np.float64)
        prev=pd.Series(bid).map(tbl["b_"+r]).to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*bn-prev*bn0)/bdn
        rate=np.where(bvalid,np.clip(rate,0.0,1.0),np.nan)
        f["bat_"+r]=rate
        f["bat_delta_"+r]=np.where(bvalid,rate-cur,np.nan)
    for r in ["ball_rate","strike_rate"]:
        cur=d["asof_pitcher_"+r].to_numpy(np.float64)
        prev=pd.Series(pid).map(tbl["p_"+r]).to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*n-prev*n0)/dn
        rate=np.where(valid,np.clip(rate,0.0,1.0),np.nan)
        f["pbs_"+r]=rate
        f["pbs_delta_"+r]=np.where(valid,rate-cur,np.nan)
    mn=d["asof_pitcher_pitchmix_n"].to_numpy(np.float64)
    mn0=pd.Series(pid).map(tbl["m_n"]).to_numpy(np.float64)
    mdn=mn-mn0; mvalid=np.isfinite(mdn)&(mdn>=20)
    f["mix_logn"]=np.where(mvalid,np.log1p(np.maximum(mdn,0)),np.nan)
    for r in ["fastball_rate","breaking_rate","offspeed_rate"]:
        cur=d["asof_pitcher_"+r].to_numpy(np.float64)
        prev=pd.Series(pid).map(tbl["m_"+r]).to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*mn-prev*mn0)/mdn
        rate=np.where(mvalid,np.clip(rate,0.0,1.0),np.nan)
        f["mix_"+r]=rate
        f["mix_delta_"+r]=np.where(mvalid,rate-cur,np.nan)
    return f

def sdall_predict(test):
    nb=joblib.load(HERE/"model"/"sdall.pkl")
    x=test[nb["base_cols"]].copy()
    for c in nb["cat_cols"]:
        x[c]=test[c].astype(str).map(nb["maps"][c]).fillna(-1).astype(np.int16)
    X=pd.concat([x,add_features(test),add_hfeatures(test),add_sd(test,nb["tbl"])],axis=1)
    X=X.reindex(columns=nb["feature_cols"])
    return np.mean([m.predict_proba(X)[:,1] for m in nb["models"]],axis=0)

"""

def main():
    t0=time.time()
    df=pd.read_csv(DATA,encoding="utf-8-sig"); y=df[TARGET].to_numpy(np.int8)
    # 조회표: 2024년 말 각 투수 상태 (학습 데이터로만)
    idx=df.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    last=df.loc[idx]
    tbl={"n":pd.Series(last.asof_pitcher_n.to_numpy(),index=last.pitcher_id.to_numpy())}
    for r in RATES:
        tbl[r]=pd.Series(last["asof_pitcher_"+r].to_numpy(),index=last.pitcher_id.to_numpy())
    bi=df.groupby("batter_id")["asof_batter_n"].idxmax(); bl=df.loc[bi]
    tbl["b_n"]=pd.Series(bl.asof_batter_n.to_numpy(),index=bl.batter_id.to_numpy())
    for r in BRATES:
        tbl["b_"+r]=pd.Series(bl["asof_batter_"+r].to_numpy(),index=bl.batter_id.to_numpy())
    for r in PBS:
        tbl["p_"+r]=pd.Series(last["asof_pitcher_"+r].to_numpy(),index=last.pitcher_id.to_numpy())
    mi=df.groupby("pitcher_id")["asof_pitcher_pitchmix_n"].idxmax(); ml=df.loc[mi]
    tbl["m_n"]=pd.Series(ml.asof_pitcher_pitchmix_n.to_numpy(),index=ml.pitcher_id.to_numpy())
    for r in MIX:
        tbl["m_"+r]=pd.Series(ml["asof_pitcher_"+r].to_numpy(),index=ml.pitcher_id.to_numpy())
    print(f"조회표 투수 {len(tbl['n']):,}명",flush=True)
    # 학습용 시즌별 조회표 (행의 시즌 S -> S-1 말 상태)
    def end_state(upto):
        s=df[df.season<=upto]
        if len(s)==0: return None
        i2=s.groupby("pitcher_id")["asof_pitcher_n"].idxmax(); l2=s.loc[i2]
        t={"n":pd.Series(l2.asof_pitcher_n.to_numpy(),index=l2.pitcher_id.to_numpy())}
        for r in RATES: t[r]=pd.Series(l2["asof_pitcher_"+r].to_numpy(),index=l2.pitcher_id.to_numpy())
        b2=s.groupby("batter_id")["asof_batter_n"].idxmax(); lb=s.loc[b2]
        t["b_n"]=pd.Series(lb.asof_batter_n.to_numpy(),index=lb.batter_id.to_numpy())
        for r in BRATES: t["b_"+r]=pd.Series(lb["asof_batter_"+r].to_numpy(),index=lb.batter_id.to_numpy())
        for r in PBS: t["p_"+r]=pd.Series(l2["asof_pitcher_"+r].to_numpy(),index=l2.pitcher_id.to_numpy())
        m2=s.groupby("pitcher_id")["asof_pitcher_pitchmix_n"].idxmax(); lm=s.loc[m2]
        t["m_n"]=pd.Series(lm.asof_pitcher_pitchmix_n.to_numpy(),index=lm.pitcher_id.to_numpy())
        for r in MIX: t["m_"+r]=pd.Series(lm["asof_pitcher_"+r].to_numpy(),index=lm.pitcher_id.to_numpy())
        return t
    tabs={S:end_state(S-1) for S in range(int(df.season.min())+1,int(df.season.max())+1)}
    ns=globals().copy(); exec(SD_CODE.replace("def sdall_predict","def _unused_sd_predict"),ns)
    add_sd=ns["add_sd"]
    parts=[]
    for S,tb in tabs.items():
        if tb is None: continue
        d=df[df.season==S]
        if len(d)==0: continue
        parts.append(add_sd(d,tb))
    d0=df[df.season==df.season.min()]
    empty=pd.DataFrame(index=d0.index,columns=parts[0].columns,dtype=float)
    sdf=pd.concat([empty]+parts).sort_index()
    print(f"학습용 sd 피처 {sdf.shape}, 유효율 {sdf.sd_success_rate.notna().mean()*100:.1f}%",flush=True)
    base_cols=[c for c in df.columns if c not in (ID,TARGET)]
    maps={c:{v:i for i,v in enumerate(sorted(df[c].dropna().astype(str).unique()))} for c in CAT}
    x=df[base_cols].copy()
    for c in CAT: x[c]=df[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
    X=pd.concat([x,add_features(df),add_hfeatures(df),sdf.reset_index(drop=True)],axis=1)
    cm=[c in CAT for c in X.columns]
    print(f"학습 데이터 {X.shape}  ({time.time()-t0:.0f}s)",flush=True)
    models=[]
    for seed in SEEDS:
        tt=time.time()
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
            min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
            random_state=seed,categorical_features=cm).fit(X,y)
        models.append(m); print(f"  seed {seed} {time.time()-tt:.0f}s",flush=True)
    joblib.dump({"models":models,"tbl":tbl,"maps":maps,"cat_cols":CAT,
                 "base_cols":base_cols,"feature_cols":list(X.columns)},PKL,compress=3)
    print("sdall pkl 저장 완료",flush=True)

if __name__=="__main__": main()

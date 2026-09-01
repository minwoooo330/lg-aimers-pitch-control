# -*- coding: utf-8 -*-
"""XGBoost + sd(시즌+타자 채널) — 체인에 아예 없던 계열. NN처럼 완전히 새 그릇이라 sd 효과가 클 수 있다.
   이전 exp83(XGBoost, sd 없음)은 체인 기여 음수로 기각됐었다. sd로 재시도."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parent))
from features import add_features
from hfeatures import add_hfeatures
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
RATES=["success_rate","middle_rate","reverse_rate"]; BRATES=["success_rate","middle_rate"]

def end_state(d, upto):
    s=d[d.season<=upto]
    if len(s)==0: return None
    idx=s.groupby("pitcher_id")["asof_pitcher_n"].idxmax(); last=s.loc[idx]
    t={"n":pd.Series(last.asof_pitcher_n.to_numpy(),index=last.pitcher_id.to_numpy())}
    for r in RATES: t[r]=pd.Series(last["asof_pitcher_"+r].to_numpy(),index=last.pitcher_id.to_numpy())
    bi=s.groupby("batter_id")["asof_batter_n"].idxmax(); bl=s.loc[bi]
    t["b_n"]=pd.Series(bl.asof_batter_n.to_numpy(),index=bl.batter_id.to_numpy())
    for r in BRATES: t["b_"+r]=pd.Series(bl["asof_batter_"+r].to_numpy(),index=bl.batter_id.to_numpy())
    return t

def build_tables(df, max_season):
    return {S: end_state(df, S-1) for S in range(int(df.season.min())+1, max_season+1)}

def add_sd(d, tables):
    n=d.asof_pitcher_n.to_numpy(np.float64); pid=d.pitcher_id.to_numpy(); seas=d.season.to_numpy()
    n0=np.full(len(d),np.nan); rr={r:np.full(len(d),np.nan) for r in RATES}
    bn0=np.full(len(d),np.nan); brr={r:np.full(len(d),np.nan) for r in BRATES}
    for S,tbl in tables.items():
        if tbl is None: continue
        mm=(seas==S)
        if not mm.any(): continue
        sub_p=pd.Series(pid[mm]); sub_b=pd.Series(d.batter_id.to_numpy()[mm])
        n0[mm]=sub_p.map(tbl["n"]).to_numpy(np.float64)
        for r in RATES: rr[r][mm]=sub_p.map(tbl[r]).to_numpy(np.float64)
        bn0[mm]=sub_b.map(tbl["b_n"]).to_numpy(np.float64)
        for r in BRATES: brr[r][mm]=sub_b.map(tbl["b_"+r]).to_numpy(np.float64)
    dn=n-n0; valid=np.isfinite(dn)&(dn>=20)
    f=pd.DataFrame(index=d.index)
    f["sd_logn"]=np.where(valid,np.log1p(np.maximum(dn,0)),np.nan)
    f["sd_isnew"]=(~np.isfinite(n0)).astype(np.int8)
    for r in RATES:
        cur=d["asof_pitcher_"+r].to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*n-rr[r]*n0)/dn
        rate=np.where(valid,np.clip(rate,0,1),np.nan)
        f["sd_"+r]=rate; f["sd_d_"+r]=np.where(valid,rate-cur,np.nan)
    bn=d.asof_batter_n.to_numpy(np.float64)
    bdn=bn-bn0; bvalid=np.isfinite(bdn)&(bdn>=20)
    f["bat_logn"]=np.where(bvalid,np.log1p(np.maximum(bdn,0)),np.nan)
    for r in BRATES:
        cur=d["asof_batter_"+r].to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*bn-brr[r]*bn0)/bdn
        rate=np.where(bvalid,np.clip(rate,0,1),np.nan)
        f["bat_"+r]=rate; f["bat_d_"+r]=np.where(bvalid,rate-cur,np.nan)
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB=build_tables(df[df.season<year], year)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,sd_on):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            parts=[x,add_features(d),add_hfeatures(d)]
            if sd_on: parts.append(add_sd(d,TAB))
            return pd.concat(parts,axis=1)
        for name,sd_on in [("base",False),("sd",True)]:
            xa=enc(tr,sd_on); xb=enc(va,sd_on)
            tt=time.time()
            m=xgb.XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=7,
                min_child_weight=50,subsample=0.8,colsample_bytree=0.7,reg_lambda=2.0,
                tree_method="hist",max_bin=256,n_jobs=-1,random_state=42,eval_metric="logloss")
            m.fit(xa,ytr)
            p=m.predict_proba(xb)[:,1]
            br=float(np.mean((p[rel]-yva[rel])**2))
            rows.append({"fold":year,"cfg":name,"n_feat":xa.shape[1],"brier_rel":br,"sec":round(time.time()-tt)})
            print(rows[-1],flush=True)
            store[(year,name)]=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p})
            del m,xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp121_xgb_sd.csv",index=False,encoding="utf-8-sig")
    for name in ["base","sd"]:
        pd.concat([store[(y,name)] for y in [2024,2022] if (y,name) in store],ignore_index=True)\
          .to_csv(RES/f"exp121_{name}_oof.csv.gz",index=False,compression="gzip")
    r=pd.DataFrame(rows); print()
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.cfg=="base")].brier_rel.iloc[0]
        bb=r[(r.fold==year)&(r.cfg=="sd")].brier_rel.iloc[0]
        print(f"  {year}: base {a:.8f} -> +sd {bb:.8f}  delta {(a-bb)*1e5:+.3f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

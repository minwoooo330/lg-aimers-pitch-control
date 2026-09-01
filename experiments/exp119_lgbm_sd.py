# -*- coding: utf-8 -*-
"""LightGBM에 sd(시즌+타자 채널) 이식 — exp105/110의 검증된 시즌별 조회표 패턴을 그대로 재사용."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"; CATS=["top_bottom","game_type","base_state"]
RATES=["success_rate","middle_rate","reverse_rate"]; BRATES=["success_rate","middle_rate"]

def add_interactions(df):
    f=df.copy()
    f["inter_pitcher_x_balls"]=f.asof_pitcher_success_rate*f.balls_before
    f["inter_pitcher_x_strikes"]=f.asof_pitcher_success_rate*f.strikes_before
    f["inter_matchup_diff"]=f.asof_pitcher_success_rate-f.asof_batter_success_rate
    f["inter_platoon_same"]=(f.pitcher_hand==f.batter_hand).astype(int)
    f["inter_pressure"]=f.num_runners_on*f.outs_before
    f["inter_li_pitcher"]=f.li*f.asof_pitcher_success_rate
    f["inter_count_diff"]=f.balls_before-f.strikes_before
    f["inter_reverse_x_base"]=f.asof_pitcher_reverse_rate*f.num_runners_on
    return f

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
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB=build_tables(df[df.season<year], year)
        chk=add_sd(va,TAB)
        print(f"[{year}] 검증행 sd 유효율 {chk.sd_success_rate.notna().mean()*100:.1f}%",flush=True)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        for name,sd_on in [("base",False),("sd",True)]:
            xtr=add_interactions(tr[base]); xva=add_interactions(va[base])
            if sd_on:
                xtr=pd.concat([xtr.reset_index(drop=True),add_sd(tr,TAB).reset_index(drop=True)],axis=1)
                xva=pd.concat([xva.reset_index(drop=True),add_sd(va,TAB).reset_index(drop=True)],axis=1)
            nums=[c for c in xtr.columns if c not in CATS]
            pre=ColumnTransformer([("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),CATS),
                                   ("num",SimpleImputer(strategy="median"),nums)])
            clf=LGBMClassifier(n_estimators=153,learning_rate=0.03331120682555046,num_leaves=61,max_depth=8,
                min_child_samples=247,reg_alpha=0.013389279116695478,reg_lambda=0.0022976624140824994,
                feature_fraction=0.905269907953647,bagging_fraction=0.8320457481218804,bagging_freq=1,
                random_state=42,n_jobs=-1,verbose=-1)
            model=Pipeline([("pre",pre),("clf",clf)])
            ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
            tt=time.time(); model.fit(xtr,ytr); p=model.predict_proba(xva)[:,1]
            br=float(np.mean((p[rel]-yva[rel])**2))
            rows.append({"fold":year,"cfg":name,"brier_rel":br,"sec":round(time.time()-tt)})
            print(rows[-1],flush=True)
            store[(year,name)]=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p})
            del model,xtr,xva; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp119_lgbm_sd.csv",index=False,encoding="utf-8-sig")
    for name in ["base","sd"]:
        pd.concat([store[(y,name)] for y in [2024,2022] if (y,name) in store],ignore_index=True)\
          .to_csv(RES/f"exp119_{name}_oof.csv.gz",index=False,compression="gzip")
    r=pd.DataFrame(rows); print()
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.cfg=="base")].brier_rel.iloc[0]
        bb=r[(r.fold==year)&(r.cfg=="sd")].brier_rel.iloc[0]
        print(f"  {year}: base {a:.8f} -> +sd {bb:.8f}  delta {(a-bb)*1e5:+.3f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

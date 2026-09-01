# -*- coding: utf-8 -*-
"""실험 103: 시즌 당해 성적 피처를 실제 모델에 투입 (3시드 x 2 clean fold).
   각 fold에서 조회표는 '학습 구간 마지막 시즌 말' 상태로 만든다 (2025 제출 시 2024 말과 동일 구조)."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from features import add_features
from hfeatures import add_hfeatures
sys.stdout.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
SEEDS=[42,7,2024]
RATES=["success_rate","middle_rate","reverse_rate","ball_rate","strike_rate"]

def season_end_state(d, upto):
    s=d[d.season<=upto]
    idx=s.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    last=s.loc[idx]
    out=last[["pitcher_id","asof_pitcher_n"]].copy()
    for r in RATES: out["r_"+r]=last["asof_pitcher_"+r].to_numpy()
    return out.set_index("pitcher_id")

def add_season_todate(d, tbl):
    n=d.asof_pitcher_n.to_numpy(np.float64)
    n0=d.pitcher_id.map(tbl.asof_pitcher_n).to_numpy(np.float64)
    f=pd.DataFrame(index=d.index)
    dn=n-n0
    f["sd_n"]=dn
    f["sd_isnew"]=(~np.isfinite(n0)).astype(np.int8)
    K=100.0; w=dn/(dn+K)
    for r in RATES:
        cur=d["asof_pitcher_"+r].to_numpy(np.float64)
        prev=d.pitcher_id.map(tbl["r_"+r]).to_numpy(np.float64)
        rate=np.where(dn>0,(cur*n-prev*n0)/np.maximum(dn,1),np.nan)
        f["sd_"+r]=rate
        f["sd_delta_"+r]=rate-cur
        f["sd_shrunk_"+r]=np.where(np.isfinite(rate), w*rate+(1-w)*cur, cur)
    f["sd_w"]=w
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        tbl=season_end_state(df, year-1)          # 학습 구간 마지막 시즌 말
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,sd_on):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            parts=[x,add_features(d),add_hfeatures(d)]
            if sd_on: parts.append(add_season_todate(d,tbl))
            return pd.concat(parts,axis=1)
        for sd_on in [False,True]:
            xa,xb=enc(tr,sd_on),enc(va,sd_on)
            cm=[c in CAT for c in xa.columns]
            ps=[]
            for seed in SEEDS:
                m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                    min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                    random_state=seed,categorical_features=cm).fit(xa,ytr)
                p=m.predict_proba(xb)[:,1]; ps.append(p)
                rows.append({"fold":year,"sd":sd_on,"seed":seed,
                    "brier_rel":float(np.mean((p[rel]-yva[rel])**2))})
                print(rows[-1],flush=True); del m; gc.collect()
            avg=np.mean(ps,axis=0)
            rows.append({"fold":year,"sd":sd_on,"seed":"avg3",
                "brier_rel":float(np.mean((avg[rel]-yva[rel])**2))})
            print(rows[-1],flush=True)
            store[(year,sd_on)]=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,
                TARGET:yva,"prediction":avg})
            del xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp103_season_todate.csv",index=False,encoding="utf-8-sig")
    for sd_on in [False,True]:
        pd.concat([store[(y,sd_on)] for y in [2024,2022] if (y,sd_on) in store],ignore_index=True)\
          .to_csv(RES/f"exp103_{'sd' if sd_on else 'base'}_oof.csv.gz",index=False,compression="gzip")
    r=pd.DataFrame(rows)
    print()
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.seed=="avg3")&(~r.sd)].brier_rel.iloc[0]
        b_=r[(r.fold==year)&(r.seed=="avg3")&(r.sd)].brier_rel.iloc[0]
        print(f"[3시드 평균] {year}: base {a:.8f} -> +시즌당해 {b_:.8f}  delta {(a-b_)*1e5:+.3f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""실험 105: 시즌 당해 성적 피처 (올바른 구성).
   exp103/103b의 버그: 조회표를 학습구간 마지막 시즌 말 하나로만 만들어 학습 행의 dn이 음수/결측이 됐다.
   여기서는 시즌별 조회표를 만들어 '시즌 S 행 -> S-1 시즌 말 상태'로 맞춘다.
   test(2025 -> 2024년 말)와 의미가 정확히 일치하며, 어느 행도 다른 test 행을 쓰지 않는다."""
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
RATES=["success_rate","middle_rate","reverse_rate"]

def end_state(d, upto):
    s=d[d.season<=upto]
    if len(s)==0: return None
    idx=s.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    last=s.loc[idx]
    out=last[["pitcher_id","asof_pitcher_n"]].copy()
    for r in RATES: out["r_"+r]=last["asof_pitcher_"+r].to_numpy()
    return out.set_index("pitcher_id")

def build_tables(df, max_season):
    """S-1 시즌 말 상태를 시즌별로 미리 만든다 (학습 데이터로만 구성)."""
    return {S: end_state(df, S-1) for S in range(df.season.min()+1, max_season+1)}

def add_sd(d, tables):
    f=pd.DataFrame(index=d.index)
    n=d.asof_pitcher_n.to_numpy(np.float64)
    n0=np.full(len(d), np.nan); prev={r:np.full(len(d),np.nan) for r in RATES}
    seas=d.season.to_numpy()
    for S,tbl in tables.items():
        if tbl is None: continue
        m=(seas==S)
        if not m.any(): continue
        pid=d.pitcher_id.to_numpy()[m]
        n0[m]=pd.Series(pid).map(tbl.asof_pitcher_n).to_numpy(np.float64)
        for r in RATES:
            prev[r][m]=pd.Series(pid).map(tbl["r_"+r]).to_numpy(np.float64)
    dn=n-n0
    valid=np.isfinite(dn)&(dn>=20)
    f["sd_logn"]=np.where(valid,np.log1p(np.maximum(dn,0)),np.nan)
    f["sd_isnew"]=(~np.isfinite(n0)).astype(np.int8)
    for r in RATES:
        cur=d["asof_pitcher_"+r].to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*n-prev[r]*n0)/dn
        rate=np.where(valid,np.clip(rate,0.0,1.0),np.nan)
        f["sd_"+r]=rate
        f["sd_delta_"+r]=np.where(valid,rate-cur,np.nan)
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        tables=build_tables(df[df.season<year], year)      # 학습 구간만으로 조회표 구성
        tables[year]=end_state(df[df.season<year], year-1) # 검증 시즌용
        chk=add_sd(tr,tables); chk2=add_sd(va,tables)
        print(f"[{year}] 학습행 당해성적 유효 {chk.sd_success_rate.notna().mean()*100:.1f}% / "
              f"검증행 {chk2.sd_success_rate.notna().mean()*100:.1f}%",flush=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,on):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            parts=[x,add_features(d),add_hfeatures(d)]
            if on: parts.append(add_sd(d,tables))
            return pd.concat(parts,axis=1)
        for on in [False,True]:
            xa,xb=enc(tr,on),enc(va,on)
            cm=[cc in CAT for cc in xa.columns]
            ps=[]
            for seed in SEEDS:
                m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                    min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                    random_state=seed,categorical_features=cm).fit(xa,ytr)
                p=m.predict_proba(xb)[:,1]; ps.append(p)
                rows.append({"fold":year,"sd":on,"seed":seed,"brier_rel":float(np.mean((p[rel]-yva[rel])**2))})
                print(rows[-1],flush=True); del m; gc.collect()
            avg=np.mean(ps,axis=0)
            rows.append({"fold":year,"sd":on,"seed":"avg3","brier_rel":float(np.mean((avg[rel]-yva[rel])**2))})
            print(rows[-1],flush=True)
            store[(year,on)]=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":avg})
            del xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp105_season_todate.csv",index=False,encoding="utf-8-sig")
    for on in [False,True]:
        pd.concat([store[(y,on)] for y in [2024,2022] if (y,on) in store],ignore_index=True)\
          .to_csv(RES/f"exp105_{'sd' if on else 'base'}_oof.csv.gz",index=False,compression="gzip")
    r=pd.DataFrame(rows); print()
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.seed=="avg3")&(~r.sd)].brier_rel.iloc[0]
        b_=r[(r.fold==year)&(r.seed=="avg3")&(r.sd)].brier_rel.iloc[0]
        print(f"[3시드 평균] {year}: base {a:.8f} -> +시즌당해 {b_:.8f}  delta {(a-b_)*1e5:+.3f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

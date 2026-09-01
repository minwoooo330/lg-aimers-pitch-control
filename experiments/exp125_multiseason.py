# -*- coding: utf-8 -*-
"""exp125 — 다중 시즌 분해(multi-season decomposition). sd의 '기간 축' 확장.

  sd는 조회표를 1개 시점(직전 시즌 말)만 써서 '당해 시즌'을 분리했다.
  조회표를 2개 시점 겹쳐 빼면 '작년 한 시즌만'이 분리되고, 3개면 '재작년'까지 나온다.
    tables[S]   = 통산(S-1 말까지)  (n1,r1)
    tables[S-1] = 통산(S-2 말까지)  (n2,r2)
    작년 한 시즌 = (n1*r1 - n2*r2)/(n1-n2)
  모델은 현재 통산(공식 asof)과 당해(sd)만 알고, '작년 성적'·'시즌간 추세'는 모른다.
  전부 pitcher_id/batter_id로 조회표만 참조하므로 test 행 간 참조 없음(sd와 동일한 규정 준수).

  1단계: HGB 빠른 스크린 (sd만) vs (sd+다중시즌). 신호 있으면 2단계에서 NN으로."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
    """S-1뿐 아니라 더 과거 시점도 필요하므로 가능한 모든 시즌 경계를 만든다."""
    lo=int(df.season.min())
    return {S: end_state(df, S-1) for S in range(lo, max_season+1)}

def add_sd(d, tables):
    """기존 sd(당해 시즌) — 변경 없음."""
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

def _lookup(ids, seas, tables, offset, keys):
    """season S인 행에 대해 tables[S-offset] (= S-offset-1 시즌 말 통산상태)를 조회."""
    out={k:np.full(len(ids),np.nan) for k in keys}
    for S in np.unique(seas):
        tbl=tables.get(int(S)-offset)
        if tbl is None: continue
        mm=(seas==S)
        sub=pd.Series(ids[mm])
        for k in keys: out[k][mm]=sub.map(tbl[k]).to_numpy(np.float64)
    return out

def add_multiseason(d, tables):
    """조회표를 2~3시점 겹쳐 빼서 '작년 한 시즌', '재작년 한 시즌', 시즌간 추세를 만든다."""
    seas=d.season.to_numpy(); pid=d.pitcher_id.to_numpy(); bid=d.batter_id.to_numpy()
    f=pd.DataFrame(index=d.index)
    pk=["n"]+RATES
    t1=_lookup(pid,seas,tables,0,pk)   # S-1 시즌 말 통산
    t2=_lookup(pid,seas,tables,1,pk)   # S-2 시즌 말 통산
    t3=_lookup(pid,seas,tables,2,pk)   # S-3 시즌 말 통산
    def season_alone(ta,tb,key,prefix):
        na,nb=ta["n"],tb["n"]
        dn=na-nb; valid=np.isfinite(dn)&(dn>=20)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(ta[key]*na-tb[key]*nb)/dn
        rate=np.where(valid,np.clip(rate,0,1),np.nan)
        return rate, np.where(valid,np.log1p(np.maximum(dn,0)),np.nan)
    # 작년(S-1) 한 시즌만
    for r in RATES:
        rate,logn=season_alone(t1,t2,r,"ls")
        f["ls_"+r]=rate
        if r=="success_rate": f["ls_logn"]=logn
    # 재작년(S-2) 한 시즌만
    r2_rate,r2_logn=season_alone(t2,t3,"success_rate","ps")
    f["ps_success_rate"]=r2_rate; f["ps_logn"]=r2_logn
    # 시즌간 추세 (작년 - 재작년)
    f["trend_ls_ps"]=f["ls_success_rate"]-f["ps_success_rate"]
    # 작년 대비 통산(작년 이전까지) — 커리어 대비 작년이 좋았나
    f["ls_vs_career"]=f["ls_success_rate"]-t2["success_rate"]
    # 경력 단계: 작년 말 통산 투구수 / 시즌 경험 여부
    f["career_logn_prev"]=np.log1p(np.maximum(t1["n"],0))
    f["has_ls"]=np.isfinite(f["ls_success_rate"]).astype(np.int8)
    f["has_ps"]=np.isfinite(f["ps_success_rate"]).astype(np.int8)
    # 타자도 동일하게 작년 한 시즌
    bk=["b_n"]+["b_"+r for r in BRATES]
    b1=_lookup(bid,seas,tables,0,bk); b2=_lookup(bid,seas,tables,1,bk)
    dnb=b1["b_n"]-b2["b_n"]; vb=np.isfinite(dnb)&(dnb>=20)
    for r in BRATES:
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(b1["b_"+r]*b1["b_n"]-b2["b_"+r]*b2["b_n"])/dnb
        f["bls_"+r]=np.where(vb,np.clip(rate,0,1),np.nan)
    f["bls_logn"]=np.where(vb,np.log1p(np.maximum(dnb,0)),np.nan)
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB=build_tables(df[df.season<year], year)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        relm=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,multi):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            parts=[x,add_features(d),add_hfeatures(d),add_sd(d,TAB)]
            if multi: parts.append(add_multiseason(d,TAB))
            return pd.concat(parts,axis=1)
        for name,multi in [("sd_only",False),("sd_multi",True)]:
            xa=enc(tr,multi); xb=enc(va,multi)
            tt=time.time()
            m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.0,max_bins=255,random_state=42)
            m.fit(xa,ytr); p=m.predict_proba(xb)[:,1]
            z=p[relm]+(yva[relm].mean()-p[relm].mean())
            br=float(np.mean((z-yva[relm])**2))
            rows.append({"fold":year,"cfg":name,"n_feat":xa.shape[1],"brier_rel_aligned":br,"sec":round(time.time()-tt)})
            print(rows[-1],flush=True)
            del m,xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp125_multiseason.csv",index=False,encoding="utf-8-sig")
    r=pd.DataFrame(rows); print()
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.cfg=="sd_only")].brier_rel_aligned.iloc[0]
        bq=r[(r.fold==year)&(r.cfg=="sd_multi")].brier_rel_aligned.iloc[0]
        print(f"  {year}: sd만 {a:.8f} -> +다중시즌 {bq:.8f}   {(a-bq)*1e5:+.2f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

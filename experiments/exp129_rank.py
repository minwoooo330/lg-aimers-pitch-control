# -*- coding: utf-8 -*-
"""exp129 — 리그 상대 순위/백분위(rank & percentile). 제미나이 제안 5종 중 유일한 미시도 축.

  동기: 우리의 1순위 구조적 문제는 시즌 드리프트다(리그 성공률 0.5495->0.4897, ABS 체제전환).
        성공률 0.52는 2019년엔 평균 이하, 2024년엔 평균 한참 위인데 모델은 절대값을 그대로 먹는다.
        지금까지 드리프트 대응은 전부 출력단(shift/샤프닝)이었고 입력단에서 고친 적이 없다.
        순위/백분위는 리그 수준이 통째로 내려가도 불변이므로 시즌을 건너 안정적이다.

  규정 준수: 기준 모집단은 tables[S](= S-1 시즌 말 상태, 학습데이터로만 제작)이다.
             2025 test 행은 2024년 말 분포를 참조하며, 다른 test 행을 보지 않는다. sd와 동일 구조.
  주의: 같은 시즌 내 순위는 불가하다(2025 행끼리 비교해야 하므로 규정 위반). 반드시 직전 시즌 말 기준.

  판정 기준선(결정론적, exp125/exp128에서 17자리까지 동일하게 재현됨):
    sd_base 2024 = 0.24788740459202305 / 2022 = 0.24849479424027926"""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from features import add_features
from hfeatures import add_hfeatures
import importlib.util
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
spec=importlib.util.spec_from_file_location("ms",HERE/"exp125_multiseason.py")
ms=importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
ID,TARGET="row_id","control_success"
CAT=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
MIN_N=200   # 기준 모집단에 넣을 최소 표본(소표본 잡음이 순위를 오염시키지 않게)

def _ref_dist(tbl, key, nkey="n"):
    """조회표에서 기준 모집단의 정렬된 값 배열과 평균/표준편차를 만든다."""
    if tbl is None: return None
    n=tbl[nkey]; v=tbl[key]
    ok=(n>=MIN_N)&np.isfinite(v)
    vals=np.sort(v[ok].to_numpy(np.float64))
    if len(vals)<20: return None
    return {"sorted":vals,"mean":float(vals.mean()),"std":float(vals.std() or 1.0),"n":len(vals)}

def _pct(ref, x):
    """백분위(0~1). 기준 분포에서 x보다 작은 값의 비율."""
    if ref is None: return np.full(len(x),np.nan)
    out=np.searchsorted(ref["sorted"],x,side="left")/ref["n"]
    return np.where(np.isfinite(x),out,np.nan)

def _z(ref, x):
    if ref is None: return np.full(len(x),np.nan)
    return np.where(np.isfinite(x),(x-ref["mean"])/ref["std"],np.nan)

def add_rank_features(d, tables, sd):
    """투수/타자의 리그 상대 위치. 전부 tables(직전 시즌 말) 기준이라 규정 준수."""
    seas=d.season.to_numpy(); pid=d.pitcher_id.to_numpy(); bid=d.batter_id.to_numpy()
    f=pd.DataFrame(index=d.index, dtype=np.float64)
    cols=["p_pct_success","p_pct_middle","p_pct_reverse","p_z_success","p_pct_workload",
          "sd_pct_success","sd_z_success","pct_delta_form",
          "b_pct_success","b_z_success"]
    for c in cols: f[c]=np.nan
    cur_s=d.asof_pitcher_success_rate.to_numpy(np.float64)
    cur_m=d.asof_pitcher_middle_rate.to_numpy(np.float64)
    cur_r=d.asof_pitcher_reverse_rate.to_numpy(np.float64)
    cur_n=d.asof_pitcher_n.to_numpy(np.float64)
    cur_b=d.asof_batter_success_rate.to_numpy(np.float64)
    sd_s =sd["sd_success_rate"].to_numpy(np.float64)
    for S in np.unique(seas):
        tbl=tables.get(int(S))
        mm=(seas==S)
        if not mm.any(): continue
        r_s=_ref_dist(tbl,"success_rate"); r_m=_ref_dist(tbl,"middle_rate")
        r_r=_ref_dist(tbl,"reverse_rate"); r_n=_ref_dist(tbl,"n")
        r_b=_ref_dist(tbl,"b_success_rate","b_n")
        f.loc[mm,"p_pct_success"]=_pct(r_s,cur_s[mm]); f.loc[mm,"p_z_success"]=_z(r_s,cur_s[mm])
        f.loc[mm,"p_pct_middle"] =_pct(r_m,cur_m[mm])
        f.loc[mm,"p_pct_reverse"]=_pct(r_r,cur_r[mm])
        f.loc[mm,"p_pct_workload"]=_pct(r_n,cur_n[mm])
        # 당해 폼을 같은 기준 분포에 투영 -> "지금 폼이 리그에서 어느 위치인가"
        f.loc[mm,"sd_pct_success"]=_pct(r_s,sd_s[mm]); f.loc[mm,"sd_z_success"]=_z(r_s,sd_s[mm])
        f.loc[mm,"b_pct_success"]=_pct(r_b,cur_b[mm]); f.loc[mm,"b_z_success"]=_z(r_b,cur_b[mm])
    # 순위로 본 폼 변화: 통산 위치 대비 올해 위치가 올랐나 내렸나
    f["pct_delta_form"]=f["sd_pct_success"]-f["p_pct_success"]
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    BASE={2024:0.24788740459202305, 2022:0.24849479424027926}
    rows=[]
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB=ms.build_tables(df[df.season<year], year)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        relm=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            sd=ms.add_sd(d,TAB)
            return pd.concat([x,add_features(d),add_hfeatures(d),sd,add_rank_features(d,TAB,sd)],axis=1)
        xa=enc(tr); xb=enc(va)
        if year==2024:
            rf=add_rank_features(va,TAB,ms.add_sd(va,TAB))
            print("\n[커버리지 %]"); print((rf.notna().mean()*100).round(1).to_string())
            print("\n[분포 점검]"); print(rf.describe().T[["mean","std","min","max"]].round(4).to_string())
            print(f"\n피처 수 {xa.shape[1]} (sd_base 160 + 순위 {rf.shape[1]})\n",flush=True)
        tt=time.time()
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.06,max_leaf_nodes=31,
            min_samples_leaf=200,l2_regularization=1.0,max_bins=255,random_state=42)
        m.fit(xa,ytr); p=m.predict_proba(xb)[:,1]
        z=p[relm]+(yva[relm].mean()-p[relm].mean())
        br=float(np.mean((z-yva[relm])**2))
        rows.append({"fold":year,"cfg":"sd_rank","n_feat":xa.shape[1],"brier_rel_aligned":br,
                     "base":BASE[year],"delta_e5":round((BASE[year]-br)*1e5,2),"sec":round(time.time()-tt)})
        print(rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(RES/"exp129_rank.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":p}).to_csv(
            RES/f"exp129_rank_oof_{year}.csv.gz",index=False,compression="gzip")
        del m,xa,xb; gc.collect()
    print()
    for r in rows:
        print(f"  {r['fold']}: sd만 {r['base']:.8f} -> +순위 {r['brier_rel_aligned']:.8f}   {r['delta_e5']:+.2f}e-5")
    print("\n노이즈 바닥 1sigma: 2024 ~3.96e-5 / 2022 ~3.35e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

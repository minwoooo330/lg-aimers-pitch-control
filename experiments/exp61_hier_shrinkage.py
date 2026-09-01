# -*- coding: utf-8 -*-
"""실험 61: 계층적 shrinkage 조건부 통계표 (아이디어 3).

exp40이 확인한 조건부 표(투수x타자손잡이 등)를 앙상블 멤버로 만들되,
shrinkage 방식 3종을 3 fold 전부에서 고정 비교한다.
  V1 exp40식      : K=200 고정, 투수 주변값 방향으로 축소
  V2 EB           : 셀 종류별 K를 베타-이항 적률로 추정
  V3 EB+시즌감쇠  : 과거 시즌에 0.5^(경과연수) 가중 후 EB
채택 조건: 단독으로 3 fold 모두 base 개선 + 앙상블 기여도 3 fold 모두 양수.
"""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
CELLS={"hand":["pitcher_id","batter_hand"],
       "cnt":["pitcher_id","_cnt"],
       "handadv":["pitcher_id","batter_hand","_adv"]}
VARIANTS=["base","V1","V2","V3"]

def prep(d):
    d=d.copy()
    d["_cnt"]=d.balls_before.astype(str)+d.strikes_before.astype(str)
    d["_adv"]=np.sign(d.strikes_before-d.balls_before).astype(int)
    return d

def eb_k(s,n,pm,prior):
    """베타-이항 적률법으로 shrinkage 강도 K 추정."""
    ok=n>=20
    if ok.sum()<50: return 200.0
    obs=(s[ok]/n[ok]).to_numpy(); base=pm[ok].to_numpy(); nn=n[ok].to_numpy()
    tot=np.mean((obs-base)**2)
    noise=np.mean(base*(1-base)/nn)
    tau2=max(tot-noise,1e-6)
    return float(np.clip(prior*(1-prior)/tau2,10.0,5000.0))

def build(hist,variant):
    """hist(과거 시즌 행)로 조건부 편차 표를 만든다."""
    if len(hist)==0: return None
    h=hist
    if variant=="V3":
        smax=h.season.max()
        w=np.power(0.5,(smax-h.season).to_numpy().astype(float))
    else:
        w=np.ones(len(h),dtype=float)
    yv=h[TARGET].to_numpy().astype(float)
    tmp=pd.DataFrame({"pitcher_id":h.pitcher_id.to_numpy(),
                      "batter_hand":h.batter_hand.to_numpy(),
                      "_cnt":h._cnt.to_numpy(),"_adv":h._adv.to_numpy(),
                      "_w":w,"_wy":w*yv})
    prior=float(tmp._wy.sum()/tmp._w.sum())
    gm=tmp.groupby("pitcher_id")[["_w","_wy"]].sum()
    K0=200.0 if variant=="V1" else eb_k(gm["_wy"],gm["_w"],
            pd.Series(prior,index=gm.index),prior)
    marg=(gm["_wy"]+K0*prior)/(gm["_w"]+K0)
    t={"marg":marg,"prior":prior}
    for name,cols in CELLS.items():
        gg=tmp.groupby(cols,observed=True)[["_w","_wy"]].sum()
        pm=pd.Series(gg.index.get_level_values(0)).map(marg)
        pm.index=gg.index
        K=200.0 if variant=="V1" else eb_k(gg["_wy"],gg["_w"],pm,prior)
        t[name]=((gg["_wy"]+K*pm)/(gg["_w"]+K))-pm
        t[name+"_n"]=np.log1p(gg["_w"])
        t[name+"_K"]=K
    return t

def apply_t(d,t):
    out=pd.DataFrame(index=range(len(d)))
    if t is None:
        for name in CELLS: out[f"pt_dev_{name}"]=np.nan; out[f"pt_n_{name}"]=np.nan
        return out
    keys={"hand":[d.pitcher_id,d.batter_hand],"cnt":[d.pitcher_id,d._cnt],
          "handadv":[d.pitcher_id,d.batter_hand,d._adv]}
    for name,arrs in keys.items():
        idx=pd.MultiIndex.from_arrays([a.to_numpy() for a in arrs])
        out[f"pt_dev_{name}"]=t[name].reindex(idx).to_numpy()
        out[f"pt_n_{name}"]=t[name+"_n"].reindex(idx).to_numpy()
    return out

def main():
    t0=time.time(); df=prep(pd.read_csv(DATA,encoding="utf-8-sig"))
    cols=[c for c in df.columns if c not in (ID,TARGET,"_cnt","_adv")]
    rows=[]; store={v:[] for v in VARIANTS if v!="base"}; store["base"]=[]
    for year in [2022,2023,2024]:
        tr=df[df.season<year].sort_values("season").reset_index(drop=True)
        va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        xt0,xv0=tr[cols].copy(),va[cols].copy()
        for c in CATS:
            vals=sorted(tr[c].dropna().astype(str).unique()); m={v:i for i,v in enumerate(vals)}
            xt0[c]=tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
            xv0[c]=va[c].astype(str).map(m).fillna(-1).astype(np.int16)
        xt0=pd.concat([xt0,add_features(tr)],axis=1); xv0=pd.concat([xv0,add_features(va)],axis=1)
        for variant in VARIANTS:
            tt=time.time()
            if variant=="base":
                xt,xv=xt0,xv0
            else:
                parts=[]
                for s in sorted(tr.season.unique()):
                    sub=tr[tr.season==s]
                    parts.append(apply_t(sub,build(df[df.season<s],variant)))
                tf=pd.concat(parts,ignore_index=True)
                vf=apply_t(va,build(df[df.season<year],variant))
                xt=pd.concat([xt0,tf],axis=1); xv=pd.concat([xv0,vf],axis=1)
            mdl=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                categorical_features=[c in CATS for c in xt.columns],random_state=42).fit(xt,ytr)
            p=mdl.predict_proba(xv)[:,1]
            rows.append({"fold":year,"variant":variant,"n_feat":xt.shape[1],
                "brier":brier_score_loss(yva,p),"auc":roc_auc_score(yva,p),
                "sec":round(time.time()-tt,1)})
            print(rows[-1],flush=True)
            store[variant].append(pd.DataFrame({ID:va[ID].to_numpy(),"season":year,
                                                TARGET:yva,"prediction":p}))
            del mdl,p; gc.collect()
        del xt0,xv0,tr,va; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp61_hier_shrinkage.csv",index=False,encoding="utf-8-sig")
    for v in VARIANTS:
        pd.concat(store[v],ignore_index=True).to_csv(RES/f"exp61_{v}_oof.csv.gz",
                                                     index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

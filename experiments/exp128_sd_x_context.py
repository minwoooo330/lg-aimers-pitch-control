# -*- coding: utf-8 -*-
"""exp128 — sd(당해 폼) x 상황 상호작용 그룹. 한 번도 스캔되지 않은 조합.

  exp94(투수상수 x 맥락 곱 72개 전수스캔)는 8/21 실행이고 sd는 8/24 발견이라 구조적으로 미스캔.
  게다가 exp94의 대상은 '투수 상수'(투수마다 고정)였는데 sd는 시즌 진행에 따라 행마다 변하는 값이라
  애초에 그 프레임에 들어가지도 않았다.

  야구 질문: "올해 유독 부진한 투수가 압박 상황에서 더 크게 무너지는가?"
  sd 이전에는 '올해 부진'을 표현할 수단이 없었다(통산 성적은 몇 년치 평균이라 올해 상태를 못 잡음).

  판정: 개별 잔차상관으로 거르지 않고 그룹 일괄 투입 (도메인43이 개별 43개 중 42개 문턱미달이었으나
        그룹으로 +18.81점을 냈던 전례를 따름)."""
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

def add_sd_x_context(d, sd):
    """당해 폼 편차(sd - 통산) x 상황 지표. 폼이 상황 대응력을 바꾸는지 본다."""
    f=pd.DataFrame(index=d.index)
    # 폼 편차 3종: 성공/가운데/의도반대 각각 '올해가 통산보다 얼마나 좋은가/나쁜가'
    dev_s=sd["sd_d_success_rate"].to_numpy(np.float64)
    dev_m=sd["sd_d_middle_rate"].to_numpy(np.float64)
    dev_r=sd["sd_d_reverse_rate"].to_numpy(np.float64)
    logn =sd["sd_logn"].to_numpy(np.float64)
    balls=d.balls_before.to_numpy(np.float64); strikes=d.strikes_before.to_numpy(np.float64)
    same=(d.pitcher_hand.to_numpy()==d.batter_hand.to_numpy()).astype(np.float64)
    late=(d.inning.to_numpy()>=7).astype(np.float64)
    scor=((d.runner_on_2b.to_numpy()==1)|(d.runner_on_3b.to_numpy()==1)).astype(np.float64)
    li=np.log1p(np.maximum(d.li.to_numpy(np.float64),0))
    hi_li=(d.li.to_numpy()>=2.0).astype(np.float64)
    three_b=(balls>=3).astype(np.float64); two_s=(strikes>=2).astype(np.float64)
    adv=np.sign(strikes-balls)                       # 카운트 우열 -1/0/+1
    close=(np.abs(d.score_diff_pitcher_team.to_numpy(np.float64))<=1).astype(np.float64)
    ctx={"3ball":three_b,"2strike":two_s,"adv":adv,"same":same,"late":late,
         "scoring":scor,"hiLI":hi_li,"logLI":li,"close":close}
    # 중심화: 주효과는 이미 모델에 있으므로 상호작용만 분리
    for nm,v in ctx.items():
        vc=v-np.nanmean(v)
        f[f"sdx_s_{nm}"]=dev_s*vc
    for nm in ["3ball","2strike","adv","same"]:
        vc=ctx[nm]-np.nanmean(ctx[nm])
        f[f"sdx_m_{nm}"]=dev_m*vc
        f[f"sdx_r_{nm}"]=dev_r*vc
    # 표본 신뢰도 x 폼: 올해 표본이 적으면 폼 신호가 덜 믿을 만하다
    f["sdx_dev_x_logn"]=dev_s*(logn-np.nanmean(logn))
    # 폼 편차의 비선형(급락/급등 구간 강조)
    f["sdx_dev_sq"]=dev_s*np.abs(dev_s)
    return f

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        TAB=ms.build_tables(df[df.season<year], year)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        relm=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        cols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,inter):
            x=d[cols].copy()
            for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            sd=ms.add_sd(d,TAB)
            parts=[x,add_features(d),add_hfeatures(d),sd]
            if inter: parts.append(add_sd_x_context(d,sd))
            return pd.concat(parts,axis=1)
        for name,inter in [("sd_base",False),("sd_x_ctx",True)]:
            xa=enc(tr,inter); xb=enc(va,inter)
            tt=time.time()
            m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.06,max_leaf_nodes=31,
                min_samples_leaf=200,l2_regularization=1.0,max_bins=255,random_state=42)
            m.fit(xa,ytr); p=m.predict_proba(xb)[:,1]
            z=p[relm]+(yva[relm].mean()-p[relm].mean())
            br=float(np.mean((z-yva[relm])**2))
            rows.append({"fold":year,"cfg":name,"n_feat":xa.shape[1],"brier_rel_aligned":br,"sec":round(time.time()-tt)})
            print(rows[-1],flush=True)
            del m,xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp128_sd_x_context.csv",index=False,encoding="utf-8-sig")
    r=pd.DataFrame(rows); print()
    for year in [2024,2022]:
        a=r[(r.fold==year)&(r.cfg=="sd_base")].brier_rel_aligned.iloc[0]
        bq=r[(r.fold==year)&(r.cfg=="sd_x_ctx")].brier_rel_aligned.iloc[0]
        print(f"  {year}: sd만 {a:.8f} -> +sd x 상황 {bq:.8f}   {(a-bq)*1e5:+.2f}e-5")
    print(f"\n노이즈 바닥 1sigma: 2024 ~3.96e-5 / 2022 ~3.35e-5 (exp98 측정)")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

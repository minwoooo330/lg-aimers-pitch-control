# -*- coding: utf-8 -*-
"""빼기 실험 C: 피처 대폭 삭감. 147개 -> 12개 / 24개.
   싱가폴 캡스톤 일화의 정신: 색깔을 다 빼고 숫자만 봤더니 1등.
   작년 수상자 회고에서도 '우승자는 피처 20개 이하'였다. 우리는 한 번도 안 해봤다."""
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
# 도메인 지식으로 고른 핵심 피처 (결측 적고 중요도 높은 것 위주)
CORE12=["asof_pitcher_success_rate","asof_pitcher_n","balls_before","strikes_before",
        "asof_pitcher_middle_rate","asof_pitcher_reverse_rate","asof_batter_success_rate",
        "season","game_type","pitcher_hand","batter_hand","inning"]
CORE24=CORE12+["asof_pitcher_strike_rate","asof_pitcher_ball_rate","asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate",
        "asof_batter_n","asof_batter_middle_rate","outs_before","num_runners_on","li",
        "game_month","top_bottom"]
def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={}
    for year in [2024,2022]:
        tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
        ytr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
        rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
        allcols=[c for c in df.columns if c not in (ID,TARGET)]
        maps={c:{v:i for i,v in enumerate(sorted(tr[c].dropna().astype(str).unique()))} for c in CAT}
        def enc(d,mode):
            if mode=="full":
                x=d[allcols].copy()
                for c in CAT: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
                return pd.concat([x,add_features(d),add_hfeatures(d)],axis=1)
            use=CORE12 if mode=="core12" else CORE24
            x=d[use].copy()
            for c in CAT:
                if c in use: x[c]=d[c].astype(str).map(maps[c]).fillna(-1).astype(np.int16)
            if mode=="core24":   # abs_era 한 개만 추가 (체제 플래그)
                x["abs_era"]=((d.game_type=="F")&(d.season>=2023)).astype(np.int8)
            return x
        for mode in ["full","core24","core12"]:
            xa,xb=enc(tr,mode),enc(va,mode)
            cm=[cc in CAT for cc in xa.columns]
            ps=[]
            for seed in SEEDS:
                m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
                    min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
                    random_state=seed,categorical_features=cm).fit(xa,ytr)
                p=m.predict_proba(xb)[:,1]; ps.append(p)
                del m; gc.collect()
            avg=np.mean(ps,axis=0)
            br=float(np.mean((avg[rel]-yva[rel])**2))
            rows.append({"fold":year,"mode":mode,"n_feat":xa.shape[1],"brier_rel":br})
            print(f"  {year} {mode:7s} 피처 {xa.shape[1]:3d}개  Brier(3시드평균) {br:.8f}  ({time.time()-t0:.0f}s)",flush=True)
            store[(year,mode)]=pd.DataFrame({ID:va[ID].to_numpy(),"season":year,TARGET:yva,"prediction":avg})
            del xa,xb; gc.collect()
        pd.DataFrame(rows).to_csv(RES/"exp106_feature_cut.csv",index=False,encoding="utf-8-sig")
    for mode in ["full","core24","core12"]:
        pd.concat([store[(y,mode)] for y in [2024,2022] if (y,mode) in store],ignore_index=True)\
          .to_csv(RES/f"exp106_{mode}_oof.csv.gz",index=False,compression="gzip")
    r=pd.DataFrame(rows); print()
    for year in [2024,2022]:
        f_=r[(r.fold==year)&(r["mode"]=="full")].brier_rel.iloc[0]
        for mode in ["core24","core12"]:
            v=r[(r.fold==year)&(r["mode"]==mode)].brier_rel.iloc[0]
            print(f"  {year} {mode} vs full: {(f_-v)*1e5:+.3f}e-5")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

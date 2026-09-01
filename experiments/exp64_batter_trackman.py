# -*- coding: utf-8 -*-
"""실험 64: 타자 Trackman ID 매핑 + 타자 프로필 사전 검사 (학습 없음).

투수 매핑(660명)과 동일한 경기 지문 + 행 단위 반복 투표 절차를 타자에 적용한다.
그 뒤 '이 타자가 상대한 공'의 프로필을 만들어 2024 잔차와의 상관만 측정한다.
채택 기준: 잔차 상관 |r| >= 0.035 (exp45에서 확정한 문턱). 미달이면 모델 실험을 하지 않는다.
"""
from pathlib import Path
import time
import numpy as np, pandas as pd
from trackman_features import match_exact_games

HERE=Path(__file__).resolve().parent; RES=HERE/"results"
CUT=2024   # 2024 fold 기준: 2024 이전 Trackman만 사용

def build_batter_mapping(main_g, tm_s, matches, cutoff, min_votes=20, min_purity=0.99):
    pairs=matches[matches["season"]<cutoff]
    mi_all=main_g.groupby("_game_idx",sort=False).indices
    ti_all=tm_s.groupby("trackman_game_id",sort=False).indices
    L,Rr=[],[]
    for row in pairs.itertuples(index=False):
        mi=mi_all[row.main_game_idx]; ti=ti_all[row.trackman_game_id]
        if len(mi)!=len(ti): continue
        L.append(main_g.iloc[mi]["batter_id"].to_numpy())
        Rr.append(tm_s.iloc[ti]["batter_trackman_id"].to_numpy())
    v=pd.DataFrame({"batter_id":np.concatenate(L),"batter_trackman_id":np.concatenate(Rr)})
    votes=v.groupby(["batter_id","batter_trackman_id"]).size().rename("votes").reset_index()
    tot=votes.groupby("batter_id")["votes"].sum().rename("total_votes")
    votes=votes.join(tot,on="batter_id"); votes["purity"]=votes.votes/votes.total_votes
    best=(votes.sort_values(["batter_id","votes"],ascending=[True,False]).drop_duplicates("batter_id"))
    best=best[(best.votes>=min_votes)&(best.purity>=min_purity)]
    best=(best.sort_values(["purity","votes"],ascending=False).drop_duplicates("batter_trackman_id"))
    return best.sort_values("batter_id").reset_index(drop=True)

def batter_profile(tm, mapping, cutoff):
    t=tm[tm.season<cutoff].copy()
    t=t.merge(mapping[["batter_id","batter_trackman_id"]],on="batter_trackman_id",how="inner")
    t["is_fb"]=(t.pitch_type_group=="fastball").astype(float)
    t["is_br"]=(t.pitch_type_group=="breaking").astype(float)
    t["is_of"]=(t.pitch_type_group=="offspeed").astype(float)
    t["tot_break"]=np.hypot(t.induced_vert_break,t.horz_break)
    t["vsL"]=(t.pitcher_hand=="Left").astype(float)
    g=t.groupby("batter_id")
    p=pd.DataFrame({
      "bt_n":np.log1p(g.size()),
      "bt_speed_faced":g.rel_speed.mean(),
      "bt_speed_sd":g.rel_speed.std(),
      "bt_spin_faced":g.spin_rate.mean(),
      "bt_break_faced":g.tot_break.mean(),
      "bt_ext_faced":g.extension.mean(),
      "bt_fb_share":g.is_fb.mean(),
      "bt_br_share":g.is_br.mean(),
      "bt_of_share":g.is_of.mean(),
      "bt_vsL_share":g.vsL.mean(),
    })
    for hand,tag in [("Left","L"),("Right","R")]:
        s=t[t.pitcher_hand==hand].groupby("batter_id")
        p[f"bt_speed_vs{tag}"]=s.rel_speed.mean()
        p[f"bt_fb_share_vs{tag}"]=s.is_fb.mean()
    p["bt_speed_gap_LR"]=p.bt_speed_vsL-p.bt_speed_vsR
    p["bt_fb_gap_LR"]=p.bt_fb_share_vsL-p.bt_fb_share_vsR
    lg=p.bt_fb_share.mean()
    p["bt_fb_dev"]=p.bt_fb_share-lg
    return p.reset_index()

def main():
    t0=time.time()
    main_df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    tm=pd.read_csv(HERE/"data"/"trackman_history.csv",encoding="utf-8-sig")
    main_g,tm_s,matches=match_exact_games(main_df,tm)
    print(f"매칭 경기 {len(matches)}개  ({time.time()-t0:.0f}s)",flush=True)
    mp=build_batter_mapping(main_g,tm_s,matches,CUT)
    n_b=main_df.batter_id.nunique()
    cover=main_df[main_df.season==2024].batter_id.isin(mp.batter_id).mean()
    print(f"타자 매핑 {len(mp)} / 전체 {n_b}명, 2024 행 커버 {cover:.3f}",flush=True)
    mp.to_csv(RES/"exp64_batter_mapping.csv",index=False,encoding="utf-8-sig")

    prof=batter_profile(tm,mp,CUT)
    print(f"프로필 타자 {len(prof)}명, 피처 {prof.shape[1]-1}개",flush=True)
    prof.to_csv(RES/"exp64_batter_profile.csv.gz",index=False,compression="gzip")

    arr=np.load(RES/"champ_resid_2024.npy")
    rid=arr[:,0].astype(np.int64); resid=arr[:,1]
    v24=main_df[main_df.season==2024].copy()
    v24["_rid"]=v24.row_id.str.replace("TRAIN_","",regex=False).astype(np.int64)
    v24=v24.set_index("_rid").loc[rid].reset_index()
    v24["resid"]=resid
    v24=v24.merge(prof,on="batter_id",how="left")
    feats=[c for c in prof.columns if c!="batter_id"]
    rng=np.random.default_rng(0); v24["_noise"]=rng.normal(size=len(v24))
    rows=[]
    low=v24.asof_batter_n.fillna(0)<300
    for c in feats+["_noise"]:
        x=v24[c].to_numpy(dtype=float); ok=np.isfinite(x)
        r_all=np.corrcoef(x[ok],v24.resid.to_numpy()[ok])[0,1] if ok.sum()>1000 else np.nan
        okl=ok&low.to_numpy()
        r_low=np.corrcoef(x[okl],v24.resid.to_numpy()[okl])[0,1] if okl.sum()>1000 else np.nan
        rows.append({"feature":c,"n_rows":int(ok.sum()),"resid_corr":r_all,"resid_corr_lowN":r_low})
    out=pd.DataFrame(rows).sort_values("resid_corr",key=lambda s:s.abs(),ascending=False)
    out.to_csv(RES/"exp64_batter_screen.csv",index=False,encoding="utf-8-sig")
    print(out.to_string(index=False),flush=True)
    best=out[out.feature!="_noise"].resid_corr.abs().max()
    print(f"\n최대 |잔차 상관| = {best:.5f}  (채택 문턱 0.035) -> "
          f"{'통과: 모델 실험 진행' if best>=0.035 else '미달: 타자 Trackman 축 종료'}",flush=True)
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

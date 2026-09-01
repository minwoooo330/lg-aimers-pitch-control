# -*- coding: utf-8 -*-
"""train_nnrank — [수정] 순위 계산에는 NaN 유지 sd를 넘긴다(exp131 검증본과 일치).
   당해 기록 없는 투수의 sd가 0.0이면 "리그 최하위 백분위"로 잘못 계산되므로 반드시 NaN이어야 한다.

   원문: 리그 상대 순위/백분위 NN (전체 데이터).
   exp131 검증: 체인 기여 10~20%에서 2024 +0.98 / 2022 +0.45.
   주의: 순위 자체는 원본의 재표현이다(z점수는 원본과 상관 1.0000). HGB에 넣으면 기여 ~0이다.
   그러나 신경망에 넣으면 학습 궤적이 달라져 체인에 기여한다 — sd(HGB +13.53 -> NN +16.95)와 같은 원리.
   기준 모집단은 tables[S](직전 시즌 말, 학습데이터로만 제작)이므로 규정 준수."""
"""NN+sd를 전체 데이터로 학습해 numpy 가중치로 내보낸다 (torch 의존 없음).
   exp120 결과: 체인 기여 45% 지점 2024 +10.00e-5 / 2022 +11.28e-5 (오늘 최대)."""
from pathlib import Path
import time, sys
import joblib
import numpy as np, pandas as pd
import torch, torch.nn as nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from features import add_features
DATA=HERE/"data"/"train.csv"; OUT=HERE/"submits_common"; PKL=OUT/"nnrank_model.pkl"
ID,TARGET="row_id","control_success"
EMB=[("pitcher_id",16),("batter_id",16),("pitcher_team_id",4),("batter_team_id",4),
     ("base_state",4),("pitcher_hand",2),("batter_hand",2),("top_bottom",2),("game_type",2)]
EPOCHS=4; BATCH=8192; SEEDS=[42,7,2024]
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
    f["sd_isnew"]=(~np.isfinite(n0)).astype(np.float64)
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
    return f   # raw(NaN 유지). 호출부에서 용도에 맞게 처리한다.

class Net(nn.Module):
    def __init__(self,sizes,ndim):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(s,d) for s,(_,d) in zip(sizes,EMB)])
        self.p_same=nn.Embedding(sizes[0],16); self.p_opp=nn.Embedding(sizes[0],16)
        dim=ndim+sum(d for _,d in EMB)+16
        self.mlp=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(128,1))
    def forward(self,xc,xn,sm):
        e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
        s=sm.unsqueeze(1)
        ph=self.p_same(xc[:,0])*s+self.p_opp(xc[:,0])*(1-s)
        return self.mlp(torch.cat(e+[ph,xn],dim=1)).squeeze(1)

MIN_N=200

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
    t0=time.time()
    df=pd.read_csv(DATA,encoding="utf-8-sig"); y=df[TARGET].to_numpy(np.float32)
    TAB=build_tables(df, int(df.season.max())+1)
    sdfeat_raw=add_sd(df, TAB)          # 순위 계산용: NaN 유지 (exp131 검증본과 동일)
    sdfeat=sdfeat_raw.fillna(0.0)       # NN 입력용: 0 대체
    print(f"sd 피처 {sdfeat.shape}  ({time.time()-t0:.0f}s)",flush=True)
    cat=np.zeros((len(df),len(EMB)),dtype=np.int64); sizes=[]; vocabs=[]
    for j,(c,_) in enumerate(EMB):
        vals=sorted(df[c].dropna().astype(str).unique()); mp={v:i+1 for i,v in enumerate(vals)}
        cat[:,j]=df[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        sizes.append(len(vals)+1); vocabs.append(mp)
    cn={c for c,_ in EMB}
    num_cols=[c for c in df.columns if c not in cn and c not in (ID,TARGET)]
    rkfeat=add_rank_features(df, TAB, sdfeat_raw)
    xn=pd.concat([df[num_cols],add_features(df),sdfeat,rkfeat],axis=1).astype(np.float32)
    feat=list(xn.columns); med=xn.median(); xn=xn.fillna(med)
    mu,sd=xn.mean(),xn.std().replace(0,1); xn=((xn-mu)/sd).to_numpy(np.float32)
    same=(df.pitcher_hand.to_numpy()==df.batter_hand.to_numpy()).astype(np.float32)
    Xc=torch.from_numpy(cat); Xn=torch.from_numpy(xn); Y=torch.from_numpy(y); SM=torch.from_numpy(same)
    n=len(df); idx=np.arange(n); nets=[]
    print(f"데이터 {xn.shape} ({time.time()-t0:.0f}s)",flush=True)
    lossf=nn.BCEWithLogitsLoss()
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); tt=time.time()
        net=Net(sizes,xn.shape[1])
        opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
        for ep in range(EPOCHS):
            np.random.shuffle(idx); net.train()
            for st in range(0,n,BATCH):
                bb=idx[st:st+BATCH]; opt.zero_grad()
                lossf(net(Xc[bb],Xn[bb],SM[bb]),Y[bb]).backward(); opt.step()
        net.eval(); L=net.mlp
        nets.append({"emb":[e.weight.detach().numpy().astype(np.float32) for e in net.embs],
            "p_same":net.p_same.weight.detach().numpy().astype(np.float32),
            "p_opp":net.p_opp.weight.detach().numpy().astype(np.float32),
            "W1":L[0].weight.detach().numpy(),"b1":L[0].bias.detach().numpy(),
            "W2":L[3].weight.detach().numpy(),"b2":L[3].bias.detach().numpy(),
            "W3":L[6].weight.detach().numpy(),"b3":L[6].bias.detach().numpy()})
        print(f"  seed {seed} {time.time()-tt:.0f}s",flush=True)
    joblib.dump({"nets":nets,"vocabs":vocabs,"emb_cols":[c for c,_ in EMB],
        "num_cols":num_cols,"feat_names":feat,"med":med.to_dict(),
        "mu":mu.to_dict(),"sd":sd.to_dict(),
        "tables":TAB},PKL,compress=3)
    print("nnsd pkl 저장 완료",round(time.time()-t0,1),"s")
if __name__=="__main__": main()

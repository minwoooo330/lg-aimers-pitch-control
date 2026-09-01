# -*- coding: utf-8 -*-
"""train_nntm — NN + sd + Trackman 프로필 (전체 데이터).
   exp122 검증: 체인 기여 10~15%에서 2024 +1.18 / 2022 +0.14.
   Trackman 상한(0.9점)은 HGB 기준 계산이었고, 신경망 그릇에서는 소폭이나마 기여가 남아 있다.
   프로필은 학습데이터(2019~2024 Trackman)로 만든 투수별 고정표이며 pkl에 담아 배포한다.
   추론 시 test 행은 자기 pitcher_id로 조회만 하므로 행 간 참조가 없다(규정 준수)."""
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
from trackman_features import match_exact_games, build_pitcher_mapping, build_trackman_profile
DATA=HERE/"data"/"train.csv"; OUT=HERE/"submits_common"; PKL=OUT/"nntm_model.pkl"
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
    return f.fillna(0.0)

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

def main():
    t0=time.time()
    df=pd.read_csv(DATA,encoding="utf-8-sig"); y=df[TARGET].to_numpy(np.float32)
    tmraw=pd.read_csv(HERE/"data"/"trackman_history.csv",encoding="utf-8-sig")
    _mg,_ts,_mt=match_exact_games(df,tmraw)
    _cut=int(df.season.max())+1
    _mp=build_pitcher_mapping(_mg,_ts,_mt,_cut)
    TMPROF=build_trackman_profile(tmraw,_mp,_cut)
    TMPROF=TMPROF[TMPROF.select_dtypes(include=[np.number]).columns.tolist()]
    print(f"Trackman 프로필 {TMPROF.shape} (경기매칭 {len(_mt)})",flush=True)
    TAB=build_tables(df, int(df.season.max())+1)
    sdfeat=add_sd(df, TAB)
    print(f"sd 피처 {sdfeat.shape}  ({time.time()-t0:.0f}s)",flush=True)
    cat=np.zeros((len(df),len(EMB)),dtype=np.int64); sizes=[]; vocabs=[]
    for j,(c,_) in enumerate(EMB):
        vals=sorted(df[c].dropna().astype(str).unique()); mp={v:i+1 for i,v in enumerate(vals)}
        cat[:,j]=df[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        sizes.append(len(vals)+1); vocabs.append(mp)
    cn={c for c,_ in EMB}
    num_cols=[c for c in df.columns if c not in cn and c not in (ID,TARGET)]
    tmfeat=df[["pitcher_id"]].join(TMPROF,on="pitcher_id").drop(columns=["pitcher_id"])
    xn=pd.concat([df[num_cols],add_features(df),sdfeat,tmfeat],axis=1).astype(np.float32)
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
        "tables":TAB,"tmprof":TMPROF},PKL,compress=3)
    print("nnsd pkl 저장 완료",round(time.time()-t0,1),"s")
if __name__=="__main__": main()

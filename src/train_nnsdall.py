# -*- coding: utf-8 -*-
"""전채널 sd 신경망(전체 데이터): 투수성적·볼스트라이크·타자·구종믹스."""
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
DATA=HERE/"data"/"train.csv"; OUT=HERE/"submits_common"; PKL=OUT/"nnsdall_model.pkl"
ID,TARGET="row_id","control_success"

CHANNELS={
 "pit":  ("pitcher_id","asof_pitcher_n",["success_rate","middle_rate","reverse_rate"]),
 "pitbs":("pitcher_id","asof_pitcher_n",["ball_rate","strike_rate"]),
 "bat":  ("batter_id","asof_batter_n",["success_rate","middle_rate"]),
 "mix":  ("pitcher_id","asof_pitcher_pitchmix_n",["fastball_rate","breaking_rate","offspeed_rate"]),
}
PREF={"pit":"asof_pitcher_","pitbs":"asof_pitcher_","bat":"asof_batter_","mix":"asof_pitcher_"}

def end_state(d, upto, key, ncol, rates, pref):
    s=d[d.season<=upto]
    if len(s)==0: return None
    i=s.groupby(key)[ncol].idxmax(); l=s.loc[i]
    t={"n":pd.Series(l[ncol].to_numpy(),index=l[key].to_numpy())}
    for r in rates: t[r]=pd.Series(l[pref+r].to_numpy(),index=l[key].to_numpy())
    return t

def add_ch(d, tabs, tag):
    key,ncol,rates=CHANNELS[tag]; pref=PREF[tag]
    f=pd.DataFrame(index=d.index)
    n=d[ncol].to_numpy(np.float64); ids=d[key].to_numpy(); seas=d.season.to_numpy()
    n0=np.full(len(d),np.nan); prev={r:np.full(len(d),np.nan) for r in rates}
    for S,tb in tabs.items():
        if tb is None: continue
        m=(seas==S)
        if not m.any(): continue
        sub=pd.Series(ids[m])
        n0[m]=sub.map(tb["n"]).to_numpy(np.float64)
        for r in rates: prev[r][m]=sub.map(tb[r]).to_numpy(np.float64)
    dn=n-n0; valid=np.isfinite(dn)&(dn>=20)
    f[f"{tag}_logn"]=np.where(valid,np.log1p(np.maximum(dn,0)),np.nan)
    f[f"{tag}_isnew"]=(~np.isfinite(n0)).astype(np.int8)
    for r in rates:
        cur=d[pref+r].to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"):
            rate=(cur*n-prev[r]*n0)/dn
        rate=np.where(valid,np.clip(rate,0.0,1.0),np.nan)
        f[f"{tag}_{r}"]=rate
        f[f"{tag}_d_{r}"]=np.where(valid,rate-cur,np.nan)
    return f

def build_tables(base_df, max_season):
    lo=int(base_df.season.min())
    return {tag:{S:end_state(base_df,S-1,*CHANNELS[tag],PREF[tag])
                 for S in range(lo+1,max_season+1)} for tag in CHANNELS}

def add_sd(d, tabs):
    return pd.concat([add_ch(d,tabs[t],t) for t in ["pit","pitbs","bat","mix"]],axis=1).fillna(0.0)

EMB=[("pitcher_id",16),("batter_id",16),("pitcher_team_id",4),("batter_team_id",4),
     ("base_state",4),("pitcher_hand",2),("batter_hand",2),("top_bottom",2),("game_type",2)]
EPOCHS=4; BATCH=8192; SEEDS=[42,7,2024]
RATES=["success_rate","middle_rate","reverse_rate"]; BRATES=["success_rate","middle_rate"]




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
    xn=pd.concat([df[num_cols],add_features(df),sdfeat],axis=1).astype(np.float32)
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

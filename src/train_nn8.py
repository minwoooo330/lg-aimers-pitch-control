# -*- coding: utf-8 -*-
"""8에폭 코사인 손잡이분할 NN을 전체 데이터로 학습해 numpy 가중치로 내보낸다 (torch 의존 없음)."""
from pathlib import Path
import time, sys
import joblib
import numpy as np, pandas as pd
import torch, torch.nn as nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from features import add_features
DATA=HERE/"data"/"train.csv"; OUT=HERE/"submits_common"; PKL=OUT/"nn8_model.pkl"
ID,TARGET="row_id","control_success"
EMB=[("pitcher_id",16),("batter_id",16),("pitcher_team_id",4),("batter_team_id",4),
     ("base_state",4),("pitcher_hand",2),("batter_hand",2),("top_bottom",2),("game_type",2)]
EPOCHS=8; BATCH=8192; SEEDS=[42,7,2024]

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
    cat=np.zeros((len(df),len(EMB)),dtype=np.int64); sizes=[]; vocabs=[]
    for j,(c,_) in enumerate(EMB):
        vals=sorted(df[c].dropna().astype(str).unique()); mp={v:i+1 for i,v in enumerate(vals)}
        cat[:,j]=df[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        sizes.append(len(vals)+1); vocabs.append(mp)
    cn={c for c,_ in EMB}
    num_cols=[c for c in df.columns if c not in cn and c not in (ID,TARGET)]
    xn=pd.concat([df[num_cols],add_features(df)],axis=1).astype(np.float32)
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
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
        for ep in range(EPOCHS):
            np.random.shuffle(idx); net.train()
            for st in range(0,n,BATCH):
                bb=idx[st:st+BATCH]; opt.zero_grad()
                lossf(net(Xc[bb],Xn[bb],SM[bb]),Y[bb]).backward(); opt.step()
            sch.step()
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
        "mu":mu.to_dict(),"sd":sd.to_dict()},PKL,compress=3)
    print("nn8 pkl 저장 완료",round(time.time()-t0,1),"s")
if __name__=="__main__": main()

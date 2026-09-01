# -*- coding: utf-8 -*-
"""실험 65: 투수 임베딩을 (타자손잡이 2) x (카운트 우열 3) = 6벌로 분리한 NN, 시드 3개, 3-fold.

근거: 조건부 잔차 스크리닝에서 투수x타자손잡이x카운트우열 +0.0188이 단일 축 중 최대였고
      (투수x타자손잡이 +0.0158 > 투수x카운트 +0.0101), 손잡이 단독 분할은 이미 채택되어 성공했다.
      카운트 단독 분할(exp53/54)은 2022에서 실패했으나 손잡이와의 결합은 미시도.
판정: 현 챔피언(hand8pb) 구성에 10% '추가'했을 때의 2024 fold 기여. 비중은 exp59 선례대로 사전 고정.
"""
from pathlib import Path
import gc, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
EMB=[("pitcher_id",16),("batter_id",16),("pitcher_team_id",4),("batter_team_id",4),
     ("base_state",4),("pitcher_hand",2),("batter_hand",2),("top_bottom",2),("game_type",2)]
EPOCHS=4; BATCH=8192; SEEDS=[1,13,77,101,777]

def cell_of(d):
    """0..5: (같은손 여부 2) x (불리/중립/유리 3)"""
    same=(d.pitcher_hand.to_numpy()==d.batter_hand.to_numpy()).astype(np.int64)
    adv=np.sign(d.strikes_before.to_numpy()-d.balls_before.to_numpy())+1  # 0,1,2
    return same*3+adv

def run_fold(df,year,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
    ytr=tr[TARGET].to_numpy(np.float32); yva=va[TARGET].to_numpy(np.float32)
    cat_tr=np.zeros((len(tr),len(EMB)),dtype=np.int64); cat_va=np.zeros((len(va),len(EMB)),dtype=np.int64)
    sizes=[]
    for j,(c,_) in enumerate(EMB):
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i+1 for i,v in enumerate(vals)}
        cat_tr[:,j]=tr[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        cat_va[:,j]=va[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        sizes.append(len(vals)+1)
    cn={c for c,_ in EMB}
    num_cols=[c for c in df.columns if c not in cn and c not in (ID,TARGET)]
    xn_tr=pd.concat([tr[num_cols],add_features(tr)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)
    cl_tr=cell_of(tr); cl_va=cell_of(va)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs=nn.ModuleList([nn.Embedding(s,d) for s,(_,d) in zip(sizes,EMB)])
            self.pc=nn.ModuleList([nn.Embedding(sizes[0],16) for _ in range(6)])
            dim=xn_tr.shape[1]+sum(d for _,d in EMB)+16
            self.mlp=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                                   nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.15),
                                   nn.Linear(128,1))
        def forward(self,xc,xnum,cl):
            e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
            ph=torch.zeros(xc.shape[0],16)
            for g in range(6):
                m=(cl==g)
                if m.any(): ph[m]=self.pc[g](xc[m,0])
            return self.mlp(torch.cat(e+[ph,xnum],dim=1)).squeeze(1)

    net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr); C=torch.from_numpy(cl_tr)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); VC=torch.from_numpy(cl_va)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            b=idx[st:st+BATCH]; opt.zero_grad()
            lossf(net(Xc[b],Xn[b],C[b]),Y[b]).backward(); opt.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536],VC[st:st+65536])).numpy())
    return va[ID].to_numpy(), yva, np.concatenate(ps)

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig"); parts=[]; rows=[]
    for year in [2022,2023,2024]:
        ps=[]
        for seed in SEEDS:
            ids,yv,p=run_fold(df,year,seed)
            rows.append({"fold":year,"seed":seed,"brier":brier_score_loss(yv,p),
                         "auc":roc_auc_score(yv,p)})
            print(rows[-1],flush=True); ps.append(p); gc.collect()
        avg=np.mean(ps,axis=0)
        rows.append({"fold":year,"seed":"avg","brier":brier_score_loss(yv,avg),
                     "auc":roc_auc_score(yv,avg)})
        print(rows[-1],flush=True)
        d={ID:ids,"season":year,TARGET:yv.astype(np.int8),"prediction":avg}
        for si,s_ in enumerate(SEEDS): d[f"p{s_}"]=ps[si]
        parts.append(pd.DataFrame(d))
        pd.DataFrame(rows).to_csv(RES/"exp65_handcnt_seeds5.csv",index=False,encoding="utf-8-sig")
    pd.concat(parts,ignore_index=True).to_csv(RES/"exp65_handcnt_seeds5_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

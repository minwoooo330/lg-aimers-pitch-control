# -*- coding: utf-8 -*-
"""exp152 — 선수 ID를 뺀 신경망. 팀원 v7의 철학(ID는 피처가 아니라 테이블로만)을 우리 그릇에.

  v7은 pitcher_id/batter_id를 피처에서 제거하고 carry/marcel 테이블로만 실력을 표현한다.
  우리 NN은 정반대로 ID 임베딩이 핵심이다. 두 방식은 같은 정보를 전혀 다르게 부호화하므로
  오차 패턴이 크게 달라질 수 있다 -> 제3멤버 후보.
  구성: 임베딩에서 pitcher_id/batter_id 제거, 손잡이분할(p_same/p_opp)도 제거.
        대신 asof_* 와 sd 채널이 실력을 담는다(수치 피처는 그대로)."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import brier_score_loss
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
sys.path.insert(0,str(HERE))
from features import add_features
import importlib.util as _u
_s=_u.spec_from_file_location("e120",HERE/"exp120_nn_sd.py")
E=_u.module_from_spec(_s); _s.loader.exec_module(E)
ID,TARGET="row_id","control_success"
EMB=[("pitcher_team_id",4),("batter_team_id",4),("base_state",4),
     ("pitcher_hand",2),("batter_hand",2),("top_bottom",2),("game_type",2)]
EPOCHS=4; BATCH=8192; SEEDS=[42,7,2024]

class Net(nn.Module):
    def __init__(self,sizes,ndim):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(s,d) for s,(_,d) in zip(sizes,EMB)])
        dim=ndim+sum(d for _,d in EMB)
        self.mlp=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(128,1))
    def forward(self,xc,xn):
        e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e+[xn],dim=1)).squeeze(1)

def run(df,year,seed,tables):
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
    drop={"pitcher_id","batter_id"}|{c for c,_ in EMB}
    num_cols=[c for c in df.columns if c not in drop and c not in (ID,TARGET)]
    xn_tr=pd.concat([tr[num_cols],add_features(tr),E.add_sd(tr,tables)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va),E.add_sd(va,tables)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)
    net=Net(sizes,xn_tr.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            bb=idx[st:st+BATCH]; opt.zero_grad()
            lossf(net(Xc[bb],Xn[bb]),Y[bb]).backward(); opt.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536])).numpy())
    return va[ID].to_numpy(), yva, np.concatenate(ps)

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; parts=[]
    for year in [2024,2022]:
        TAB=E.build_tables(df[df.season<year], year)
        ps=[]
        for seed in SEEDS:
            ids,yv,p=run(df,year,seed,TAB); ps.append(p); gc.collect()
        avg=np.mean(ps,axis=0)
        rows.append({"fold":year,"brier":brier_score_loss(yv,avg)})
        print(rows[-1],flush=True)
        parts.append(pd.DataFrame({ID:ids,"season":year,TARGET:yv.astype(np.int8),"prediction":avg}))
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp152_noid_oof.csv.gz",index=False,compression="gzip")
        pd.DataFrame(rows).to_csv(RES/"exp152_noid.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()

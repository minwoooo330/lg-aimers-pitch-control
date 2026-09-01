# -*- coding: utf-8 -*-
"""exp141_seed_nnrank — nnrank 시드 5개 추가 -> 8시드 평균.
  근거: 오늘 nnsd를 3->8 시드로 확장하니 체인 기여 2024 +1.74 / 2022 +1.39가 나왔다.
  hepta에 새로 들어간 nnrank는 아직 시드 3개다. 같은 처방을 적용한다.
  주의 선례: 손잡이NN은 3->8 확장에서 체인 기여 -0.05e-5(포화)였다. 모델마다 다르므로 실측한다.
  8시드 평균 = (3*기존avg3 + 5*신규avg5)/8 으로 정확히 계산 가능(평균의 선형성)."""
"""exp131 — 신경망 + sd + 리그상대순위. exp129의 진짜 시험대.

  exp129에서 순위 피처는 단독으로 두 fold 모두 개선했으나(2024 +3.22 / 2022 +4.98e-5)
  HGB 체인 기여는 ~0이었다(챔피언과 상관 0.9522 — 기존 HGB 멤버와 겹침).
  sd 선례: HGB에서 +13.53점, 같은 정보를 신경망에 넣자 +16.95점이 추가로 나왔다.
  따라서 '순위 정보가 무가치하다'와 'HGB 그릇에서 겹친다'를 구분하려면 신경망 경로를 봐야 한다."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parent))
from features import add_features
import importlib.util
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
spec=importlib.util.spec_from_file_location("ms",HERE/"exp125_multiseason.py")
ms=importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
spec2=importlib.util.spec_from_file_location("rk",HERE/"exp129_rank.py")
rk=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rk)
ID,TARGET="row_id","control_success"
EMB=[("pitcher_id",16),("batter_id",16),("pitcher_team_id",4),("batter_team_id",4),
     ("base_state",4),("pitcher_hand",2),("batter_hand",2),("top_bottom",2),("game_type",2)]
EPOCHS=4; BATCH=8192; SEEDS=[1,13,77,101,777]

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

def run_fold(df,year,seed,tables):
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
    cat_names={c for c,_ in EMB}
    num_cols=[c for c in df.columns if c not in cat_names and c not in (ID,TARGET)]
    sd_tr=ms.add_sd(tr,tables).fillna(0.0); sd_va=ms.add_sd(va,tables).fillna(0.0)
    rk_tr=rk.add_rank_features(tr,tables,ms.add_sd(tr,tables))
    rk_va=rk.add_rank_features(va,tables,ms.add_sd(va,tables))
    xn_tr=pd.concat([tr[num_cols].reset_index(drop=True),add_features(tr).reset_index(drop=True),
                     sd_tr.reset_index(drop=True),rk_tr.reset_index(drop=True)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols].reset_index(drop=True),add_features(va).reset_index(drop=True),
                     sd_va.reset_index(drop=True),rk_va.reset_index(drop=True)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sg=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sg).to_numpy(np.float32); xn_va=((xn_va-mu)/sg).to_numpy(np.float32)
    same_tr=(tr.pitcher_hand.to_numpy()==tr.batter_hand.to_numpy()).astype(np.float32)
    same_va=(va.pitcher_hand.to_numpy()==va.batter_hand.to_numpy()).astype(np.float32)
    net=Net(sizes,xn_tr.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr); S=torch.from_numpy(same_tr)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); VS=torch.from_numpy(same_va)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            bb=idx[st:st+BATCH]; opt.zero_grad()
            lossf(net(Xc[bb],Xn[bb],S[bb]),Y[bb]).backward(); opt.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536],VS[st:st+65536])).numpy())
    return va[ID].to_numpy(), yva, np.concatenate(ps)

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; parts=[]
    for year in [2024,2022]:
        TAB=ms.build_tables(df[df.season<year], year)
        ps={}
        for seed in SEEDS:
            ids,yv,p=run_fold(df,year,seed,TAB)
            rows.append({"fold":year,"seed":seed,"brier":brier_score_loss(yv,p)})
            print(rows[-1],flush=True); ps[seed]=p; gc.collect()
        avg=np.mean([ps[s] for s in SEEDS],axis=0)
        rows.append({"fold":year,"seed":"avg3","brier":brier_score_loss(yv,avg)})
        print(rows[-1],flush=True)
        parts.append(pd.DataFrame({ID:ids,"season":year,TARGET:yv.astype(np.int8),"prediction":avg}))
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp141_seed_nnrank_oof.csv.gz",index=False,compression="gzip")
        pd.DataFrame(rows).to_csv(RES/"exp141_seed_nnrank.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

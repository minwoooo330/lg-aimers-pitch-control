# -*- coding: utf-8 -*-
"""실험 53: 투수 임베딩을 카운트 우열 3벌로 분리 (투수x카운트 상한 0.0534 겨냥). 2024 화면."""
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
EPOCHS=4; BATCH=8192

def run_fold(df,year):
    torch.manual_seed(42); np.random.seed(42)
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
    xn_tr=pd.concat([tr[num_cols],add_features(tr)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)

    def cbin(d):
        b=d.balls_before.to_numpy(); s_=d.strikes_before.to_numpy()
        return np.where(b-s_>=2,0,np.where(s_-b>=1,2,1)).astype(np.int64)  # 0=불리 1=중립 2=유리
    same_tr=cbin(tr); same_va=cbin(va)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs=nn.ModuleList([nn.Embedding(s,d) for s,(_,d) in zip(sizes,EMB)])
            self.p_cnt=nn.ModuleList([nn.Embedding(sizes[0],16) for _ in range(3)])
            dim=xn_tr.shape[1]+sum(d for _,d in EMB)+16
            self.mlp=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                                   nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.15),
                                   nn.Linear(128,1))
        def forward(self,xc,xn,sm):
            e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
            ph=sum(self.p_cnt[k](xc[:,0])*(sm==k).float().unsqueeze(1) for k in range(3))
            return self.mlp(torch.cat(e+[ph,xn],dim=1)).squeeze(1)

    net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr); S=torch.from_numpy(same_tr)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); VS=torch.from_numpy(same_va)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for s in range(0,n,BATCH):
            b=idx[s:s+BATCH]; opt.zero_grad()
            lossf(net(Xc[b],Xn[b],S[b]),Y[b]).backward(); opt.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for s in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[s:s+65536],Vn[s:s+65536],VS[s:s+65536])).numpy())
    p=np.concatenate(ps)
    return va[ID].to_numpy(), yva, p

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig"); parts=[]; rows=[]
    for year in [2024]:
        ids,yv,p=run_fold(df,year)
        rows.append({"fold":year,"brier":brier_score_loss(yv,p),"auc":roc_auc_score(yv,p)})
        parts.append(pd.DataFrame({ID:ids,"season":year,TARGET:yv.astype(np.int8),"prediction":p}))
        print(rows[-1],flush=True); gc.collect()
    pd.concat(parts,ignore_index=True).to_csv(RES/"exp53_nn_cnt_2024_oof.csv.gz",index=False,compression="gzip")
    pd.DataFrame(rows).to_csv(RES/"exp53_nn_cnt.csv",index=False,encoding="utf-8-sig")
    print("참고: 기존 임베딩NN 2024 = 0.248685 (시드42 단일)")
    print(f"total={time.time()-t0:.1f}s")

if __name__=="__main__": main()

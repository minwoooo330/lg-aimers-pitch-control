# -*- coding: utf-8 -*-
"""실험 73: FiLM 연속 게이팅 NN (도메인 에이전트 제안).

이산 분할 축(손잡이/카운트/타자)은 스크리닝상 소진. 게이트가 상호작용을 스스로 찾게 한다.
  ph = pitcher_emb * (1 + g(상황벡터))   <- 게이트 출력을 0 근처로 초기화해 항등에서 출발
상황벡터: 같은손 여부, 카운트 우열, 볼/스트라이크, 이닝, 주자수, LI, 점수차, asof_n(log)
시드 3개, 3-fold. 판정은 10% 고정 추가 시 2024 기여.
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
EPOCHS=4; BATCH=8192; SEEDS=[42,7,2024]

def ctx_of(d):
    same=(d.pitcher_hand.to_numpy()==d.batter_hand.to_numpy()).astype(np.float32)
    adv=np.sign(d.strikes_before.to_numpy()-d.balls_before.to_numpy()).astype(np.float32)
    return np.c_[same,adv,
        d.balls_before.to_numpy()/3.0, d.strikes_before.to_numpy()/2.0,
        np.clip(d.inning.to_numpy(),1,12)/12.0,
        d.num_runners_on.to_numpy()/3.0,
        np.log1p(np.clip(d.li.to_numpy(),0,12))/2.6,
        np.clip(d.score_diff_pitcher_team.to_numpy(),-10,10)/10.0,
        np.log1p(d.asof_pitcher_n.fillna(0).to_numpy())/10.0].astype(np.float32)

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
    ct_tr=ctx_of(tr); ct_va=ctx_of(va); CD=ct_tr.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs=nn.ModuleList([nn.Embedding(s,d) for s,(_,d) in zip(sizes,EMB)])
            self.pemb=nn.Embedding(sizes[0],16)
            self.gate=nn.Sequential(nn.Linear(CD,32),nn.ReLU(),nn.Linear(32,16))
            nn.init.zeros_(self.gate[2].weight); nn.init.zeros_(self.gate[2].bias)  # 항등에서 출발
            dim=xn_tr.shape[1]+sum(d for _,d in EMB)+16
            self.mlp=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                                   nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.15),
                                   nn.Linear(128,1))
        def forward(self,xc,xnum,ctx):
            e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
            ph=self.pemb(xc[:,0])*(1.0+self.gate(ctx))
            return self.mlp(torch.cat(e+[ph,xnum],dim=1)).squeeze(1)

    net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr); Ct=torch.from_numpy(ct_tr)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); Cv=torch.from_numpy(ct_va)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            b=idx[st:st+BATCH]; opt.zero_grad()
            lossf(net(Xc[b],Xn[b],Ct[b]),Y[b]).backward(); opt.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536],Cv[st:st+65536])).numpy())
    return va[ID].to_numpy(), yva, np.concatenate(ps)

def main():
    t0=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig"); parts=[]; rows=[]
    for year in [2022,2023,2024]:
        ps=[]
        for seed in SEEDS:
            ids,yv,p=run_fold(df,year,seed)
            rows.append({"fold":year,"seed":seed,"brier":brier_score_loss(yv,p),"auc":roc_auc_score(yv,p)})
            print(rows[-1],flush=True); ps.append(p); gc.collect()
        avg=np.mean(ps,axis=0)
        rows.append({"fold":year,"seed":"avg","brier":brier_score_loss(yv,avg),"auc":roc_auc_score(yv,avg)})
        print(rows[-1],flush=True)
        parts.append(pd.DataFrame({ID:ids,"season":year,TARGET:yv.astype(np.int8),"prediction":avg}))
        pd.DataFrame(rows).to_csv(RES/"exp73_film.csv",index=False,encoding="utf-8-sig")
    pd.concat(parts,ignore_index=True).to_csv(RES/"exp73_film_oof.csv.gz",index=False,compression="gzip")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

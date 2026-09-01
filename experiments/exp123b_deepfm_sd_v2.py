# -*- coding: utf-8 -*-
"""DeepFM + sd — 완전히 다른 수학 구조.
   지금까지 GBDT(트리 분기)와 MLP(암묵적 상호작용) 두 계열만 썼다.
   FM(Factorization Machine)은 범주형 필드 간 쌍별 상호작용을 저랭크 쌍곱으로 명시적 모델링한다
   (추천시스템 CTR 예측의 표준 기법, 트리·단순MLP와 수학적으로 다른 유일한 시도).
   Deep 브랜치는 기존 성공한 구조(손잡이분할+sd)를 그대로 유지."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parent))
from features import add_features
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
EMB=[("pitcher_id",16),("batter_id",16),("pitcher_team_id",4),("batter_team_id",4),
     ("base_state",4),("pitcher_hand",2),("batter_hand",2),("top_bottom",2),("game_type",2)]
FM_DIM=16
EPOCHS=8; BATCH=8192; SEEDS=[42,7,2024]
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

class DeepFM(nn.Module):
    """진짜 DeepFM 구조(v2): FM 임베딩을 Deep 브랜치 입력에도 concat해서
       explicit 상호작용(FM)과 implicit 상호작용(MLP)이 서로의 정보를 공유하게 한다.
       FM_DIM 8->16, 에폭 4->8+코사인 스케줄(nn8에서 검증된 패턴)."""
    def __init__(self,sizes,ndim):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(s,d) for s,(_,d) in zip(sizes,EMB)])
        self.p_same=nn.Embedding(sizes[0],16); self.p_opp=nn.Embedding(sizes[0],16)
        self.fm_embs=nn.ModuleList([nn.Embedding(s,FM_DIM) for s,_ in zip(sizes,EMB)])
        self.fm_linear=nn.ModuleList([nn.Embedding(s,1) for s,_ in zip(sizes,EMB)])
        for e in self.fm_embs: nn.init.normal_(e.weight, std=0.01)
        for e in self.fm_linear: nn.init.zeros_(e.weight)
        n_fields=len(sizes)
        dim=ndim+sum(d for _,d in EMB)+16+n_fields*FM_DIM
        self.mlp=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(128,1))
    def forward(self,xc,xn,sm):
        e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
        s=sm.unsqueeze(1)
        ph=self.p_same(xc[:,0])*s+self.p_opp(xc[:,0])*(1-s)
        fv=torch.stack([emb(xc[:,j]) for j,emb in enumerate(self.fm_embs)],dim=1)  # (B,F,d)
        sum_sq=fv.sum(dim=1).pow(2).sum(dim=1)
        sq_sum=fv.pow(2).sum(dim=1).sum(dim=1)
        fm_inter=0.5*(sum_sq-sq_sum)
        fm_lin=sum(lin(xc[:,j]).squeeze(1) for j,lin in enumerate(self.fm_linear))
        fm_logit=(fm_lin+fm_inter).unsqueeze(1)
        fv_flat=fv.reshape(fv.shape[0],-1)   # Deep 브랜치가 FM 임베딩도 직접 보게 함
        deep_logit=self.mlp(torch.cat(e+[ph,xn,fv_flat],dim=1)).squeeze(1)
        return deep_logit+fm_logit.squeeze(1)

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
    xn_tr=pd.concat([tr[num_cols].reset_index(drop=True),add_features(tr).reset_index(drop=True),
                     add_sd(tr,tables).reset_index(drop=True)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols].reset_index(drop=True),add_features(va).reset_index(drop=True),
                     add_sd(va,tables).reset_index(drop=True)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)
    same_tr=(tr.pitcher_hand.to_numpy()==tr.batter_hand.to_numpy()).astype(np.float32)
    same_va=(va.pitcher_hand.to_numpy()==va.batter_hand.to_numpy()).astype(np.float32)
    net=DeepFM(sizes,xn_tr.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    n_steps_per_ep=(len(tr)+BATCH-1)//BATCH
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*n_steps_per_ep)
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr); S=torch.from_numpy(same_tr)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); VS=torch.from_numpy(same_va)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            bb=idx[st:st+BATCH]; opt.zero_grad()
            lossf(net(Xc[bb],Xn[bb],S[bb]),Y[bb]).backward(); opt.step(); sched.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536],VS[st:st+65536])).numpy())
    p=np.concatenate(ps)
    return va[ID].to_numpy(), yva, p

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; parts=[]
    for year in [2024,2022]:
        TAB=build_tables(df[df.season<year], year)
        ps={}
        for seed in SEEDS:
            ids,yv,p=run_fold(df,year,seed,TAB)
            rows.append({"fold":year,"seed":seed,"brier":brier_score_loss(yv,p)})
            print(rows[-1],flush=True); ps[seed]=p; gc.collect()
        avg=np.mean([ps[s] for s in SEEDS],axis=0)
        rows.append({"fold":year,"seed":"avg3","brier":brier_score_loss(yv,avg)})
        print(rows[-1],flush=True)
        parts.append(pd.DataFrame({ID:ids,"season":year,TARGET:yv.astype(np.int8),"prediction":avg}))
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp123b_deepfm_sd_v2_oof.csv.gz",index=False,compression="gzip")
        pd.DataFrame(rows).to_csv(RES/"exp123b_deepfm_sd_v2.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

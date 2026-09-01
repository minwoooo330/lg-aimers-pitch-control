# -*- coding: utf-8 -*-
"""exp145 — 신경망 하이퍼파라미터 랜덤 서치. 한 번도 안 해본 축.

  근거:
   (1) 우리 NN 계열이 최종 예측의 80%(nnsd 단독 31%)를 차지한다.
   (2) 그런데 은닉폭·드롭아웃·lr·weight_decay·임베딩차원·배치를 한 번도 탐색한 적이 없다.
       처음 짤 때 찍은 값 그대로이고, 건드린 건 에폭(4->8, 4->2)뿐이며 둘 다 실패했다.
   (3) 팀원의 Optuna 튜닝 LightGBM은 우리 미튜닝판보다 +103e-5였다(우리 프로토콜로 재현).
       exp32 중첩검증에서 HGB 튜닝은 홀드아웃 연도에서도 +11.94e-5로 일반화가 확인됐다.
   (4) 1등 1209는 우리 +38.6e-5이고 이는 반칙모델 여지의 30%다. 정상 모델링으로 닿는 범위다.

  절차: 2024 fold 1시드로 넓게 스크린 -> 상위 후보만 3시드 x 2fold 정밀 -> 체인 기여 판정.
  주의: 하이퍼파라미터 '골라내기'는 중첩검증 없이는 실현되지 않는다는 기록이 있다(exp29~32).
        따라서 여기서는 순위만 잡고, 최종 채택 전에 2022 fold로 별도 확인한다."""
from pathlib import Path
import sys, time, gc, json, itertools
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parent))
from features import add_features
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
ID,TARGET="row_id","control_success"
BATCH_DEF=8192
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
        with np.errstate(invalid="ignore",divide="ignore"): rate=(cur*n-rr[r]*n0)/dn
        rate=np.where(valid,np.clip(rate,0,1),np.nan)
        f["sd_"+r]=rate; f["sd_d_"+r]=np.where(valid,rate-cur,np.nan)
    bn=d.asof_batter_n.to_numpy(np.float64); bdn=bn-bn0
    bvalid=np.isfinite(bdn)&(bdn>=20)
    f["bat_logn"]=np.where(bvalid,np.log1p(np.maximum(bdn,0)),np.nan)
    for r in BRATES:
        cur=d["asof_batter_"+r].to_numpy(np.float64)
        with np.errstate(invalid="ignore",divide="ignore"): rate=(cur*bn-brr[r]*bn0)/bdn
        rate=np.where(bvalid,np.clip(rate,0,1),np.nan)
        f["bat_"+r]=rate; f["bat_d_"+r]=np.where(bvalid,rate-cur,np.nan)
    return f.fillna(0.0)

class Net(nn.Module):
    def __init__(self,sizes,ndim,cfg):
        super().__init__()
        ed=cfg["emb_dim"]; dims=[ed,ed,4,4,4,2,2,2,2]
        self.embs=nn.ModuleList([nn.Embedding(s,d) for s,d in zip(sizes,dims)])
        self.p_same=nn.Embedding(sizes[0],ed); self.p_opp=nn.Embedding(sizes[0],ed)
        dim=ndim+sum(dims)+ed
        h1,h2=cfg["h1"],cfg["h2"]; dr=cfg["dropout"]
        layers=[nn.Linear(dim,h1),nn.ReLU(),nn.Dropout(dr),nn.Linear(h1,h2),nn.ReLU(),nn.Dropout(dr)]
        if cfg.get("h3"): layers+=[nn.Linear(h2,cfg["h3"]),nn.ReLU(),nn.Dropout(dr),nn.Linear(cfg["h3"],1)]
        else: layers+=[nn.Linear(h2,1)]
        self.mlp=nn.Sequential(*layers)
    def forward(self,xc,xn,sm):
        e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
        s=sm.unsqueeze(1)
        ph=self.p_same(xc[:,0])*s+self.p_opp(xc[:,0])*(1-s)
        return self.mlp(torch.cat(e+[ph,xn],dim=1)).squeeze(1)

EMB_COLS=["pitcher_id","batter_id","pitcher_team_id","batter_team_id","base_state",
          "pitcher_hand","batter_hand","top_bottom","game_type"]

def run(df,year,seed,tables,cfg):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
    ytr=tr[TARGET].to_numpy(np.float32); yva=va[TARGET].to_numpy(np.float32)
    cat_tr=np.zeros((len(tr),len(EMB_COLS)),dtype=np.int64); cat_va=np.zeros((len(va),len(EMB_COLS)),dtype=np.int64)
    sizes=[]
    for j,c in enumerate(EMB_COLS):
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i+1 for i,v in enumerate(vals)}
        cat_tr[:,j]=tr[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        cat_va[:,j]=va[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        sizes.append(len(vals)+1)
    cn=set(EMB_COLS); num_cols=[c for c in df.columns if c not in cn and c not in (ID,TARGET)]
    xn_tr=pd.concat([tr[num_cols],add_features(tr),add_sd(tr,tables)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va),add_sd(va,tables)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)
    smt=(tr.pitcher_hand.to_numpy()==tr.batter_hand.to_numpy()).astype(np.float32)
    smv=(va.pitcher_hand.to_numpy()==va.batter_hand.to_numpy()).astype(np.float32)
    net=Net(sizes,xn_tr.shape[1],cfg)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"])
    B=cfg["batch"]; EP=cfg["epochs"]
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP*((len(tr)+B-1)//B)) if cfg["cos"] else None
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr); S=torch.from_numpy(smt)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); VS=torch.from_numpy(smv)
    n=len(tr); idx=np.arange(n)
    for ep in range(EP):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,B):
            bb=idx[st:st+B]; opt.zero_grad()
            lossf(net(Xc[bb],Xn[bb],S[bb]),Y[bb]).backward(); opt.step()
            if sched: sched.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536],VS[st:st+65536])).numpy())
    p=np.concatenate(ps)
    rel=((va.game_type=="R")|(va.season>=2023)).to_numpy()
    z=p[rel]+(yva[rel].mean()-p[rel].mean())
    return float(np.mean((z-yva[rel])**2)), va[ID].to_numpy(), yva, p

BASE={"h1":256,"h2":128,"h3":None,"dropout":0.15,"lr":2e-3,"wd":1e-5,
      "emb_dim":16,"batch":8192,"epochs":4,"cos":False}
def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    TAB=build_tables(df[df.season<2024],2024)
    rng=np.random.default_rng(7)
    cfgs=[dict(BASE,**{"tag":"base"})]
    for i in range(19):
        c=dict(BASE)
        c["h1"]=int(rng.choice([128,256,384,512]))
        c["h2"]=int(rng.choice([64,128,192,256]))
        c["h3"]=int(rng.choice([0,0,64,128])) or None
        c["dropout"]=float(rng.choice([0.05,0.10,0.15,0.25,0.35]))
        c["lr"]=float(rng.choice([5e-4,1e-3,2e-3,3e-3]))
        c["wd"]=float(rng.choice([0.0,1e-5,1e-4,1e-3]))
        c["emb_dim"]=int(rng.choice([8,16,24,32]))
        c["batch"]=int(rng.choice([4096,8192,16384]))
        c["epochs"]=int(rng.choice([3,4,5,6]))
        c["cos"]=bool(rng.choice([False,True]))
        c["tag"]=f"t{i:02d}"
        cfgs.append(c)
    rows=[]
    for k,c in enumerate(cfgs):
        tt=time.time()
        try: br,_,_,_=run(df,2024,42,TAB,c)
        except Exception as e:
            print(f"[{c['tag']}] 실패 {e}",flush=True); continue
        rows.append(dict(c,brier=br,sec=round(time.time()-tt)))
        print(f"[{k+1:2d}/{len(cfgs)}] {c['tag']:5s} brier {br:.8f}  "
              f"h{c['h1']}-{c['h2']}{'-'+str(c['h3']) if c['h3'] else ''} dr{c['dropout']} "
              f"lr{c['lr']:.0e} wd{c['wd']:.0e} emb{c['emb_dim']} B{c['batch']} ep{c['epochs']} "
              f"{'cos' if c['cos'] else '---'}  ({time.time()-tt:.0f}s)",flush=True)
        pd.DataFrame(rows).to_csv(RES/"exp145_nn_search.csv",index=False,encoding="utf-8-sig")
    r=pd.DataFrame(rows).sort_values("brier")
    print("\n=== 상위 8 (2024 fold, 1시드) ===")
    print(r.head(8)[["tag","brier","h1","h2","h3","dropout","lr","wd","emb_dim","batch","epochs","cos"]].to_string(index=False))
    b=r[r.tag=="base"].brier.iloc[0]
    print(f"\n기준(현행 설정) {b:.8f}")
    print(f"최고 {r.brier.iloc[0]:.8f}  개선 {(b-r.brier.iloc[0])*1e5:+.2f}e-5")
    print(f"total={time.time()-t0:.0f}s")
if __name__=="__main__": main()

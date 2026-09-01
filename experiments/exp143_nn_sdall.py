# -*- coding: utf-8 -*-
"""exp143 — NN + sd 전채널(투수 성적 / 투수 볼스트 / 타자 / 구종믹스). 조합표의 빈 칸.

  역대 최대 이득 2건: sd x HGB(+13.53점), sd x NN(+16.95점).
  채널 확장판 sdall은 HGB로만 존재하고 실효가중 3.41%에 그친다.
  최강 그릇(NN, nnsd 실효가중 31%)에 전채널을 넣은 조합이 미시도다.

  기본 sd(13피처) 대비 추가: pitbs(볼/스트라이크) + mix(구종믹스) 채널.
  구현은 검증본 exp110_channels.py의 CHANNELS/end_state/add_ch를 그대로 사용한다.
  [주의] train_sdall.py의 함수는 배포용이라 조회표를 하나만 받는다. fold 검증에
         그걸 쓰면 dn이 음수가 되는 과거 버그(exp103)가 재발하므로 쓰지 않는다."""
"""신경망 + sd(시즌+타자 채널) — 지금까지 sd를 안 넣어본 유일한 그릇.
   NN은 트리와 근본적으로 다른 오차 패턴을 만들어 상관이 가장 낮았다(0.87~0.95).
   거기에 새 정보(sd)까지 더하면 다양성이 곱해질 가능성. 손잡이분할 구조에 sd 피처만 추가."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parent))
from features import add_features
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
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

def build_all_tabs(base_df, year):
    """채널별 시즌별 조회표. 반드시 시즌별이어야 한다 —
       고정 단일 표를 쓰면 dn이 음수가 되어 효과가 0 또는 역효과가 된다(과거 exp103 버그)."""
    lo=int(base_df.season.min())
    return {tag:{S:end_state(base_df,S-1,*CHANNELS[tag],PREF[tag])
                 for S in range(lo+1,year+1)} for tag in CHANNELS}

def add_sdall_nn(d, tabs):
    f=pd.concat([add_ch(d,tabs[t],t) for t in ["pit","pitbs","bat","mix"]],axis=1)
    return f.fillna(0.0)   # NN 입력: 결측 0 대체(nnsd 관례)

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
    xn_tr=pd.concat([tr[num_cols],add_features(tr),add_sdall_nn(tr,tables)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va),add_sdall_nn(va,tables)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)
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
    p=np.concatenate(ps)
    return va[ID].to_numpy(), yva, p

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; parts=[]
    for year in [2024,2022]:
        TAB=build_all_tabs(df[df.season<year], year)
        ps={}
        for seed in SEEDS:
            ids,yv,p=run_fold(df,year,seed,TAB)
            rows.append({"fold":year,"seed":seed,"brier":brier_score_loss(yv,p)})
            print(rows[-1],flush=True); ps[seed]=p; gc.collect()
        avg=np.mean([ps[s] for s in SEEDS],axis=0)
        rows.append({"fold":year,"seed":"avg3","brier":brier_score_loss(yv,avg)})
        print(rows[-1],flush=True)
        parts.append(pd.DataFrame({ID:ids,"season":year,TARGET:yv.astype(np.int8),"prediction":avg}))
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp143_nn_sdall_oof.csv.gz",index=False,compression="gzip")
        pd.DataFrame(rows).to_csv(RES/"exp143_nn_sdall.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

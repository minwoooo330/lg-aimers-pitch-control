# -*- coding: utf-8 -*-
"""train_nnloadgap — 시즌 누적 부하 + 당해 기준 투수-타자 격차 NN (전체 데이터).
   exp135 검증: 체인 기여 20%에서 2024 +1.55 / 2022 +0.46 (단일 추가 후보 중 2024 최대).
   sd 없이는 계산 자체가 불가능했던 축이다 — 통산 격차가 아니라 *당해 시즌* 격차,
   그리고 sd의 dn(올해 투구수)과 game_month로 만든 시즌 누적 부하.
   전부 행 자신의 공식 컬럼 + 학습데이터 조회표만 사용하므로 규정 준수."""
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
DATA=HERE/"data"/"train.csv"; OUT=HERE/"submits_common"; PKL=OUT/"nnloadgap_model.pkl"
ID,TARGET="row_id","control_success"
EMB=[("pitcher_id",16),("batter_id",16),("pitcher_team_id",4),("batter_team_id",4),
     ("base_state",4),("pitcher_hand",2),("batter_hand",2),("top_bottom",2),("game_type",2)]
EPOCHS=4; BATCH=8192; SEEDS=[42,7,2024]
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

def add_load_gap(d, tables, sd):
    """시즌 누적 부하 + 당해 기준 투수-타자 격차. 둘 다 sd 없이는 계산 불가능했던 축.
       전부 (행 자신의 공식 컬럼) 또는 (학습데이터 조회표)만 사용 -> 규정 준수."""
    f=pd.DataFrame(index=d.index)
    seas=d.season.to_numpy(); pid=d.pitcher_id.to_numpy()
    dn=np.expm1(sd["sd_logn"].to_numpy(np.float64))          # 올해 던진 공 수
    bdn=np.expm1(sd["bat_logn"].to_numpy(np.float64))
    month=d.game_month.to_numpy(np.float64)
    elapsed=np.clip(month-2.0,1.0,None)                       # 시즌 개막(3월) 이후 경과 개월

    # --- A. 시즌 누적 부하 ---
    with np.errstate(invalid="ignore",divide="ignore"):
        rate=dn/elapsed
    f["load_per_month"]=np.log1p(np.maximum(rate,0))
    f["load_frac_season"]=np.clip(elapsed/7.0,0,1)*np.log1p(np.maximum(dn,0))   # 시즌 진행도 x 누적량
    # 통산 대비 올해 부하: 활동 시즌 수로 나눈 평균과 비교
    n_prev=np.full(len(d),np.nan); act=np.full(len(d),np.nan)
    for S in np.unique(seas):
        tbl=tables.get(int(S)); mm=(seas==S)
        if tbl is None or not mm.any(): continue
        sub=pd.Series(pid[mm]); n_prev[mm]=sub.map(tbl["n"]).to_numpy(np.float64)
        cnt=np.zeros(mm.sum())
        for s2 in range(int(min(tables.keys())), int(S)):
            t2=tables.get(s2); t3=tables.get(s2+1)
            if t2 is None or t3 is None: continue
            a=sub.map(t2["n"]).to_numpy(np.float64); bq=sub.map(t3["n"]).to_numpy(np.float64)
            cnt+=np.where(np.isfinite(bq-a)&((bq-a)>0),1.0,0.0)
        act[mm]=cnt
    act=np.where(act>0,act,np.nan)
    with np.errstate(invalid="ignore",divide="ignore"):
        typical=n_prev/act                                    # 평년 한 시즌 투구량
        ratio=dn/np.where(elapsed>0,typical*(elapsed/7.0),np.nan)
    f["load_vs_typical"]=np.clip(ratio,0,5)                    # 1보다 크면 평년보다 과부하
    f["active_seasons"]=act
    f["late_x_load"]=(month>=8).astype(np.float64)*f["load_per_month"]

    # --- B. 당해 기준 투수-타자 지배력 격차 ---
    p_s=sd["sd_success_rate"].to_numpy(np.float64); b_s=sd["bat_success_rate"].to_numpy(np.float64)
    pc =d.asof_pitcher_success_rate.to_numpy(np.float64); bc=d.asof_batter_success_rate.to_numpy(np.float64)
    f["gap_sd"]=p_s-b_s
    f["gap_career"]=pc-bc
    f["gap_shift"]=f["gap_sd"]-f["gap_career"]                 # 올해 들어 우열이 이동했나
    f["gap_middle"]=sd["sd_middle_rate"].to_numpy(np.float64)-sd["bat_middle_rate"].to_numpy(np.float64)
    f["exp_gap"]=sd["sd_logn"].to_numpy(np.float64)-sd["bat_logn"].to_numpy(np.float64)

    # --- C. 부하 x 폼 (피로 징후) ---
    f["load_x_form"]=f["load_per_month"]*sd["sd_d_success_rate"].to_numpy(np.float64)
    return f

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
    lgfeat=add_load_gap(df, TAB, sdfeat)
    xn=pd.concat([df[num_cols],add_features(df),sdfeat,lgfeat],axis=1).astype(np.float32)
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

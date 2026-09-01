# -*- coding: utf-8 -*-
"""exp162 — 멀티태스크 NN. 본 타깃(제구 성공)과 함께 그 투구의 실제 Trackman 물리량
   5종을 보조 타깃으로 동시 학습한다(공유 trunk + 2 head, 결측 마스킹).
   보조타깃은 입력이 아니라 타깃이므로 추론 시 필요 없다 = 규정 위반 없음.
   AUX_W로 보조손실 비중 조절."""
AUX_W=0.1
AUXTAB=None
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
    return f.fillna(0.0)   # NN은 NaN 처리 불가 -> 결측 0 대체 + isnew 플래그로 구분 가능

class Net(nn.Module):
    def __init__(self,sizes,ndim):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(s,d) for s,(_,d) in zip(sizes,EMB)])
        self.p_same=nn.Embedding(sizes[0],16); self.p_opp=nn.Embedding(sizes[0],16)
        dim=ndim+sum(d for _,d in EMB)+16
        self.mlp=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(128,1))
        self.trunk=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                                 nn.Linear(256,128),nn.ReLU())
        self.head_y=nn.Linear(128,1); self.head_a=nn.Linear(128,5)
    def forward(self,xc,xn,sm):
        e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
        s=sm.unsqueeze(1)
        ph=self.p_same(xc[:,0])*s+self.p_opp(xc[:,0])*(1-s)
        h=self.trunk(torch.cat(e+[ph,xn],dim=1))
        return self.head_y(h).squeeze(1), self.head_a(h)

AUX=["rel_speed","spin_rate","induced_vert_break","horz_break","extension"]

def attach_aux(df):
    """정확 매칭 경기를 행 단위로 정렬해 그 투구의 실제 물리량을 row_id에 붙인다.
       exp43에서 검증된 정렬 절차를 그대로 쓴다. 보조 타깃 전용."""
    import gc
    from trackman_features import match_exact_games
    TMC=["trackman_id","season","trackman_game_id","pitch_no","inning","top_bottom",
         "balls_before","strikes_before","outs_before","pitcher_trackman_id",
         "pitcher_hand","batter_hand"]+AUX
    tm=pd.read_csv(HERE/"data"/"trackman_history.csv",usecols=TMC)
    mg,ts,matches=match_exact_games(df,tm)
    del tm; gc.collect()
    mi=mg.groupby("_game_idx",sort=False).indices
    ti=ts.groupby("trackman_game_id",sort=False).indices
    parts=[]
    for row in matches.itertuples(index=False):
        a_,b_=mi[row.main_game_idx],ti[row.trackman_game_id]
        if len(a_)!=len(b_): continue
        L=mg.iloc[a_][[ID]].reset_index(drop=True)
        R=ts.iloc[b_][AUX].reset_index(drop=True)
        parts.append(pd.concat([L,R],axis=1))
    al=pd.concat(parts,ignore_index=True).set_index(ID)
    print(f"  보조타깃 정렬 {len(al):,}행", flush=True)
    return al


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
    xn_tr=pd.concat([tr[num_cols],add_features(tr),add_sd(tr,tables)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va),add_sd(va,tables)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)
    same_tr=(tr.pitcher_hand.to_numpy()==tr.batter_hand.to_numpy()).astype(np.float32)
    same_va=(va.pitcher_hand.to_numpy()==va.batter_hand.to_numpy()).astype(np.float32)
    net=Net(sizes,xn_tr.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss()
    # 보조타깃: 학습 구간만, 투수별 표준화 후 결측 마스킹
    aux=AUXTAB.reindex(tr[ID]).to_numpy(np.float32)
    am=~np.isnan(aux)
    mu_=np.nanmean(aux,axis=0); sd_=np.nanstd(aux,axis=0); sd_[sd_<1e-6]=1.0
    aux=np.nan_to_num((aux-mu_)/sd_,nan=0.0).astype(np.float32)
    A=torch.from_numpy(aux); AM=torch.from_numpy(am.astype(np.float32))
    print(f"    보조타깃 적용 {am.any(axis=1).mean():.1%} 행", flush=True)
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr); S=torch.from_numpy(same_tr)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); VS=torch.from_numpy(same_va)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            bb=idx[st:st+BATCH]; opt.zero_grad()
            zy,za=net(Xc[bb],Xn[bb],S[bb])
            l=lossf(zy,Y[bb])
            m_=AM[bb]
            if m_.sum()>0:
                l=l+AUX_W*(((za-A[bb])**2)*m_).sum()/m_.sum()
            l.backward(); opt.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536],VS[st:st+65536])[0]).numpy())
    p=np.concatenate(ps)
    return va[ID].to_numpy(), yva, p

def main():
    global AUXTAB
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    AUXTAB=attach_aux(df)
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
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp162_nn_mtl_oof.csv.gz",index=False,compression="gzip")
        pd.DataFrame(rows).to_csv(RES/"exp120_nn_sd.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""DeepFM v3 — FM 필드를 상황변수까지 확장.
   v1/v2는 FM이 투수/타자/팀/손잡이 등 9개 필드만 봤다. 상황변수(카운트·아웃·이닝·점수차·LI·주자)는
   MLP 브랜치에만 원시값으로 들어가고 FM의 쌍별 상호작용에서는 빠져 있었다.
   exp94에서 "투수x카운트","투수x이닝" 등 개별 축은 전부 문턱 미달로 기각됐지만, 그건 각 축을
   따로 떼어 잔차상관을 잰 것이다. FM은 모든 필드 쌍의 상호작용을 동시에 공유 파라미터로 학습하므로
   개별로는 약한 여러 축이 결합돼 신호가 될 수 있다(도메인43이 개별 스크리닝 실패 후 그룹으로 통했던 것과 같은 논리)."""
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
# FM 전용 상황 필드(정수 버킷, 소카디널리티) — deep 브랜치의 원시 수치와 별개로 FM 쌍별 상호작용에 참여
CTX_FIELDS=["balls_before","strikes_before","outs_before","num_runners_on",
            "inning_b","scorediff_b","li_b","scoring_pos"]
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

def add_ctx_fields(d):
    o=pd.DataFrame(index=d.index)
    o["balls_before"]=d.balls_before.clip(0,3).astype(np.int64)
    o["strikes_before"]=d.strikes_before.clip(0,2).astype(np.int64)
    o["outs_before"]=d.outs_before.clip(0,2).astype(np.int64)
    o["num_runners_on"]=d.num_runners_on.clip(0,3).astype(np.int64)
    o["inning_b"]=pd.cut(d.inning,[0,2,4,6,8,99],labels=False).fillna(4).astype(np.int64)
    o["scorediff_b"]=pd.cut(d.score_diff_pitcher_team,[-99,-4,-1,1,4,99],labels=False).fillna(2).astype(np.int64)
    o["li_b"]=pd.qcut(d.li.rank(method="first"),4,labels=False).astype(np.int64)
    o["scoring_pos"]=((d.runner_on_2b==1)|(d.runner_on_3b==1)).astype(np.int64)
    return o

class DeepFM(nn.Module):
    def __init__(self,sizes,ctx_sizes,ndim):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(s,d) for s,(_,d) in zip(sizes,EMB)])
        self.p_same=nn.Embedding(sizes[0],16); self.p_opp=nn.Embedding(sizes[0],16)
        all_sizes=sizes+ctx_sizes
        self.fm_embs=nn.ModuleList([nn.Embedding(s,FM_DIM) for s in all_sizes])
        self.fm_linear=nn.ModuleList([nn.Embedding(s,1) for s in all_sizes])
        for e in self.fm_embs: nn.init.normal_(e.weight, std=0.01)
        for e in self.fm_linear: nn.init.zeros_(e.weight)
        n_fields=len(all_sizes)
        dim=ndim+sum(d for _,d in EMB)+16+n_fields*FM_DIM
        self.mlp=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.15),
                               nn.Linear(128,1))
    def forward(self,xc,xctx,xn,sm):
        allc=torch.cat([xc,xctx],dim=1)
        e=[emb(xc[:,j]) for j,emb in enumerate(self.embs)]
        s=sm.unsqueeze(1)
        ph=self.p_same(xc[:,0])*s+self.p_opp(xc[:,0])*(1-s)
        fv=torch.stack([emb(allc[:,j]) for j,emb in enumerate(self.fm_embs)],dim=1)  # (B,F,d)
        sum_sq=fv.sum(dim=1).pow(2).sum(dim=1)
        sq_sum=fv.pow(2).sum(dim=1).sum(dim=1)
        fm_inter=0.5*(sum_sq-sq_sum)
        fm_lin=sum(lin(allc[:,j]).squeeze(1) for j,lin in enumerate(self.fm_linear))
        fm_logit=fm_lin+fm_inter
        fv_flat=fv.reshape(fv.shape[0],-1)
        deep_logit=self.mlp(torch.cat(e+[ph,xn,fv_flat],dim=1)).squeeze(1)
        return deep_logit+fm_logit

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
    ctx_tr_df=add_ctx_fields(tr); ctx_va_df=add_ctx_fields(va)
    ctx_sizes=[int(max(ctx_tr_df[c].max(),ctx_va_df[c].max()))+1 for c in CTX_FIELDS]
    ctx_tr=ctx_tr_df[CTX_FIELDS].to_numpy(dtype=np.int64); ctx_va=ctx_va_df[CTX_FIELDS].to_numpy(dtype=np.int64)
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
    net=DeepFM(sizes,ctx_sizes,xn_tr.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    n_steps_per_ep=(len(tr)+BATCH-1)//BATCH
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*n_steps_per_ep)
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xctx=torch.from_numpy(ctx_tr); Xn=torch.from_numpy(xn_tr)
    Y=torch.from_numpy(ytr); S=torch.from_numpy(same_tr)
    Vc=torch.from_numpy(cat_va); Vctx=torch.from_numpy(ctx_va); Vn=torch.from_numpy(xn_va); VS=torch.from_numpy(same_va)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            bb=idx[st:st+BATCH]; opt.zero_grad()
            lossf(net(Xc[bb],Xctx[bb],Xn[bb],S[bb]),Y[bb]).backward(); opt.step(); sched.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vctx[st:st+65536],Vn[st:st+65536],VS[st:st+65536])).numpy())
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
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp124_deepfm_ctx_oof.csv.gz",index=False,compression="gzip")
        pd.DataFrame(rows).to_csv(RES/"exp124_deepfm_ctx.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

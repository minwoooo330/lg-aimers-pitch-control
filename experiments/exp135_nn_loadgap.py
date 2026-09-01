# -*- coding: utf-8 -*-
"""exp134 — 통합 신경망: sd + Trackman + 리그순위를 한 모델에.

  재감사에서 exp122(NN+sd+TM)와 exp131(NN+sd+순위)이 각각 게이트를 통과했고,
  둘을 각각 15%씩 넣은 조합이 2024 +1.59 / 2022 +0.35로 단독보다 좋았다.
  두 후보의 상호 상관은 0.9619다.
  미검증 질문: 피처를 한 모델에 다 넣는 것과, 따로 학습해 블렌딩하는 것 중 무엇이 나은가?
  일반적으로 블렌딩이 다양성 때문에 유리하지만 실측해야 안다.
  판정은 체인 한계기여로 하며, 기존 블렌딩(TM15%+순위15%)과 직접 비교한다."""
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
    sd_tr=add_sd(tr,tables); sd_va=add_sd(va,tables)
    lg_tr=add_load_gap(tr,tables,sd_tr); lg_va=add_load_gap(va,tables,sd_va)
    xn_tr=pd.concat([tr[num_cols],add_features(tr),sd_tr,lg_tr],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va),sd_va,lg_va],axis=1).astype(np.float32)
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
        pd.concat(parts,ignore_index=True).to_csv(RES/"exp135_nn_loadgap_oof.csv.gz",index=False,compression="gzip")
        pd.DataFrame(rows).to_csv(RES/"exp135_nn_loadgap.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s",flush=True)
if __name__=="__main__": main()

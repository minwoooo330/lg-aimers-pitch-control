# -*- coding: utf-8 -*-
"""exp149 — 팀원 v7의 역할/레버리지 + Marcel 피처를 우리 NN(nnsd)에 이식 + halflife 가중.

  근거(재개장 룰): '역할' 축은 Trackman 구현으로 기각했었으나 v7은 메인데이터 상황컬럼으로
  성공했다. '시즌 가중'은 sd 이전 GBDT에서 기각했으나 v7은 halflife=2로 성공했다.
  두 축 모두 우리 NN 그릇에서는 미시도.
  변형: A = +역할/Marcel 피처만, B = A + halflife 가중(시즌 감쇠 2).
  판정: oct 혼합(v7 55% + nnsd16 45%)에서 nnsd16 쪽을 교체했을 때의 개선."""
from pathlib import Path
import sys, time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import brier_score_loss
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
sys.path.insert(0,str(HERE))
from features import add_features
import v7_pipeline as V7
V7.SF_LEAGUE_KEYS=[]; V7.MRC_LEAGUE_KEYS=["season"]
import importlib.util as _u
_s=_u.spec_from_file_location("e120",HERE/"exp120_nn_sd.py")
E=_u.module_from_spec(_s); _s.loader.exec_module(E)
ID,TARGET="row_id","control_success"
SEEDS=[42,7,2024]; EPOCHS=4; BATCH=8192
ROLE_COLS=None  # 아래에서 결정

def v7_extra(d, full, year):
    """v7의 carry(역할 포함) + marcel 피처에서 '우리 NN에 없는 것'만 뽑는다."""
    hist=full[full.season<year]
    st=V7.build_carry_state(hist)
    sf=V7.apply_carry_state(d, st)
    label_cols=["season","game_type","pitcher_id",TARGET]
    proj=V7.build_projection(hist[label_cols],["pitcher_id"],int(year),reg=V7.MRC_REG)
    mrc=V7.apply_projection(d, proj, "mrc_pit")
    keep=[c for c in sf.columns if ("role" in c) or ("li_vs" in c)]
    out=pd.concat([sf[keep], mrc],axis=1)
    return out

def run(df,year,seed,tables,use_hl):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=df[df.season<year].reset_index(drop=True); va=df[df.season==year].reset_index(drop=True)
    ytr=tr[TARGET].to_numpy(np.float32); yva=va[TARGET].to_numpy(np.float32)
    EMB=E.EMB
    cat_tr=np.zeros((len(tr),len(EMB)),dtype=np.int64); cat_va=np.zeros((len(va),len(EMB)),dtype=np.int64)
    sizes=[]
    for j,(c,_) in enumerate(EMB):
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i+1 for i,v in enumerate(vals)}
        cat_tr[:,j]=tr[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        cat_va[:,j]=va[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        sizes.append(len(vals)+1)
    cn={c for c,_ in EMB}
    num_cols=[c for c in df.columns if c not in cn and c not in (ID,TARGET)]
    ex_tr=[]; 
    for s in np.sort(tr.season.unique()):
        sub=tr[tr.season==s]
        ex_tr.append(v7_extra(sub, df, int(s)))
    ex_tr=pd.concat(ex_tr).reindex(tr.index)
    ex_va=v7_extra(va, df, year)
    xn_tr=pd.concat([tr[num_cols],add_features(tr),E.add_sd(tr,tables),ex_tr],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va),E.add_sd(va,tables),ex_va],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)
    smt=(tr.pitcher_hand.to_numpy()==tr.batter_hand.to_numpy()).astype(np.float32)
    smv=(va.pitcher_hand.to_numpy()==va.batter_hand.to_numpy()).astype(np.float32)
    wt=np.ones(len(tr),dtype=np.float32)
    if use_hl:
        age=(year-1)-tr.season.to_numpy()
        wt=np.power(2.0,-age/2.0).astype(np.float32); wt=wt/wt.mean()
    net=E.Net(sizes,xn_tr.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss(reduction="none")
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr)
    S=torch.from_numpy(smt); W=torch.from_numpy(wt)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); VS=torch.from_numpy(smv)
    n=len(tr); idx=np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            bb=idx[st:st+BATCH]; opt.zero_grad()
            (lossf(net(Xc[bb],Xn[bb],S[bb]),Y[bb])*W[bb]).mean().backward(); opt.step()
    net.eval(); ps=[]
    with torch.no_grad():
        for st in range(0,len(va),65536):
            ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536],VS[st:st+65536])).numpy())
    return va[ID].to_numpy(), yva, np.concatenate(ps)

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    rows=[]; store={"A":[],"B":[]}
    for year in [2024,2022]:
        TAB=E.build_tables(df[df.season<year], year)
        for tag,hl in [("A",False),("B",True)]:
            ps=[]
            for seed in SEEDS:
                ids,yv,p=run(df,year,seed,TAB,hl)
                ps.append(p); gc.collect()
            avg=np.mean(ps,axis=0)
            rows.append({"fold":year,"var":tag,"brier":brier_score_loss(yv,avg)})
            print(rows[-1],flush=True)
            store[tag].append(pd.DataFrame({ID:ids,"season":year,TARGET:yv.astype(np.int8),"prediction":avg}))
            pd.concat(store[tag],ignore_index=True).to_csv(RES/f"exp149_{tag}_oof.csv.gz",index=False,compression="gzip")
            pd.DataFrame(rows).to_csv(RES/"exp149_role_marcel.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()

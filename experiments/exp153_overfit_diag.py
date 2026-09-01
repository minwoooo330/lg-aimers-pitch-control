# -*- coding: utf-8 -*-
"""exp153 — 과적합 직접 진단. 에폭별 학습셋/검증셋 Brier 격차를 추적한다.
   과적합이면: 학습 Brier는 계속 내려가고 검증 Brier는 어느 시점부터 올라간다."""
from pathlib import Path
import sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
HERE=Path(__file__).resolve().parent; RES=HERE/"results"
sys.path.insert(0,str(HERE))
from features import add_features
import importlib.util as _u
_s=_u.spec_from_file_location("e120",HERE/"exp120_nn_sd.py")
E=_u.module_from_spec(_s); _s.loader.exec_module(E)
ID,TARGET="row_id","control_success"
EMB=E.EMB; BATCH=8192; MAXEP=8

def main():
    t0=time.time(); df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    year=2024
    TAB=E.build_tables(df[df.season<year], year)
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
    xn_tr=pd.concat([tr[num_cols],add_features(tr),E.add_sd(tr,TAB)],axis=1).astype(np.float32)
    xn_va=pd.concat([va[num_cols],add_features(va),E.add_sd(va,TAB)],axis=1).astype(np.float32)
    med=xn_tr.median(); xn_tr=xn_tr.fillna(med); xn_va=xn_va.fillna(med)
    mu,sd=xn_tr.mean(),xn_tr.std().replace(0,1)
    xn_tr=((xn_tr-mu)/sd).to_numpy(np.float32); xn_va=((xn_va-mu)/sd).to_numpy(np.float32)
    smt=(tr.pitcher_hand.to_numpy()==tr.batter_hand.to_numpy()).astype(np.float32)
    smv=(va.pitcher_hand.to_numpy()==va.batter_hand.to_numpy()).astype(np.float32)
    relv=((va.game_type=="R")|(va.season>=2023)).to_numpy()
    torch.manual_seed(42); np.random.seed(42)
    net=E.Net(sizes,xn_tr.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss()
    Xc=torch.from_numpy(cat_tr); Xn=torch.from_numpy(xn_tr); Y=torch.from_numpy(ytr); S=torch.from_numpy(smt)
    Vc=torch.from_numpy(cat_va); Vn=torch.from_numpy(xn_va); VS=torch.from_numpy(smv)
    n=len(tr); idx=np.arange(n)
    sub=np.random.default_rng(0).choice(n,size=200000,replace=False)   # 학습셋 평가용 표본
    rows=[]
    print(f"{'ep':>3s} {'학습Brier':>12s} {'검증Brier':>12s} {'격차(e-5)':>11s}",flush=True)
    for ep in range(1,MAXEP+1):
        np.random.shuffle(idx); net.train()
        for st in range(0,n,BATCH):
            bb=idx[st:st+BATCH]; opt.zero_grad()
            lossf(net(Xc[bb],Xn[bb],S[bb]),Y[bb]).backward(); opt.step()
        net.eval(); ps=[]
        with torch.no_grad():
            for st in range(0,len(va),65536):
                ps.append(torch.sigmoid(net(Vc[st:st+65536],Vn[st:st+65536],VS[st:st+65536])).numpy())
            pv=np.concatenate(ps)
            pt=[]
            for st in range(0,len(sub),65536):
                k=sub[st:st+65536]
                pt.append(torch.sigmoid(net(Xc[k],Xn[k],S[k])).numpy())
            pt=np.concatenate(pt)
        bt=float(np.mean((pt-ytr[sub])**2))
        z=pv[relv]+(yva[relv].mean()-pv[relv].mean())
        bv=float(np.mean((z-yva[relv])**2))
        rows.append({"epoch":ep,"train":bt,"val":bv,"gap_e5":(bv-bt)*1e5})
        print(f"{ep:3d} {bt:12.8f} {bv:12.8f} {(bv-bt)*1e5:11.2f}",flush=True)
        pd.DataFrame(rows).to_csv(RES/"exp153_overfit.csv",index=False,encoding="utf-8-sig")
    print(f"\ntotal={time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""실험 24: asof 누적비율 역산으로 복원한 투구단위 실패유형 다중분류 2024 gate."""
from pathlib import Path
import time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from features import add_features
HERE=Path(__file__).resolve().parent; DATA=HERE/"data"/"train.csv"; RES=HERE/"results"
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]

def reconstruct_labels(df):
    """파일 순서=투수 시간순임을 이용해 각 투구의 실패유형을 복원한다(학습 라벨 전용)."""
    order=np.arange(len(df)); g=df.assign(_o=order).sort_values(["pitcher_id","_o"])
    n=g["asof_pitcher_n"].to_numpy(float); pid=g["pitcher_id"].to_numpy()
    nxt_same=np.r_[pid[1:]==pid[:-1],False]
    out={}
    for key,col in [("succ","asof_pitcher_success_rate"),("rev","asof_pitcher_reverse_rate"),
                    ("mid","asof_pitcher_middle_rate")]:
        cnt=np.round(g[col].to_numpy(float)*n); d=np.r_[cnt[1:]-cnt[:-1],np.nan]
        d[~nxt_same]=np.nan; out[key]=d
    lab=pd.DataFrame(out,index=g.index)
    y=np.full(len(df),-1,dtype=np.int8)
    valid=lab.notna().all(axis=1).to_numpy()
    s=lab.succ.to_numpy(); r=lab.rev.to_numpy(); m=lab.mid.to_numpy()
    cls=np.where(s==1,0,np.where(m==1,2,np.where(r==1,1,3))).astype(np.int8)
    cls[~valid]=-1
    y[g.index.to_numpy()]=cls
    # 복원 실패 행은 이진 정답으로 최소 보정: 성공=0, 실패=3(미분류)
    fallback=df[TARGET].to_numpy()
    y=np.where(y>=0,y,np.where(fallback==1,0,3)).astype(np.int8)
    return y

def encode(tr,va,base):
    a=tr[base].copy(); b=va[base].copy()
    for c in CATS:
        vals=sorted(tr[c].dropna().astype(str).unique()); mp={v:i for i,v in enumerate(vals)}
        a[c]=tr[c].astype(str).map(mp).fillna(-1).astype(np.int16)
        b[c]=va[c].astype(str).map(mp).fillna(-1).astype(np.int16)
    return a,b

def main():
    start=time.time(); df=pd.read_csv(DATA,encoding="utf-8-sig")
    ymc=reconstruct_labels(df)
    print("복원 라벨 분포:",pd.Series(ymc).value_counts().sort_index().to_dict())
    check=((ymc==0).astype(int)==df[TARGET].to_numpy()).mean()
    print(f"class0 == control_success 일치율: {check:.6f}")
    base=[c for c in df.columns if c not in (ID,TARGET)]
    tr=df[df.season<2024]; va=df[df.season==2024]
    xtr,xva=encode(tr,va,base)
    xtr=pd.concat([xtr,add_features(tr)],axis=1); xva=pd.concat([xva,add_features(va)],axis=1)
    ybin_tr=tr[TARGET].to_numpy(np.int8); yva=va[TARGET].to_numpy(np.int8)
    ymc_tr=ymc[df.season.to_numpy()<2024]
    mask=[c in CATS for c in xtr.columns]
    rows=[]
    for name,ytrain,multi in [("binary_baseline",ybin_tr,False),("multiclass_outcome",ymc_tr,True)]:
        t=time.time()
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=31,
            min_samples_leaf=200,l2_regularization=1.,early_stopping=False,
            categorical_features=mask,random_state=42)
        m.fit(xtr,ytrain); proba=m.predict_proba(xva)
        p=proba[:,list(m.classes_).index(0)] if multi else proba[:,1]
        row={"model":name,"fold":2024,"brier":brier_score_loss(yva,p),
             "logloss":log_loss(yva,p,labels=[0,1]),"roc_auc":roc_auc_score(yva,p),
             "pred_mean":float(p.mean()),"seconds":time.time()-t}
        rows.append(row); print(row)
    res=pd.DataFrame(rows); gain=float(res.iloc[0].brier-res.iloc[1].brier)
    res["gain_vs_binary"]=[0.0,gain]; res["pass"]=gain>=1e-5
    RES.mkdir(exist_ok=True); res.to_csv(RES/"exp24_multiclass_outcome_2024.csv",index=False,encoding="utf-8-sig")
    np.save(RES/"exp24_reconstructed_labels.npy",ymc)
    print("PASS" if gain>=1e-5 else "STOP", f"gain={gain:.8f}, total={time.time()-start:.1f}s")

if __name__=="__main__": main()

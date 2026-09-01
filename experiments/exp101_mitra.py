# -*- coding: utf-8 -*-
"""실험 101: Mitra (AutoGluon 타뷸러 파운데이션 모델, apache-2.0) 스크리닝.
   ICL 방식이라 학습 행 전부를 context로 올릴 수 없다 → 서브샘플 컨텍스트로 2024 fold 일부를 예측.
   게이트: 단독 Brier <= 0.2490  AND  챔피언 체인과 상관 <= 0.90 (FTT 0.95보다 확실히 낮아야 함)."""
import sys, time, gc, traceback
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp")
sys.path.insert(0,str(HERE))
from features import add_features
from hfeatures import add_hfeatures
ID,TARGET="row_id","control_success"
CATS=["top_bottom","game_type","base_state","pitcher_hand","batter_hand"]
RES=HERE/"results"

def main():
    t0=time.time()
    df=pd.read_csv(HERE/"data"/"train.csv",encoding="utf-8-sig")
    tr=df[df.season<2024]; va=df[df.season==2024]
    rng=np.random.RandomState(42)
    # 평가 대상: 2024에서 3만 행 표본 (스크리닝 목적)
    va_idx=rng.choice(len(va),size=min(30000,len(va)),replace=False)
    va_s=va.iloc[va_idx].reset_index(drop=True)
    cols=[c for c in df.columns if c not in (ID,TARGET)]
    def prep(d):
        x=d[cols].copy()
        for c in CATS: x[c]=x[c].astype(str)
        return pd.concat([x.reset_index(drop=True),
                          add_features(d).reset_index(drop=True),
                          add_hfeatures(d).reset_index(drop=True)],axis=1)
    Xva=prep(va_s); yva=va_s[TARGET].to_numpy()
    print(f"평가 표본 {Xva.shape}  실제 성공률 {yva.mean():.4f}  ({time.time()-t0:.0f}s)",flush=True)
    from autogluon.tabular import TabularPredictor, TabularDataset
    rows=[]
    for n_ctx in [5000, 20000]:
        try:
            tt=time.time()
            ctx_idx=rng.choice(len(tr),size=min(n_ctx,len(tr)),replace=False)
            ctx=tr.iloc[ctx_idx].reset_index(drop=True)
            Xtr=prep(ctx); Xtr[TARGET]=ctx[TARGET].to_numpy()
            pred=TabularPredictor(label=TARGET, eval_metric="log_loss",
                                  path=str(HERE/f"_mitra_{n_ctx}"), verbosity=1)
            pred.fit(TabularDataset(Xtr), hyperparameters={"MITRA":{"fine_tune":False}},
                     time_limit=3600)
            p=pred.predict_proba(TabularDataset(Xva))[1].to_numpy()
            q=p+(yva.mean()-p.mean())
            br=float(np.mean((q-yva)**2))
            rows.append({"n_ctx":n_ctx,"brier":br,"sec":round(time.time()-tt)})
            print(f"[Mitra ctx={n_ctx}] Brier {br:.8f}  ({time.time()-tt:.0f}s)",flush=True)
            np.save(RES/f"exp101_mitra_ctx{n_ctx}_pred.npy", p)
            va_s[[ID]].to_csv(RES/"exp101_mitra_rowids.csv",index=False)
            del pred; gc.collect()
        except Exception as e:
            print(f"[Mitra ctx={n_ctx}] 실패: {e}",flush=True)
            traceback.print_exc()
    if rows:
        pd.DataFrame(rows).to_csv(RES/"exp101_mitra.csv",index=False,encoding="utf-8-sig")
    print(f"total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""DeepFM+sd 체인 한계기여 확인."""
import numpy as np, pandas as pd, sys, importlib.util
from pathlib import Path
HERE=Path("/mnt/c/Users/nynu0/Desktop/LG해커톤/comp"); RES=HERE/"results"
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("ec",HERE/"eval_candidate.py")
ec=importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
b,gt=ec.load(); y=b.control_success.to_numpy(); season=b.season.to_numpy()
rel=(gt=="R")|(season>=2023)
Z=np.load(RES/"material_cache.npz",allow_pickle=True)
def M(nm): return Z[nm].astype(np.float64)
def col(fn):
    d=pd.read_csv(RES/fn); return b.row_id.map(d.set_index("row_id")["prediction"]).to_numpy()
def seedavg(fn):
    d=pd.read_csv(RES/fn); sc=[c for c in d.columns if c.startswith("p") and c!="prediction"]
    sc=sc if sc else ["prediction"]
    return d.set_index("row_id")[sc].mean(axis=1)
bb={"ms":b.row_id.map(seedavg("exp68_multisplit_oof.csv.gz")).to_numpy(),
    "hc5":b.row_id.map(seedavg("exp65_handcnt_seeds5_oof.csv.gz")).to_numpy()}
bigv=b.row_id.map(pd.concat([seedavg("exp84_hand_big_oof.csv.gz"),seedavg("exp84b_2023_oof.csv.gz")])).to_numpy()
W6=[0.06726666,0.11952455,0.15582993,0.17546554,0.23856908,0.24334425]
C6=["hgb_domain","cat_domain","tm_mean","league_role","cat_time","lgbm"]
flat={c:0.80*0.90*0.65*0.85*w for w,c in zip(W6,C6)}
flat.update({"nn3":0.80*0.90*0.65*0.15,"hand8":0.80*0.90*0.25,"pbhand3":0.80*0.90*0.10,
             "hc3":0.80*0.10,"ms":0.10,"hc5":0.10})
V={k:(bb[k] if k in bb else b[k].to_numpy()) for k in flat}
for k,fn in {"hgb_domain":"exp81_hf_hgb_domain_oof.csv.gz","tm_mean":"exp81_hf_tm_mean_oof.csv.gz",
            "league_role":"exp81_hf_league_role_oof.csv.gz","cat_domain":"exp81_hf_cat_domain_oof.csv.gz",
            "cat_time":"exp81_hf_cat_time_oof.csv.gz"}.items(): V[k]=col(fn)
s=sum(flat.values()); hfc=sum((v/s)*V[k] for k,v in flat.items())
chain=0.85*(0.90*hfc+0.10*bigv)+0.15*col("exp60_recent2_oof.csv.gz")
C=chain.mean(); sharp=np.clip(C+1.05*(chain-C),0,1)
sdb=M("exp110_sd_pit_bat"); sda=M("exp110_sd_all")
d24=pd.read_csv(RES/"exp86_B_8ep_코사인_oof.csv.gz"); d22=pd.read_csv(RES/"exp109_B_8ep_코사인_2022_oof.csv.gz")
nn8=b.row_id.map(pd.concat([d24.set_index("row_id")["prediction"],d22.set_index("row_id")["prediction"]])).to_numpy(np.float64)
cs24=pd.read_csv(RES/"exp112_cat_sd_oof.csv.gz"); cs23=pd.read_csv(RES/"exp112b_cat_sd_2023_oof.csv.gz")
cats=b.row_id.map(pd.concat([cs24.set_index("row_id")["prediction"],cs23.set_index("row_id")["prediction"]])).to_numpy(np.float64)
nnsd=b.row_id.map(pd.read_csv(RES/"exp120_nn_sd_oof.csv.gz").set_index("row_id")["prediction"]).to_numpy()
p=sharp.copy()
m1=np.isfinite(sdb); p[m1]=0.60*sharp[m1]+0.40*sdb[m1]
m2=np.isfinite(nn8); p[m2]=0.85*p[m2]+0.15*nn8[m2]
m3=np.isfinite(sda); p[m3]=0.90*p[m3]+0.10*sda[m3]
m4=np.isfinite(cats); p[m4]=0.90*p[m4]+0.10*cats[m4]
m5=np.isfinite(nnsd); p[m5]=0.55*p[m5]+0.45*nnsd[m5]
champ=p
def ev(q,yr):
    mm=(season==yr)&rel
    z=q[mm]+(y[mm].mean()-q[mm].mean()); return float(np.mean((z-y[mm])**2))
b24,b22=ev(champ,2024),ev(champ,2022)
print(f"pent 가상챔피언  2024 {b24:.8f}  2022 {b22:.8f}\n")

import os
def get(fn):
    p=RES/fn
    if not p.exists(): return None
    return b.row_id.map(pd.read_csv(p).set_index("row_id")["prediction"]).to_numpy(np.float64)

def report(nm, v, ws=(0.05,0.10,0.15,0.20,0.25,0.30,0.40)):
    if v is None: print(f"{nm}: 파일 없음\n"); return None
    m=np.isfinite(v)
    print(f"=== {nm} ===")
    print(f"  상관 {np.corrcoef(champ[m],v[m])[0,1]:.4f}   단독 2024 {ev(v,2024):.8f}  2022 {ev(v,2022):.8f}")
    best=None
    for w in ws:
        q=champ.copy(); q[m]=(1-w)*champ[m]+w*v[m]
        d24=(b24-ev(q,2024))*1e5; d22=(b22-ev(q,2022))*1e5
        sc_,sh=ec.score(q,y,season,rel)
        ok=(d24>0 and d22>0)
        print(f"    {int(w*100):>2}%:  2024 {d24:+7.2f}  2022 {d22:+7.2f}  shift {sh:.12f}{'  <<< 통과' if ok else ''}")
        if ok and (best is None or (d24+d22)>(best[1]+best[2])): best=(w,d24,d22,sh)
    print(f"  -> 최적 {int(best[0]*100)}% (2024 {best[1]:+.2f} / 2022 {best[2]:+.2f})" if best else "  -> 통과 비중 없음")
    print()
    return best


def get(fn):
    p=RES/fn
    return b.row_id.map(pd.read_csv(p).set_index("row_id")["prediction"]).to_numpy(np.float64) if p.exists() else None

# ---- hepta(현 챔피언 1054.41) 재구성 ----
a3=get("exp120_nn_sd_oof.csv.gz"); a5=get("exp133_nnsd_seeds5_oof.csv.gz"); avg8=(3.0*a3+5.0*a5)/8.0
tm3=get("exp122_nn_sdtm_oof.csv.gz"); rn3=get("exp131_nn_rank_oof.csv.gz"); lg3=get("exp135_nn_loadgap_oof.csv.gz")
def chain(tm=None, rn=None, lg=None, extra=None):
    q=sharp.copy()
    q[m1]=0.60*sharp[m1]+0.40*sdb[m1]; q[m2]=0.85*q[m2]+0.15*nn8[m2]
    q[m3]=0.90*q[m3]+0.10*sda[m3];     q[m4]=0.90*q[m4]+0.10*cats[m4]
    mb=np.isfinite(avg8); q[mb]=0.55*q[mb]+0.45*avg8[mb]
    for v,w in [(tm if tm is not None else tm3,0.10),(rn if rn is not None else rn3,0.10),
                (lg if lg is not None else lg3,0.15)]:
        mm=np.isfinite(v); q[mm]=(1-w)*q[mm]+w*v[mm]
    if extra is not None:
        v,w=extra; mm=np.isfinite(v); q[mm]=(1-w)*q[mm]+w*v[mm]
    return q
HEP=chain(); h24=ev(HEP,2024); h22=ev(HEP,2022)
print("#"*74); print("# 밤샘 자동 판정  (기준 = hepta_v1, 리더보드 1054.4065)"); print("#"*74)
print(f"hepta 기준  2024 {h24:.8f}  2022 {h22:.8f}\n")


import glob, os
print("="*74); print("[NN 튜닝 후보] 체인 한계기여 (기준 = hepta, 리더보드 1054.4065)"); print("="*74)
print(f"현행 nnsd8 단독:  2024 {ev(avg8,2024):.8f}  2022 {ev(avg8,2022):.8f}\n")
files=sorted(glob.glob(str(RES/"exp146_*_oof.csv.gz")))
if not files: print("정밀검증 산출물 없음")
best=None
for f in files:
    tag=os.path.basename(f).replace("exp146_","").replace("_oof.csv.gz","")
    v=b.row_id.map(pd.read_csv(f).set_index("row_id")["prediction"]).to_numpy(np.float64)
    m=np.isfinite(v)
    s24,s22=ev(v,2024),ev(v,2022)
    d_solo24=(ev(avg8,2024)-s24)*1e5; d_solo22=(ev(avg8,2022)-s22)*1e5
    print(f"--- {tag} ---")
    print(f"    단독 2024 {s24:.8f} ({d_solo24:+.2f} vs nnsd8)   2022 {s22:.8f} ({d_solo22:+.2f})")
    print(f"    상관(hepta) {np.corrcoef(HEP[m],v[m])[0,1]:.4f}   상관(nnsd8) {np.corrcoef(avg8[m],v[m])[0,1]:.4f}")
    # (A) nnsd 슬롯 교체
    q=sharp.copy()
    q[m1]=0.60*sharp[m1]+0.40*sdb[m1]; q[m2]=0.85*q[m2]+0.15*nn8[m2]
    q[m3]=0.90*q[m3]+0.10*sda[m3];     q[m4]=0.90*q[m4]+0.10*cats[m4]
    q[m]=0.55*q[m]+0.45*v[m]
    for vv,w in [(tm3,0.10),(rn3,0.10),(lg3,0.15)]:
        mm=np.isfinite(vv); q[mm]=(1-w)*q[mm]+w*vv[mm]
    a24=(h24-ev(q,2024))*1e5; a22=(h22-ev(q,2022))*1e5
    sc_,sh=ec.score(q,y,season,rel)
    print(f"    [교체형] 2024 {a24:+6.2f}  2022 {a22:+6.2f}  shift {sh:.12f}  {'<<< 통과' if (a24>0 and a22>0) else ''}")
    # (B) 새 멤버로 추가
    bestadd=None
    for w in [0.10,0.15,0.20,0.25]:
        q2=chain(extra=(v,w))
        c24=(h24-ev(q2,2024))*1e5; c22=(h22-ev(q2,2022))*1e5
        if c24>0 and c22>0 and (bestadd is None or c24+c22>bestadd[1]+bestadd[2]):
            bestadd=(w,c24,c22)
    if bestadd:
        w,c24,c22=bestadd
        sc_,sh2=ec.score(chain(extra=(v,w)),y,season,rel)
        print(f"    [추가형] 최적 {int(w*100)}%: 2024 {c24:+6.2f}  2022 {c22:+6.2f}  shift {sh2:.12f}  <<< 통과")
    else:
        print(f"    [추가형] 통과 비중 없음")
    cand=max([x for x in [(a24,a22,'교체'),(bestadd[1],bestadd[2],'추가') if bestadd else None] if x],
             key=lambda z:z[0]+z[1])
    if cand[0]>0 and cand[1]>0 and (best is None or cand[0]+cand[1]>best[1]+best[2]):
        best=(tag,cand[0],cand[1],cand[2])
    print()
if best:
    print(f"[최선] {best[0]} ({best[3]}형)  2024 {best[1]:+.2f} / 2022 {best[2]:+.2f}")
    print(f"       평균 {(best[1]+best[2])/2:+.2f}e-5  이론 {(best[1]+best[2])/2*4.0:+.1f}점  실현기대 {(best[1]+best[2])/2*4.0*0.37:+.1f}점")
else:
    print("[결과] 통과 후보 없음")

# -*- coding: utf-8 -*-
"""asof 3종의 실제 의미 검증: 정말 실패 3유형에 대응하는가?"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
d = pd.read_pickle(HERE/"results"/"exp49_base.pkl")
R = d[(d.game_type=="R") & (d.asof_pitcher_n>=500)].copy()
for k,v in [("rev",1),("mid",2),("bal",3)]:
    R[k] = (R.cls==v).astype(float)
R["suc"] = (R.cls==0).astype(float)

print("실제 발생률 vs asof 평균 (asof_n>=500 행)")
print("%-16s %10s %10s" % ("", "실제평균", "asof평균"))
for a,b,nm in [("suc","asof_pitcher_success_rate","성공"),
               ("rev","asof_pitcher_reverse_rate","의도반대"),
               ("mid","asof_pitcher_middle_rate","한가운데"),
               ("bal","asof_pitcher_ball_rate","3번째실패?")]:
    print("%-16s %10.4f %10.4f" % (nm, R[a].mean(), R[b].mean()))
print("%-16s %10s %10.4f" % ("strike_rate", "-", R.asof_pitcher_strike_rate.mean()))
print("\n합계 확인: rev+mid+bal(asof) = %.4f, success+그것 = %.4f"
      % ((R.asof_pitcher_reverse_rate+R.asof_pitcher_middle_rate+R.asof_pitcher_ball_rate).mean(),
         (R.asof_pitcher_success_rate+R.asof_pitcher_reverse_rate
          +R.asof_pitcher_middle_rate+R.asof_pitcher_ball_rate).mean()))
print("ball+strike(asof) = %.4f" % (R.asof_pitcher_ball_rate+R.asof_pitcher_strike_rate).mean())

print("\n투수 단위 상관 (n>=2000 투수, 시즌<2024):")
g = R[R.season<2024].groupby("pitcher_id").agg(
    n=("suc","size"), suc=("suc","mean"), rev=("rev","mean"),
    mid=("mid","mean"), bal=("bal","mean"),
    a_suc=("asof_pitcher_success_rate","mean"), a_rev=("asof_pitcher_reverse_rate","mean"),
    a_mid=("asof_pitcher_middle_rate","mean"), a_bal=("asof_pitcher_ball_rate","mean"))
g = g[g.n>=2000]
print("  투수수 %d" % len(g))
for a,b,nm in [("suc","a_suc","성공"),("rev","a_rev","의도반대"),
               ("mid","a_mid","한가운데"),("bal","a_bal","3번째실패 vs asof_ball")]:
    print("  %-24s corr=%+.4f" % (nm, np.corrcoef(g[a],g[b])[0,1]))
print("  %-24s corr=%+.4f" % ("3번째실패 vs asof_strike",
      np.corrcoef(g.bal, R[R.season<2024].groupby("pitcher_id").asof_pitcher_strike_rate.mean().loc[g.index])[0,1]))

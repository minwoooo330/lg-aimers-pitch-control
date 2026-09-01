# -*- coding: utf-8 -*-
"""실험 49: 실패 유형별 메커니즘 조사 (기술통계).

asof 3종(reverse/middle/ball)을 실패 메커니즘 신호로 취급하고,
각 유형을 유발하는 '상황' 조건을 리그 수준에서 직접 측정한다.
투수 고유 성향과 분리하기 위해 투수 기준선을 뺀 잔차도 함께 본다.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 200)

d = pd.read_pickle(HERE/"results"/"exp49_base.pkl")
R = d[(d.game_type == "R")].copy()          # 1군만 (체제 혼입 회피)
R["rev"] = (R.cls == 1).astype(np.float32)
R["mid"] = (R.cls == 2).astype(np.float32)
R["bal"] = (R.cls == 3).astype(np.float32)
R["suc"] = (R.cls == 0).astype(np.float32)
R["cnt"] = R.balls_before.astype(str) + "-" + R.strikes_before.astype(str)

def tab(g, name, minn=3000):
    t = R.groupby(g).agg(n=("suc","size"), 성공=("suc","mean"),
                         의도반대=("rev","mean"), 한가운데=("mid","mean"),
                         크게벗어남=("bal","mean"))
    t = t[t.n >= minn]
    print("\n=== %s ===" % name)
    print(t.round(4).to_string())
    return t

print("리그(1군) 전체 %d행" % len(R))
base = R[["suc","rev","mid","bal"]].mean()
print("전체 평균:", base.round(4).to_dict())

tab("cnt", "카운트별 실패 유형 구성")
tab("num_runners_on", "주자 수별")
tab(pd.cut(R.inning, [0,3,6,9,20], labels=["1-3","4-6","7-9","10+"]), "이닝 구간별")
tab([R.pitcher_hand, R.batter_hand], "투타 손잡이 조합별")
tab("outs_before", "아웃 카운트별")
tab(pd.cut(R.score_diff_pitcher_team, [-99,-6,-2,-1,0,1,2,6,99]), "점수차별")
tab("game_month", "월별")
tab(pd.cut(R.li, [0,0.5,1.0,1.5,2.5,99]), "LI 구간별")

# --- 투수 기준선 제거 후에도 상황 효과가 남는가 ---
print("\n\n=== 투수 기준선(asof) 제거 후 잔차 평균 ===")
print("(양수 = 그 상황에서 투수 평소보다 해당 실패가 더 잦다)")
R["e_rev"] = R.rev - R.asof_pitcher_reverse_rate
R["e_mid"] = R.mid - R.asof_pitcher_middle_rate
R["e_bal"] = R.bal - R.asof_pitcher_ball_rate
for g, nmg in [("cnt","카운트"), ("num_runners_on","주자수"),
               (pd.cut(R.inning,[0,3,6,9,20],labels=["1-3","4-6","7-9","10+"]),"이닝")]:
    t = R.groupby(g).agg(n=("e_rev","size"), d의도반대=("e_rev","mean"),
                         d한가운데=("e_mid","mean"), d크게벗어남=("e_bal","mean"))
    print("\n-- %s --" % nmg)
    print(t[t.n>=3000].round(4).to_string())

# -*- coding: utf-8 -*-
"""실험 74: '투수 상수 x 행가변 맥락' 곱 전수 스크리닝 — 커버리지 구멍 메우기.

동기: 팀원의 breaking_x_same_hand는 내가 세운 필터("투수 상수 x 행가변 맥락이
유일한 승리 범주")가 정확히 지목하는 조합인데 내가 안 봤다. 팔각도로는
손잡이와 교차했으면서 구종으로는 단독만 봤다. 같은 누락이 더 있을 것이므로
가능한 곱을 전수로 훑는다.

특히 실패유형 성향 x 카운트가 유망하다. 내 exp45/46에서 실패 유형 구성이
카운트에 강하게 의존함을 확인했다(3-0: 의도반대 0.2415/한가운데 0.1815,
0-2: 0.1394/0.1180). 투수마다 실패유형 프로필이 다르므로, 그 프로필과
카운트의 곱은 야구적으로 근거가 있고 아직 아무도 안 봤다.

주의(2026-08-21 확인): 체인 실제 HGB는 max_iter=200/lr=.06/leaves=31이며
tuned_hgb_params.json(600/.027/15/max_features=.63)이 아니다. 내 exp62/71이
잘못된 기준선을 썼다. 여기서는 잔차를 체인 파라미터와 일치하는
exp04_hgb_domain_oof + 최신 NN(exp63_handcnt)으로 근사한다.

중심화: 주효과는 이미 모델에 있으므로 (x-x̄)(m-m̄) 형태로 상호작용만 분리한다.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
TARGET = "control_success"
USE = ["row_id", "season", "game_type", "inning", "top_bottom", "balls_before",
       "strikes_before", "outs_before", "num_runners_on", "li",
       "score_diff_pitcher_team", "pitcher_hand", "batter_hand",
       "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
       "asof_pitcher_reverse_rate", "asof_pitcher_ball_rate",
       "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
       "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate", TARGET]


def load(f, name):
    d = pd.read_csv(HERE / "results" / f)
    c = [x for x in d.columns if x in ("prediction", "pred", "y_pred")][0]
    return d[["row_id", c]].rename(columns={c: name})


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", usecols=USE)
    # 체인 파라미터와 일치하는 GBDT OOF + 최신 채택 NN
    g = load("exp04_hgb_domain_oof.csv.gz", "pg")
    try:
        n = load("exp63_handcnt_oof.csv.gz", "pn")
    except Exception:
        n = load("exp55_hand_seedavg_oof.csv.gz", "pn")
    d = df.merge(g, on="row_id").merge(n, on="row_id")
    d["p"] = 0.75 * d.pg + 0.25 * d.pn
    d["resid"] = d[TARGET] - d.p
    d = d[(d.game_type == "R") | (d.season >= 2023)].copy()
    for y in (2022, 2023, 2024):
        s = d[d.season == y]
        print("기준선 %d Brier %.6f (n=%d)" % (y, (s.resid ** 2).mean(), len(s)))

    # 투수 상수 (as-of, 행마다 값은 같은 투수면 거의 불변)
    P = {
        "실력": d.asof_pitcher_success_rate.fillna(0.52),
        "한가운데성향": d.asof_pitcher_middle_rate.fillna(0.15),
        "의도반대성향": d.asof_pitcher_reverse_rate.fillna(0.20),
        "볼성향": d.asof_pitcher_ball_rate.fillna(0.35),
        "직구비중": d.asof_pitcher_fastball_rate.fillna(1 / 3),
        "변화구비중": d.asof_pitcher_breaking_rate.fillna(1 / 3),
        "완급비중": d.asof_pitcher_offspeed_rate.fillna(1 / 3),
        "경력": np.log1p(d.asof_pitcher_n.fillna(0)),
    }
    # 행가변 맥락
    adv = np.sign(d.strikes_before - d.balls_before)
    M = {
        "같은손": (d.pitcher_hand == d.batter_hand).astype(float),
        "카운트우열": adv.astype(float),
        "3볼": (d.balls_before == 3).astype(float),
        "2스트라이크": (d.strikes_before == 2).astype(float),
        "이닝후반": (d.inning >= 7).astype(float),
        "주자있음": (d.num_runners_on > 0).astype(float),
        "고LI": (d.li >= 2.0).astype(float),
        "접전": (d.score_diff_pitcher_team.abs() <= 2).astype(float),
        "퓨처스": (d.game_type == "F").astype(float),
    }
    print("\n=== 중심화 곱 전수 스크리닝 (2025 유형 행, |상관| 표시) ===")
    print("문턱: 0.0105 통과(손잡이 임베딩) / 0.0073 기각(카운트 임베딩)\n")
    hdr = "%-14s" % "투수상수\\맥락"
    for m in M:
        hdr += "%8s" % m
    print(hdr); print("-" * len(hdr))
    rows = []
    for pn_, pv in P.items():
        line = "%-14s" % pn_
        for mn, mv in M.items():
            cl = []
            for y in (2022, 2024):           # clean fold만
                s = d.season == y
                x = (pv[s] - pv[s].mean()) * (mv[s] - mv[s].mean())
                if x.std() < 1e-12:
                    cl.append(0.0); continue
                cl.append(abs(np.corrcoef(x, d.resid[s])[0, 1]))
            v = min(cl)
            rows.append((pn_, mn, v, cl))
            line += "%8.4f" % v
        print(line)
    print("\n=== 상위 12개 (clean fold 최소 기준) ===")
    rows.sort(key=lambda r: -r[2])
    print("%-14s %-12s %10s %9s %9s %s" % ("투수상수", "맥락", "clean최소", "2022", "2024", "판정"))
    for pn_, mn, v, cl in rows[:12]:
        mk = "통과" if v >= 0.0105 else ("경계" if v >= 0.0073 else "미달")
        print("%-14s %-12s %10.5f %9.5f %9.5f %s" % (pn_, mn, v, cl[0], cl[1], mk))
    best = rows[0]
    print("\n최대값 %.5f (%s x %s) — 문턱 0.0105 대비 %.2f배"
          % (best[2], best[0], best[1], best[2] / 0.0105))


if __name__ == "__main__":
    main()

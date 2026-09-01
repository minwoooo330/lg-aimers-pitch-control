# -*- coding: utf-8 -*-
"""실험 61: 투수 x 행가변 맥락 조건부 재스크리닝 (fold별 이전성 포함).

지금까지 결과가 하나의 규칙으로 정리된다.
  투수 단위 상수 (퓨처스이력/Trackman프로필/팔각도/구종비중)  전부 기각
  집단 단위 상수 (팀/구장)                                 전부 기각
  투수 x 행가변 맥락 (타자손잡이)                           유일한 성공
pitcher_id가 이미 투수 개인을 담고 있으므로 투수마다 고정된 값은 그 요약본일
뿐이다. 값이 있으려면 같은 투수가 행마다 다른 값을 가져야 한다.

기존 스크리닝은 투수 x {손잡이, 카운트, 이닝, 주자수, LI}만 했다.
아래는 미시도 축이며, game_type(리그)은 팀원 제안에서 나왔다.

방법 개선 두 가지:
  1) 기준선을 최신 강한 OOF로 (exp30_tuned_league_role + exp55_hand_seedavg)
     약한 모델의 잔차에 보이는 신호는 강한 모델이 이미 흡수했을 수 있다.
  2) fold별로 따로 측정. 한 시즌 측정은 이전성을 보장하지 않는다
     (카운트 축 exp53, 팀/구장 축 exp56에서 두 번 확인).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
K = 200.0
TARGET = "control_success"
USE = ["row_id", "season", "game_month", "game_type", "top_bottom", "inning",
       "balls_before", "strikes_before", "outs_before", "score_diff_pitcher_team",
       "num_runners_on", "li", "pitcher_id", "batter_id", "batter_team_id",
       "pitcher_hand", "batter_hand", "asof_pitcher_n", TARGET]


def prep(d):
    d = d.copy()
    d["_adv"] = np.sign(d.strikes_before - d.balls_before).astype(int)
    d["_set"] = (d.num_runners_on > 0).astype(int)
    d["_sd"] = pd.cut(d.score_diff_pitcher_team, [-99, -4, -1, 1, 4, 99],
                      labels=[0, 1, 2, 3, 4]).astype(float)
    d["_mon"] = d.game_month.clip(4, 9)
    d["_out"] = d.outs_before
    d["_tb"] = d.top_bottom.astype(str)
    return d


def main():
    df = prep(pd.read_csv(HERE / "data" / "train.csv", usecols=USE))
    o1 = pd.read_csv(HERE / "results" / "exp30_tuned_league_role_oof.csv.gz")
    o2 = pd.read_csv(HERE / "results" / "exp55_hand_seedavg_oof.csv.gz")
    c1 = [c for c in o1.columns if c in ("prediction", "pred", "y_pred")][0]
    c2 = [c for c in o2.columns if c in ("prediction", "pred", "y_pred")][0]
    o = (o1[["row_id", "season", TARGET, c1]].rename(columns={c1: "pg"})
         .merge(o2[["row_id", c2]].rename(columns={c2: "pn"}), on="row_id"))
    o["p"] = 0.85 * o.pg + 0.15 * o.pn
    o["resid"] = o[TARGET] - o.p
    d = df.merge(o[["row_id", "p", "resid"]], on="row_id")
    for y in (2022, 2023, 2024):
        s = d[d.season == y]
        print("기준선 %d Brier %.6f" % (y, (s.resid ** 2).mean()))
    print()

    AXES = [
        ("투수 x 리그(game_type)", ["pitcher_id", "game_type"], ["pitcher_id", "game_type"]),
        ("투수 x 월", ["pitcher_id", "_mon"], ["pitcher_id", "_mon"]),
        ("투수 x 점수차구간", ["pitcher_id", "_sd"], ["pitcher_id", "_sd"]),
        ("투수 x 아웃카운트", ["pitcher_id", "_out"], ["pitcher_id", "_out"]),
        ("투수 x 홈원정", ["pitcher_id", "_tb"], ["pitcher_id", "_tb"]),
        ("투수 x 세트포지션", ["pitcher_id", "_set"], ["pitcher_id", "_set"]),
        ("투수 x 타자팀", ["pitcher_id", "batter_team_id"], ["pitcher_id", "batter_team_id"]),
        ("[대조] 투수 x 타자손잡이", ["pitcher_id", "batter_hand"], ["pitcher_id", "batter_hand"]),
        ("[대조] 투수 x 카운트우열", ["pitcher_id", "_adv"], ["pitcher_id", "_adv"]),
    ]

    print("%-24s %8s %8s %8s | %9s %6s" % ("축", "2022", "2023", "2024", "평균", "일관"))
    print("-" * 74)
    for name, keys, curkeys in AXES:
        cors, covs = [], []
        for y in (2022, 2023, 2024):
            hist = df[(df.season < y) & ~((df.game_type == "F") & (df.season <= 2022))]
            cur = d[d.season == y]
            if len(hist) == 0:
                cors.append(np.nan); continue
            prior = hist[TARGET].mean()
            gp = hist.groupby("pitcher_id")[TARGET].agg(["sum", "count"])
            marg = (gp["sum"] + K * prior) / (gp["count"] + K)
            gg = hist.groupby(keys, observed=True)[TARGET].agg(["sum", "count"])
            pm = gg.index.get_level_values(0).map(marg)
            dev = ((gg["sum"] + K * pm) / (gg["count"] + K)) - pm
            idx = pd.MultiIndex.from_arrays([cur[k].to_numpy() for k in curkeys])
            v = dev.reindex(idx).to_numpy()
            r = cur.resid.to_numpy()
            ok = np.isfinite(v)
            covs.append(ok.mean())
            cors.append(np.corrcoef(v[ok], r[ok])[0, 1] if ok.sum() > 3000 else np.nan)
        arr = np.array(cors, dtype=float)
        good = np.isfinite(arr)
        same = "O" if good.sum() == 3 and (np.all(arr > 0.002) or np.all(arr < -0.002)) else "-"
        print("%-24s %8s %8s %8s | %+9.5f %6s" % (
            name,
            "%+.5f" % arr[0] if good[0] else "  n/a",
            "%+.5f" % arr[1] if good[1] else "  n/a",
            "%+.5f" % arr[2] if good[2] else "  n/a",
            np.nanmean(arr), same))
    print("\n※ 일관 O = 세 fold 모두 같은 부호이고 크기가 0.002 이상")
    print("※ 대조군: 타자손잡이는 실제 채택 성공, 카운트우열은 실제 기각(exp53/54)")


if __name__ == "__main__":
    main()

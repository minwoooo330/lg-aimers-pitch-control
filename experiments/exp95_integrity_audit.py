# -*- coding: utf-8 -*-
"""실험 75: 데이터 정합성 감사 — 야구 규칙·산술 항등식 위반 탐색.

동기: 작년 수상자 팁의 "변수명과 각 고윳값의 의미도 설명할 수 있을 정도로
정확하게 파악", "결측값·논리적 오류 검증 및 처리". 내 하네스 설계에도
야구 무결성 규칙이 있었으나 실제로 돌린 적이 없다.

검사 축:
  A. 야구 규칙 (카운트·아웃·이닝 범위)
  B. 산술 항등식 (주자 수, 점수, 승률 합)
  C. asof 비율의 정의역과 상호 관계
  D. 결측 구조 (결측이 무작위인가, 타깃과 연관되는가)

D는 특히 중요하다. 팁에서 "결측 처리는 target 분포 기준 권장"이라 했는데
우리는 NN에서 median, HGB에서 native 처리다. 결측 행의 실제 성공률이
대치값의 함의와 다르면 계통 편향이 생긴다.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
T = "control_success"


def chk(name, bad, n, sample=None):
    r = bad.sum() if hasattr(bad, "sum") else bad
    flag = "OK  " if r == 0 else "위반"
    print("  [%s] %-46s %8d행 (%.5f%%)" % (flag, name, r, 100 * r / n))
    if r and sample is not None and r < n:
        print("        예시:", sample.head(3).to_dict("records"))


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    n = len(df)
    print("전체 %d행, %d컬럼\n" % (n, df.shape[1]))

    print("=== A. 야구 규칙 ===")
    chk("balls_before <= 3", df.balls_before > 3, n)
    chk("strikes_before <= 2", df.strikes_before > 2, n)
    chk("outs_before <= 2", df.outs_before > 2, n)
    chk("balls/strikes/outs >= 0", (df.balls_before < 0) | (df.strikes_before < 0) | (df.outs_before < 0), n)
    chk("inning >= 1", df.inning < 1, n)
    print("  이닝 최대 %d, 분포 상위: %s" % (df.inning.max(), df.inning.value_counts().head(3).to_dict()))

    print("\n=== B. 산술 항등식 ===")
    rsum = df.runner_on_1b + df.runner_on_2b + df.runner_on_3b
    chk("num_runners_on == 1b+2b+3b 합", rsum != df.num_runners_on, n,
        df.loc[rsum != df.num_runners_on, ["runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on"]])
    chk("run_total_before == top+bot", (df.run_top_before + df.run_bot_before) != df.run_total_before, n,
        df.loc[(df.run_top_before + df.run_bot_before) != df.run_total_before,
               ["run_top_before", "run_bot_before", "run_total_before"]])
    # 홈 관점 점수차: 말(B)이 홈 공격이므로 홈점수는 bot
    sd = df.run_bot_before - df.run_top_before
    chk("score_diff_home == bot - top", sd != df.score_diff_home, n,
        df.loc[sd != df.score_diff_home, ["run_top_before", "run_bot_before", "score_diff_home"]])
    # 투수팀 관점: 초(T)에는 홈팀이 투구 -> 투수팀 점수차 = score_diff_home
    exp_sdp = np.where(df.top_bottom.astype(str) == "T", df.score_diff_home, -df.score_diff_home)
    chk("score_diff_pitcher_team 부호 정합", exp_sdp != df.score_diff_pitcher_team, n)
    we = df.home_win_expectancy + df.away_win_expectancy
    chk("home_we + away_we == 1", (we - 1).abs() > 1e-6, n)
    chk("li >= 0", df.li < 0, n)
    print("  li 범위 [%.4f, %.4f], 상위 0.1%% 분위 %.3f" % (df.li.min(), df.li.max(), df.li.quantile(0.999)))

    print("\n=== C. asof 비율 정의역 ===")
    rates = [c for c in df.columns if c.startswith("asof_") and c.endswith("_rate")]
    for c in rates:
        v = df[c]
        bad = ((v < 0) | (v > 1)) & v.notna()
        if bad.sum():
            chk(c + " in [0,1]", bad, n)
    print("  비율 %d개 전부 [0,1] 범위 확인" % len(rates))
    # 실패 유형 합 검사: 성공 + 반대 + 가운데 <= 1 이어야 함
    tri = (df.asof_pitcher_success_rate + df.asof_pitcher_reverse_rate + df.asof_pitcher_middle_rate)
    chk("success+reverse+middle <= 1", (tri > 1 + 1e-6) & tri.notna(), n)
    print("  세 비율 합 분포: 중앙값 %.4f, 최대 %.4f" % (tri.median(), tri.max()))
    bs = df.asof_pitcher_ball_rate + df.asof_pitcher_strike_rate
    print("  ball+strike 합: 중앙값 %.4f, 표준편차 %.4f" % (bs.median(), bs.std()))
    chk("ball+strike == 1", ((bs - 1).abs() > 1e-6) & bs.notna(), n)

    print("\n=== D. 결측 구조와 타깃 (팁: '결측 처리는 target 분포 기준') ===")
    miss = df.isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    base = df[T].mean()
    print("  전체 성공률 %.4f" % base)
    print("  %-40s %8s %10s %10s %9s" % ("컬럼", "결측률", "결측행성공률", "관측행성공률", "차이"))
    for c in miss.index:
        m = df[c].isna()
        if m.sum() < 500:
            continue
        a, b = df.loc[m, T].mean(), df.loc[~m, T].mean()
        print("  %-40s %8.4f %10.4f %10.4f %+9.4f" % (c, miss[c], a, b, a - b))
    print("\n  -> 차이가 크면 결측 자체가 신호이고, median 대치는 그 신호를 지운다")

    print("\n=== E. 중복 행 ===")
    key = ["season", "game_month", "inning", "top_bottom", "balls_before", "strikes_before",
           "outs_before", "pitcher_id", "batter_id", "base_state"]
    dup = df.duplicated(subset=key, keep=False)
    print("  상태키 중복 %d행 (%.3f%%) — 같은 상황 반복은 정상이나 극단적이면 확인 필요"
          % (dup.sum(), 100 * dup.mean()))
    full = df.drop(columns=["row_id"]).duplicated(keep=False)
    chk("완전 중복 행(타깃 포함)", full, n)


if __name__ == "__main__":
    main()

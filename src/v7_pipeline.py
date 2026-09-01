"""KBO 제구 예측 — 학습 진입점 (v7, 리더보드 1099.4774).

실행:
    export PYTHONPATH=.

    # (1) 학습 — CatBoost 5설정 x 3시드 = 15모델, 피처 175개. 24코어 약 60분.
    #     결과는 submit/model/ 에 저장된다 (.cbm 15개 + 캐리 테이블 + meta.json)
    python train.py --tag v7 --trackman --mono --legacy-norm --halflife 2         --platoon --l2 1000 --rsm 0.6 --seeds 42,7,2024

    # (2) 캘리브레이션 상수 이월 — 학습 직후 반드시 실행
    python ref_stats.py --bundle submit --apply

필요 입력:
    data/train.csv, data/trackman_history.csv
    work/train.parquet, work/trackman.parquet   (위 csv 의 parquet 변환본)
    트랙맨 ID 대응표

주의: ref_stats.py 가 내놓는 lambda/scale 은 '이월 추정치'다. v7 에서 두 축
모두 크게 빗나갔다 (lambda 0.924 지시 -> 1.389 실측, scale 0.962 -> 1.077).
실제 값은 리더보드에 점을 찍어 포물선으로 확정해야 한다. 곡률은 lambda 71.4
(모델 무관 상수), scale 985 이므로 각각 2점/3점이면 정점이 나온다.

--- 아래는 원본 설계 메모 ---
v3 대비 바뀐 것:

1) 레짐 오염 제거 (--drop-pre-regime-f)
   퓨처스(F)의 제구 성공률은 2019~2022 에 .59~.71 이었다가 2023 에 .47 로 무너졌다.
   2020~2022 F행 79,522개(학습의 6.4%)는 라벨 평균 .6718 로, 현 레짐(.4656)과
   0.21 차이나는 죽은 분포다. 2023 홀드아웃이 모든 조합에서 음수였던 것도 이 때문
   (F행만 떼면 BSS -2315, R행만은 +208).

2) 리그 정규화 단위 수정 (season_form.py / marcel.py)
   role_dev_* 와 mrc_dev 의 기대값을 (시즌 x game_type) 리그 평균으로 잡는다.
   이전에는 role 이 전 구간 통합 평균 하나, marcel 이 시즌 평균만 썼다.

3) recency 가중 (--halflife)
   시즌 단위 지수 감쇠. 0 이면 균등(v3 와 동일).

4) 캘리브레이션 상수 자동 이월
   λ*/scale* 는 리더보드 6점/4점 실측으로 확정돼 있다 (곡률 70.6 / 1002).
   새 모델은 예측 평균과 분산이 달라지므로 그만큼만 옮겨서 재탐색 비용을 없앤다.
       λ*_new     = λ*_ref + (m_new - m_ref) / 0.0134812
       scale*_new = scale*_ref x (std_ref / std_new)
   기준(v3, LB 1073.73): m_ref=0.487700, std_ref=0.122990, λ*=0.4017, scale*=1.0869"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from catboost import Pool


REQUIREMENTS = """numpy==2.4.1
pandas==2.3.3
lightgbm==4.7.0
catboost==1.2.10
"""


# ==========================================================================
# calib.py
#   시즌 단위 base rate 외삽 및 로그오즈 shift 보정.
#
#   평가지표가 Brier Skill Score이므로 예측 확률의 전역 수준(mean)이 실제 base rate와
#   어긋나면 그 차이의 제곱만큼 점수가 그대로 깎인다. 학습 구간(2019~2024)에서 제구
#   성공률이 매년 하락해 왔으므로, 트리 모델이 마지막 학습 시즌 수준에 고정되는 것을
#   보정하기 위해 시즌별 rate를 외삽한 목표값으로 로그오즈를 평행이동한다.
#
#   추정기는 2022/2023/2024를 각각 과거 시즌만으로 예측하는 백테스트에서 RMSE가 가장
#   낮았던 두 가지(전체 선형회귀, 최근 3개 시즌 증분 평균)의 평균을 사용한다.
# ==========================================================================
def estimate_base_rate(years, rates, target_year):
    """과거 시즌 rate 시계열로 target_year의 base rate를 추정한다."""
    years = np.asarray(years, dtype=float)
    rates = np.asarray(rates, dtype=float)
    lin = np.polyval(np.polyfit(years, rates, 1), target_year)
    d3 = rates[-1] + np.diff(rates)[-3:].mean()
    return float((lin + d3) / 2)


def apply_logit_shift(p, target_mean, eps=1e-6):
    """예측 확률의 평균이 target_mean이 되도록 로그오즈 공간에서 평행이동한다."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    if not (0 < target_mean < 1):
        return p
    lo = np.log(p / (1 - p))

    def mean_at(b):
        return float(np.mean(1.0 / (1.0 + np.exp(-(lo + b)))))

    lo_b, hi_b = -5.0, 5.0
    if not (mean_at(lo_b) < target_mean < mean_at(hi_b)):
        return p
    for _ in range(60):  # 이분법 (scipy 의존 제거)
        mid = (lo_b + hi_b) / 2
        if mean_at(mid) < target_mean:
            lo_b = mid
        else:
            hi_b = mid
    return np.clip(1.0 / (1.0 + np.exp(-(lo + (lo_b + hi_b) / 2))), eps, 1 - eps)


# ==========================================================================
# features.py
#   투구 직전 정보만 사용하는 피처 빌더. 학습/추론에서 동일하게 사용한다.
# ==========================================================================
ID_COL = "row_id"
TARGET_COL = "control_success"

# 전역 사전 통계(학습셋 기준). shrinkage prior로만 쓴다.
GLOBAL_SUCCESS = 0.5237659752747625
GLOBAL_MIDDLE = 0.16
GLOBAL_REVERSE = 0.25

PRIOR_N = 300.0  # 베이지안 shrinkage 강도


def _shrink(rate, n, prior, prior_n=PRIOR_N):
    """표본 수가 적은 rate를 전역 평균 쪽으로 축소한다(cold-start 안정화)."""
    n = n.fillna(0).clip(lower=0)
    r = rate.fillna(prior)
    return (r * n + prior * prior_n) / (n + prior_n)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)

    # ---- 경기 맥락 ----
    x["season"] = df["season"].astype("float32")
    x["game_month"] = df["game_month"].astype("float32")
    x["game_dayofweek"] = df["game_dayofweek"].astype("float32")
    x["inning"] = df["inning"].astype("float32")
    x["is_bottom"] = (df["top_bottom"] == "B").astype("float32")
    x["is_regular"] = (df["game_type"] == "R").astype("float32")

    # ---- 볼카운트 ----
    b = df["balls_before"].astype("float32")
    s = df["strikes_before"].astype("float32")
    x["balls_before"] = b
    x["strikes_before"] = s
    x["outs_before"] = df["outs_before"].astype("float32")
    x["count_state"] = (b * 3 + s).astype("float32")      # 0~11 고유 카운트
    x["count_diff"] = (s - b).astype("float32")           # +면 투수 유리
    x["count_total"] = (s + b).astype("float32")
    x["two_strikes"] = (s == 2).astype("float32")
    x["three_balls"] = (b == 3).astype("float32")
    x["full_count"] = ((b == 3) & (s == 2)).astype("float32")
    x["first_pitch"] = ((b == 0) & (s == 0)).astype("float32")
    # 투수가 스트라이크를 꼭 넣어야 하는 상황일수록 존 가운데로 몰림 → 제구 실패
    x["must_strike"] = (b - s).clip(lower=0).astype("float32")

    # ---- 점수/주자 상황 ----
    x["run_total_before"] = df["run_total_before"].astype("float32")
    x["score_diff_pitcher_team"] = df["score_diff_pitcher_team"].astype("float32")
    x["abs_score_diff"] = df["score_diff_pitcher_team"].abs().astype("float32")
    x["is_close_game"] = (df["score_diff_pitcher_team"].abs() <= 1).astype("float32")
    x["runner_on_1b"] = df["runner_on_1b"].astype("float32")
    x["runner_on_2b"] = df["runner_on_2b"].astype("float32")
    x["runner_on_3b"] = df["runner_on_3b"].astype("float32")
    x["num_runners_on"] = df["num_runners_on"].astype("float32")
    x["scoring_position"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype("float32")
    x["li"] = df["li"].astype("float32")
    x["log_li"] = np.log1p(df["li"].clip(lower=0)).astype("float32")
    x["home_win_expectancy"] = df["home_win_expectancy"].astype("float32")
    # 투수 팀 기준 기대승률로 통일 (초 공격=원정팀)
    is_bottom = df["top_bottom"] == "B"
    pit_we = np.where(is_bottom, df["away_win_expectancy"], df["home_win_expectancy"])
    x["pitcher_team_win_exp"] = pit_we.astype("float32")
    x["we_uncertainty"] = (50 - np.abs(pit_we - 50)).astype("float32")

    # ---- 선수/팀 ----
    ph = df["pitcher_hand"].astype("float32")
    bh = df["batter_hand"].astype("float32")
    x["pitcher_hand"] = ph
    x["batter_hand"] = bh
    x["same_hand"] = (ph == bh).astype("float32")
    x["pitcher_id"] = df["pitcher_id"].astype("int32")
    x["batter_id"] = df["batter_id"].astype("int32")
    x["pitcher_team_id"] = df["pitcher_team_id"].astype("int32")
    x["batter_team_id"] = df["batter_team_id"].astype("int32")

    # ---- 투수 누적 이력 ----
    pn = df["asof_pitcher_n"].astype("float32")
    x["asof_pitcher_n"] = pn
    x["log_pitcher_n"] = np.log1p(pn).astype("float32")
    x["is_new_pitcher"] = (pn < 50).astype("float32")

    for col, prior in [
        ("asof_pitcher_success_rate", GLOBAL_SUCCESS),
        ("asof_pitcher_reverse_rate", GLOBAL_REVERSE),
        ("asof_pitcher_middle_rate", GLOBAL_MIDDLE),
        ("asof_pitcher_ball_rate", 0.38),
        ("asof_pitcher_strike_rate", 0.42),
    ]:
        x[col] = df[col].astype("float32")
        x[col + "_shrunk"] = _shrink(df[col], pn, prior).astype("float32")

    # 최근 폼(직전 1/3/5경기)과 그 추세
    for col in [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]:
        x[col] = df[col].astype("float32")

    car_s = df["asof_pitcher_success_rate"]
    x["form_1_vs_career"] = (df["asof_pitcher_prev1_game_success_rate"] - car_s).astype("float32")
    x["form_3_vs_career"] = (df["asof_pitcher_prev3_game_success_rate"] - car_s).astype("float32")
    x["form_5_vs_career"] = (df["asof_pitcher_prev5_game_success_rate"] - car_s).astype("float32")
    x["form_1_vs_5"] = (df["asof_pitcher_prev1_game_success_rate"]
                        - df["asof_pitcher_prev5_game_success_rate"]).astype("float32")
    car_m = df["asof_pitcher_middle_rate"]
    x["mid_form_1_vs_career"] = (df["asof_pitcher_prev1_game_middle_rate"] - car_m).astype("float32")
    x["mid_form_5_vs_career"] = (df["asof_pitcher_prev5_game_middle_rate"] - car_m).astype("float32")
    x["prev_form_missing"] = df["asof_pitcher_prev1_game_success_rate"].isna().astype("float32")

    # ---- 타자 누적 이력 ----
    bn = df["asof_batter_n"].astype("float32")
    x["asof_batter_n"] = bn
    x["log_batter_n"] = np.log1p(bn).astype("float32")
    x["asof_batter_success_rate"] = df["asof_batter_success_rate"].astype("float32")
    x["asof_batter_middle_rate"] = df["asof_batter_middle_rate"].astype("float32")
    x["batter_success_shrunk"] = _shrink(df["asof_batter_success_rate"], bn, GLOBAL_SUCCESS).astype("float32")
    x["batter_middle_shrunk"] = _shrink(df["asof_batter_middle_rate"], bn, GLOBAL_MIDDLE).astype("float32")

    # ---- 투수 x 타자 매치업 ----
    ps = _shrink(df["asof_pitcher_success_rate"], pn, GLOBAL_SUCCESS)
    bs = _shrink(df["asof_batter_success_rate"], bn, GLOBAL_SUCCESS)
    x["matchup_success_diff"] = (ps - bs).astype("float32")
    x["matchup_success_sum"] = (ps + bs - GLOBAL_SUCCESS).astype("float32")
    pm = _shrink(df["asof_pitcher_middle_rate"], pn, GLOBAL_MIDDLE)
    bm = _shrink(df["asof_batter_middle_rate"], bn, GLOBAL_MIDDLE)
    x["matchup_middle_sum"] = (pm + bm - GLOBAL_MIDDLE).astype("float32")
    # 로그오즈 결합(오즈비 관점의 매치업 추정치)
    eps = 1e-4
    lo = lambda v: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))
    x["matchup_logodds"] = (lo(ps) + lo(bs) - lo(GLOBAL_SUCCESS)).astype("float32")

    # ---- 구종 믹스 ----
    x["asof_pitcher_pitchmix_n"] = df["asof_pitcher_pitchmix_n"].astype("float32")
    x["asof_pitcher_fastball_rate"] = df["asof_pitcher_fastball_rate"].astype("float32")
    x["asof_pitcher_breaking_rate"] = df["asof_pitcher_breaking_rate"].astype("float32")
    x["asof_pitcher_offspeed_rate"] = df["asof_pitcher_offspeed_rate"].astype("float32")
    x["offspeed_plus_breaking"] = (df["asof_pitcher_breaking_rate"].fillna(0)
                                   + df["asof_pitcher_offspeed_rate"].fillna(0)).astype("float32")
    # 카운트별로 구종 선택이 달라짐 → 카운트 x 구종믹스 상호작용
    x["fastball_x_twostrike"] = (df["asof_pitcher_fastball_rate"].fillna(0.5)
                                 * (s == 2)).astype("float32")
    x["breaking_x_threeball"] = (df["asof_pitcher_breaking_rate"].fillna(0.3)
                                 * (b == 3)).astype("float32")

    return x


CATEGORICAL = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]


# ==========================================================================
# season_form.py
#   현재 시즌 성적 분리(within-season back-out) + 투수 역할/레버리지 프로파일.
#
#   [within-season back-out]
#   asof_* 는 통산 누적이라 베테랑일수록 현재 시즌 성적이 희석된다(15,000구 중 2,000구 = 12%).
#   그런데 asof_pitcher_n 이 학습 데이터의 행 수와 정확히 일치하므로, 직전 시즌 종료 시점의
#   누적 상태 (n0, r0) 를 학습 데이터에서 계산해 두면 각 행에서 현재 시즌분만 뽑아낼 수 있다.
#
#       현재시즌 투구수 = asof_n − n0
#       현재시즌 성공률 = (asof_n·asof_rate − n0·r0) / (asof_n − n0)
#
#   이 계산은 **그 행 자신의 asof 값 + 학습 데이터로 만든 상수**만 쓴다. 평가 데이터의 다른
#   행을 참조하지 않으므로 규정을 위반하지 않는다.
#
#   [역할/레버리지 프로파일]
#   점수차가 크게 벌어진 상황에 나오는 투수는 추격조·패전처리인 경우가 많아 기량이 낮고,
#   반대로 접전 후반에 나오는 투수는 필승조다. 투수의 과거 등판 상황(li, 점수차, 이닝)을
#   평균 내면 이 역할이 드러나고, '현재 상황이 그 투수의 평소보다 중요한가'도 만들 수 있다.
# ==========================================================================
# (asof 누적 rate 컬럼, 대응하는 누적 카운트 컬럼)
PITCHER_RATES = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
]
BATTER_RATES = [
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
]
# role_dev_* 의 리그 기대값을 어느 단위로 잡을지. [] 는 전 구간 통합 평균(v3 기존 동작).
SF_LEAGUE_KEYS = ["season", "game_type"]

MIN_SEASON_N = 10       # 현재 시즌 표본이 이보다 적으면 원본 rate를 신뢰하지 않음
SHRINK_K = 300.0        # 현재시즌 rate를 통산 쪽으로 수축시키는 강도(시즌 초반 안정화)
ROLE_COLS = ["li", "abs_score_diff", "inning", "num_runners_on"]


def build_carry_state(hist):
    """hist(과거 시즌 전체)의 마지막 시점 누적 상태와 역할 프로파일을 투수/타자별로 만든다."""
    out = {}

    # --- 투수 누적 상태: asof_pitcher_n 이 최대인 행이 그 투수의 마지막 투구 ---
    idx = hist.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    last = hist.loc[idx]
    pit = pd.DataFrame({"n0": last["asof_pitcher_n"].values}, index=last["pitcher_id"].values)
    for col, _ in PITCHER_RATES:
        pit[col] = last[col].values
    pit["mix_n0"] = last["asof_pitcher_pitchmix_n"].values
    for col in ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                "asof_pitcher_offspeed_rate"]:
        pit[col] = last[col].values
    out["pitcher"] = pit

    idx = hist.groupby("batter_id")["asof_batter_n"].idxmax()
    last = hist.loc[idx]
    bat = pd.DataFrame({"n0": last["asof_batter_n"].values}, index=last["batter_id"].values)
    for col, _ in BATTER_RATES:
        bat[col] = last[col].values
    out["batter"] = bat

    # --- 역할 프로파일: 그 투수가 평소 어떤 상황에 나오는가 + 상황별 성적 ---
    h = hist.assign(abs_score_diff=hist["score_diff_pitcher_team"].abs())
    # 리그 기대값은 (시즌 x game_type) 단위로 잡는다. 통합 평균 하나를 쓰면 편차가
    # 실력이 아니라 '언제 어느 리그에서 던졌나'를 담는다 — 1군 제구율이 .565 -> .486
    # 으로 흐른 데다 퓨처스(F)는 2023 에 .71 -> .47 로 무너졌기 때문이다.
    h = h.assign(_lg=h.groupby(SF_LEAGUE_KEYS)["control_success"].transform("mean")
                 if SF_LEAGUE_KEYS else h["control_success"].mean())
    role = h.groupby("pitcher_id").agg(
        role_li=("li", "mean"),
        role_absdiff=("abs_score_diff", "mean"),
        role_inning=("inning", "mean"),
        role_runners=("num_runners_on", "mean"),
        role_n=("control_success", "size"),
    )
    role["role_early"] = h.assign(e=(h["inning"] <= 2).astype(float)) \
        .groupby("pitcher_id")["e"].mean()          # 선발 지표
    role["role_blowout"] = h.assign(b=(h["abs_score_diff"] >= 5).astype(float)) \
        .groupby("pitcher_id")["b"].mean()          # 추격조/패전처리 지표
    # 고/저 레버리지 상황에서의 리그 대비 성적 (표본 적으면 0으로 수축).
    # 기대 성공 수는 그 투구가 속한 (시즌, game_type) 리그 평균의 합으로 센다.
    for tag, m in [("hi", h["li"] >= 1.5), ("lo", h["li"] < 0.5)]:
        sub = h[m].groupby("pitcher_id").agg(s=("control_success", "sum"),
                                             e=("_lg", "sum"),
                                             c=("control_success", "size"))
        role[f"role_dev_{tag}li"] = (sub["s"] - sub["e"]) / (sub["c"] + 300.0)
    out["role"] = role
    return out


def apply_carry_state(df, state):
    """각 행에 현재 시즌 성적과 역할 관련 피처를 붙인다."""
    x = pd.DataFrame(index=df.index)
    pit, bat, role = state["pitcher"], state["batter"], state["role"]
    pid, bid = df["pitcher_id"].values, df["batter_id"].values

    def _within(prefix, table, ids, rate_cols, n_col):
        # 과거 시즌에 등장하지 않은 선수(신인)는 통산 기록 전체가 현재 시즌분이므로 n0=0
        n0 = pd.Series(ids).map(table["n0"]).fillna(0.0).values.astype("float64")
        n1 = df[n_col].values.astype("float64")
        dn = n1 - n0
        ok = np.isfinite(dn) & (dn >= MIN_SEASON_N)
        x[f"{prefix}_season_n"] = np.where(np.isfinite(dn), np.maximum(dn, 0), 0).astype("float32")
        x[f"{prefix}_season_known"] = ok.astype("float32")
        for col, _ in rate_cols:
            r0 = pd.Series(ids).map(table[col]).fillna(0.0).values.astype("float64")
            r1 = df[col].values.astype("float64")
            with np.errstate(invalid="ignore", divide="ignore"):
                w = (n1 * r1 - n0 * r0) / dn
            w = np.where(ok & np.isfinite(w), np.clip(w, 0, 1), np.nan)
            short = col.replace("asof_", "").replace("_rate", "")
            x[f"{prefix}_season_{short}"] = w.astype("float32")
            # 현재 시즌이 통산 대비 얼마나 좋은가 = 폼 변화
            x[f"{prefix}_season_vs_career_{short}"] = (w - r1).astype("float32")
            # 표본이 적은 시즌 초반에는 통산 쪽으로 수축시킨 안정 버전도 함께 제공
            dnc = np.where(np.isfinite(dn), np.maximum(dn, 0), 0)
            shrunk = np.where(np.isfinite(w),
                              (dnc * w + SHRINK_K * r1) / (dnc + SHRINK_K), r1)
            x[f"{prefix}_season_{short}_shrunk"] = shrunk.astype("float32")

    _within("pit", pit, pid, PITCHER_RATES, "asof_pitcher_n")
    _within("bat", bat, bid, BATTER_RATES, "asof_batter_n")

    # 구종 믹스도 동일하게 현재 시즌분 분리
    n0 = pd.Series(pid).map(pit["mix_n0"]).fillna(0.0).values.astype("float64")
    n1 = df["asof_pitcher_pitchmix_n"].values.astype("float64")
    dn = n1 - n0
    ok = np.isfinite(dn) & (dn >= MIN_SEASON_N)
    for col in ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate"]:
        r0 = pd.Series(pid).map(pit[col]).fillna(0.0).values.astype("float64")
        r1 = df[col].values.astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            w = (n1 * r1 - n0 * r0) / dn
        x["pit_season_" + col.split("_")[-2]] = np.where(
            ok & np.isfinite(w), np.clip(w, 0, 1), np.nan).astype("float32")

    # 역할 프로파일 + 현재 상황이 평소 대비 어떤가
    for c in role.columns:
        x[c] = pd.Series(pid).map(role[c]).values.astype("float32")
    x["li_vs_role"] = (df["li"].values - x["role_li"].values).astype("float32")
    x["absdiff_vs_role"] = (df["score_diff_pitcher_team"].abs().values
                            - x["role_absdiff"].values).astype("float32")
    x["inning_vs_role"] = (df["inning"].values - x["role_inning"].values).astype("float32")
    return x


EXTRA_KS = [100.0, 1000.0]      # 기본 300 외에 추가로 제공할 수축 강도


def extra_season_features(df, x):
    """within-season 결과 위에 얹는 파생 피처.

    - 최근 경기 성적을 '통산'이 아니라 '현재 시즌' 기준으로 비교 -> 시즌 내 추세
    - 수축 강도를 여러 개 제공해 모델이 표본 크기에 맞는 신뢰도를 직접 고르게 함
    - 현재 시즌이 커리어에서 차지하는 비중 -> 베테랑/신인 구분
    """
    e = pd.DataFrame(index=df.index)
    season_s = x["pit_season_pitcher_success"].values.astype("float64")
    season_m = x["pit_season_pitcher_middle"].values.astype("float64")
    dn = x["pit_season_n"].values.astype("float64")

    for k in [1, 3, 5]:
        prev = df[f"asof_pitcher_prev{k}_game_success_rate"].values.astype("float64")
        e[f"prev{k}_vs_season"] = (prev - season_s).astype("float32")
    for k in [1, 5]:
        prev = df[f"asof_pitcher_prev{k}_game_middle_rate"].values.astype("float64")
        e[f"prev{k}_mid_vs_season"] = (prev - season_m).astype("float32")

    career_s = df["asof_pitcher_success_rate"].values.astype("float64")
    career_m = df["asof_pitcher_middle_rate"].values.astype("float64")
    for K in EXTRA_KS:
        for tag, sv, cv in [("succ", season_s, career_s), ("mid", season_m, career_m)]:
            sh = np.where(np.isfinite(sv), (dn * sv + K * cv) / (dn + K), cv)
            e[f"pit_season_{tag}_shrunk{int(K)}"] = sh.astype("float32")

    n_all = df["asof_pitcher_n"].values.astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        e["pit_season_frac"] = np.where(n_all > 0, dn / n_all, np.nan).astype("float32")
    e["log_pit_season_n"] = np.log1p(np.maximum(dn, 0)).astype("float32")

    # 현재 시즌 제구력과 상황의 상호작용 (몰릴 위험이 큰 카운트일수록 실력차가 벌어짐)
    b = df["balls_before"].values.astype("float64")
    s = df["strikes_before"].values.astype("float64")
    base = np.where(np.isfinite(season_s), season_s, career_s)
    e["season_succ_x_mustStrike"] = (base * np.clip(b - s, 0, None)).astype("float32")
    e["season_succ_x_twostrike"] = (base * (s == 2)).astype("float32")
    return e


def add_season_form(df, label_df, cols_needed):
    """시즌 s 행에는 s 미만 시즌으로 만든 상태만 적용 (out-of-time)."""
    parts = []
    for s in np.sort(df["season"].unique()):
        hist = label_df[label_df["season"] < s]
        sub = df[df["season"] == s]
        if hist.empty:
            parts.append(pd.DataFrame(np.nan, index=sub.index, columns=cols_needed))
            continue
        parts.append(apply_carry_state(sub, build_carry_state(hist)))
    out = pd.concat(parts).reindex(df.index)
    return out.astype("float32")


# ==========================================================================
# marcel.py
#   Marcel 스타일 투수/타자 제구 능력 추정치.
#
#   세이버메트릭스의 Marcel projection 을 이 문제에 맞춰 옮긴 것:
#     1) 시즌별 성적을 **리그 평균 대비 편차**로 바꾼다 (ERA+ / wRC+ 의 정규화와 같은 발상).
#        리그 제구 성공률이 0.565 -> 0.486 으로 계속 떨어졌기 때문에, 원본
#        asof_pitcher_success_rate 는 '실력'과 '어느 시대에 던졌나'가 섞여 있다.
#     2) 최근 3시즌에 5:4:3 가중치를 준다 (최신 정보를 더 신뢰).
#     3) 표본이 적으면 리그 평균(편차 0) 쪽으로 회귀시킨다 (regression to the mean).
#
#   핵심 제약: 어떤 시즌 s 행의 피처는 반드시 s 미만 시즌 데이터로만 만든다(out-of-time).
#   따라서 평가 시점 이후 정보가 들어가지 않는다.
# ==========================================================================
WEIGHTS = {1: 5.0, 2: 4.0, 3: 3.0}   # t-1, t-2, t-3 시즌 가중치
REG_PITCHES = 1500.0                  # 리그 평균으로 회귀시키는 강도(투구 수 단위)

# mrc_dev 의 리그 기대값 단위. ["season"] 이 v3 기존 동작.
MRC_LEAGUE_KEYS = ["season", "game_type"]


def _season_stats(df, keys):
    """(keys..., season) 별 성공 수/투구 수 및 리그 평균 대비 편차.

    리그 기대값은 (season, game_type) 단위다. 시즌만으로 정규화하면 퓨처스(F)의
    2023 레짐 붕괴(.71 -> .47)가 그대로 편차에 실려, 2022 이전에 F 비중이 높았던
    투수가 실력이 좋은 것처럼 보인다.
    """
    d = df.assign(_exp=df.groupby(MRC_LEAGUE_KEYS)["control_success"].transform("mean"))
    g = d.groupby(keys + ["season"]).agg(
        **{"sum": ("control_success", "sum"),
           "count": ("control_success", "size"),
           "exp": ("_exp", "sum")}).reset_index()
    g["dev_sum"] = g["sum"] - g["exp"]        # 리그 대비 초과 성공 수
    return g


def build_projection(hist, keys, target_season, reg=REG_PITCHES):
    """target_season 미만 데이터(hist)로 target_season 용 Marcel 추정치를 만든다."""
    g = _season_stats(hist, keys)
    g["lag"] = target_season - g["season"]
    g = g[g["lag"].isin(WEIGHTS)]
    if g.empty:
        return pd.DataFrame(columns=keys + ["mrc_dev", "mrc_n"]).set_index(keys)
    g["w"] = g["lag"].map(WEIGHTS)
    g["wn"] = g["w"] * g["count"]
    g["wd"] = g["w"] * g["dev_sum"]
    agg = g.groupby(keys)[["wn", "wd"]].sum()
    agg["mrc_dev"] = agg["wd"] / (agg["wn"] + reg)   # 표본 적으면 0(리그평균)으로 수축
    agg["mrc_n"] = agg["wn"]
    return agg[["mrc_dev", "mrc_n"]]


def add_marcel_features(df, label_df, prefix_keys, out_prefix, reg=REG_PITCHES):
    """df 각 행에, 그 행의 시즌보다 이전 데이터로만 만든 Marcel 피처를 붙인다.

    label_df: control_success 를 가진 전체 학습 데이터(과거 시즌 참조용).
    """
    out_dev = np.full(len(df), np.nan)
    out_n = np.zeros(len(df))
    seasons = np.sort(df["season"].unique())
    for s in seasons:
        hist = label_df[label_df["season"] < s]
        if hist.empty:
            continue
        proj = build_projection(hist, prefix_keys, int(s), reg=reg)
        mask = (df["season"] == s).values
        idx = pd.MultiIndex.from_frame(df.loc[mask, prefix_keys]) if len(prefix_keys) > 1 \
            else pd.Index(df.loc[mask, prefix_keys[0]])
        out_dev[mask] = proj["mrc_dev"].reindex(idx).values
        out_n[mask] = np.nan_to_num(proj["mrc_n"].reindex(idx).values)
    res = pd.DataFrame(index=df.index)
    res[f"{out_prefix}_dev"] = np.nan_to_num(out_dev).astype("float32")
    res[f"{out_prefix}_n"] = np.log1p(out_n).astype("float32")
    res[f"{out_prefix}_missing"] = np.isnan(out_dev).astype("float32")
    return res


def apply_projection(df, proj, out_prefix):
    """미리 만들어 둔 Marcel 추정치 테이블을 각 행에 붙인다 (추론용)."""
    dev = df["pitcher_id"].map(proj["mrc_dev"]).values.astype("float64")
    n = df["pitcher_id"].map(proj["mrc_n"]).values.astype("float64")
    res = pd.DataFrame(index=df.index)
    res[f"{out_prefix}_dev"] = np.nan_to_num(dev).astype("float32")
    res[f"{out_prefix}_n"] = np.log1p(np.nan_to_num(n)).astype("float32")
    res[f"{out_prefix}_missing"] = np.isnan(dev).astype("float32")
    return res


# ==========================================================================
# trackman_feats.py
#   trackman 기반 투수 물리 프로파일.
#
#   trackman 에는 위치(plate location) 정보가 없어 제구를 직접 계산할 수 없다. 대신
#   릴리스 포인트의 반복성(rel_height / rel_side 의 표준편차)이 기계적 일관성을 나타내는
#   고전적 제구 대리지표다. 구속/회전/무브먼트의 산포도 같은 맥락에서 쓴다.
#
#   누출 방지: 시즌 s 행의 피처는 반드시 trackman 의 시즌 < s 데이터로만 만든다.
#   2025 추론용은 2019~2024 전체를 쓴다.
# ==========================================================================
NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
       "extension", "rel_height", "rel_side", "zone_speed"]
STD_OF = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
          "rel_height", "rel_side"]


def build_profile(tm_hist):
    """과거 trackman 로그로 투수별 물리 프로파일을 만든다 (index=pitcher_trackman_id)."""
    g = tm_hist.groupby("pitcher_trackman_id")
    out = pd.DataFrame({"tm_n": g.size()})
    for c in NUM:
        out[f"tm_{c}_mean"] = g[c].mean()
    for c in STD_OF:
        out[f"tm_{c}_std"] = g[c].std()

    # 패스트볼만 따로: 구종 구성에 좌우되지 않는 순수 반복성 지표
    fb = tm_hist[tm_hist["pitch_type_group"] == "fastball"]
    gf = fb.groupby("pitcher_trackman_id")
    out["tm_fb_n"] = gf.size()
    for c in ["rel_speed", "spin_rate", "rel_height", "rel_side", "extension"]:
        out[f"tm_fb_{c}_mean"] = gf[c].mean()
    for c in ["rel_speed", "rel_height", "rel_side"]:
        out[f"tm_fb_{c}_std"] = gf[c].std()

    # 구종 간 릴리스 포인트 차이(작을수록 구종을 숨김 = 일관된 메커니즘)
    byt = tm_hist.groupby(["pitcher_trackman_id", "pitch_type_group"])[
        ["rel_height", "rel_side"]].mean()
    spread = byt.groupby("pitcher_trackman_id").agg(["max", "min"])
    out["tm_rel_height_typespread"] = (spread[("rel_height", "max")]
                                       - spread[("rel_height", "min")])
    out["tm_rel_side_typespread"] = (spread[("rel_side", "max")]
                                     - spread[("rel_side", "min")])
    # 구속 손실 (릴리스 -> 홈플레이트). 회전효율/구질과 관련
    out["tm_speed_drop"] = out["tm_rel_speed_mean"] - out["tm_zone_speed_mean"]
    return out.astype("float32")


def tm_attach(df, profile, pid2tid):
    """각 행에 투수의 물리 프로파일을 붙인다."""
    tid = df["pitcher_id"].map(pid2tid)
    idx = pd.Index(tid.values)
    out = pd.DataFrame(index=df.index)
    for c in profile.columns:
        out[c] = profile[c].reindex(idx).values
    out["tm_missing"] = tid.isna().values.astype("float32")
    out["tm_log_n"] = np.log1p(np.nan_to_num(out["tm_n"].values)).astype("float32")
    return out.astype("float32")


def load_map():
    res = pd.read_csv("work/pitcher_map_raw.csv", index_col="pid")
    # 중복 배정은 표본이 큰 쪽만 남긴다
    res = res.sort_values("n", ascending=False)
    res = res[~res["trackman_id"].duplicated()]
    return res["trackman_id"].to_dict()


# ==========================================================================
# platoon.py
#   투수 x 타자좌우(플래툰) · 투수 x 카운트국면 캐리 테이블.
#
#   season_form.py 와 같은 원칙을 따른다.
#     - 리그 기대값은 (시즌 x game_type) 단위로 잡는다. 통합 평균 하나를 쓰면 편차가
#       실력이 아니라 '언제 어느 리그에서 던졌나'를 담는다.
#     - 시즌 s 행에는 s 미만 시즌으로 만든 테이블만 붙인다 (out-of-time).
#
#   왜 필요한가: 참가자 A/B 분해에서 투수 ID 단독이 653, 투수x타자좌우가 772 였는데
#   피처셋에는 pitcher_id(범주형)와 batter_hand(수치)가 따로 들어갈 뿐 투수별 플래툰
#   스플릿을 명시하는 컬럼이 없다. 과거 시즌 캐리 테이블 단독 검증에서 투수 단독
#   274.7 -> 플래툰 추가 360.5, 투수별 좌우 편차 표준편차 0.0495, 전/후반기 상관 0.634.
# ==========================================================================
TARGET = "control_success"
KS = (100.0, 500.0)


def _dev(hist):
    """리그 (시즌 x game_type) 평균 대비 편차."""
    lg = hist.groupby(["season", "game_type"])[TARGET].transform("mean")
    return hist[TARGET].values - lg.values


def build_platoon_state(hist):
    """과거 시즌 로그에서 투수x좌우 / 투수x카운트국면 편차 테이블을 만든다."""
    h = hist.copy()
    h["__d"] = _dev(h)
    h["__cs"] = np.where(h["strikes_before"].values == 2, "2S",
                         np.where(h["balls_before"].values >= 3, "3B", "ML"))
    ph = h.groupby(["pitcher_id", "batter_hand"])["__d"].agg(["sum", "size"])
    po = h.groupby("pitcher_id")["__d"].agg(["sum", "size"])
    cs = h.groupby(["pitcher_id", "__cs"])["__d"].agg(["sum", "size"])
    return {"hand": ph, "pit": po, "count": cs}


def apply_platoon(df, state):
    """각 행에 해당 투수의 플래툰 / 카운트국면 편차를 붙인다."""
    x = pd.DataFrame(index=df.index)
    ph, po, cs = state["hand"], state["pit"], state["count"]

    hand_idx = pd.MultiIndex.from_arrays([df["pitcher_id"].values,
                                          df["batter_hand"].values])
    s_h = ph["sum"].reindex(hand_idx).values
    n_h = ph["size"].reindex(hand_idx).values
    s_p = po["sum"].reindex(df["pitcher_id"].values).values
    n_p = po["size"].reindex(df["pitcher_id"].values).values

    # 반대손 타자 상대 편차 — 스플릿 폭을 직접 준다
    other = np.where(df["batter_hand"].values == df["batter_hand"].values.max(),
                     df["batter_hand"].values.min(), df["batter_hand"].values.max())
    oth_idx = pd.MultiIndex.from_arrays([df["pitcher_id"].values, other])
    s_o = ph["sum"].reindex(oth_idx).values
    n_o = ph["size"].reindex(oth_idx).values

    n_h0, n_p0, n_o0 = (np.nan_to_num(v) for v in (n_h, n_p, n_o))
    for K in KS:
        k = int(K)
        d_h = np.nan_to_num(s_h) / (n_h0 + K)
        d_p = np.nan_to_num(s_p) / (n_p0 + K)
        d_o = np.nan_to_num(s_o) / (n_o0 + K)
        x[f"plt_dev{k}"] = d_h.astype("float32")
        # 투수 전체 편차를 뺀 '순수 플래툰 성분'
        x[f"plt_resid{k}"] = (d_h - d_p).astype("float32")
        x[f"plt_split{k}"] = (d_h - d_o).astype("float32")

    x["plt_n"] = n_h0.astype("float32")
    x["plt_log_n"] = np.log1p(n_h0).astype("float32")
    x["plt_missing"] = (~np.isfinite(n_h)).astype("float32")

    csv_ = np.where(df["strikes_before"].values == 2, "2S",
                    np.where(df["balls_before"].values >= 3, "3B", "ML"))
    cs_idx = pd.MultiIndex.from_arrays([df["pitcher_id"].values, csv_])
    s_c = cs["sum"].reindex(cs_idx).values
    n_c = cs["size"].reindex(cs_idx).values
    n_c0 = np.nan_to_num(n_c)
    d_p200 = np.nan_to_num(s_p) / (n_p0 + 200.0)
    x["cs_dev200"] = (np.nan_to_num(s_c) / (n_c0 + 200.0)).astype("float32")
    x["cs_resid200"] = (x["cs_dev200"].values - d_p200).astype("float32")
    x["cs_n"] = n_c0.astype("float32")
    return x


def add_platoon_features(df, label_df):
    """시즌 s 행에는 s 미만 시즌으로 만든 상태만 적용 (out-of-time)."""
    parts, cols = [], None
    for s in np.sort(df["season"].unique()):
        hist = label_df[label_df["season"] < s]
        sub = df[df["season"] == s]
        if hist.empty:
            parts.append(pd.DataFrame(np.nan, index=sub.index,
                                      columns=cols if cols else ["plt_dev100"]))
            continue
        out = apply_platoon(sub, build_platoon_state(hist))
        cols = list(out.columns)
        parts.append(out)
    return pd.concat(parts).reindex(df.index)[cols]


# --- 타자 측 플래툰 -------------------------------------------------------
# 타자 ID 단독은 A/B 분해에서 45 로 약하지만, 타자 x 투수좌우는 다르다.
# 캐리 테이블 단독 검증에서 투수 플래툰(+85.8) 위에 추가로 +13.2 를 얹는다.
# 타자별 좌우 편차 표준편차 0.0425, 전/후반기 상관 0.299 (투수 0.634 보다 약하지만 실재).
BKS = (200.0, 800.0)


def build_bplatoon_state(hist):
    h = hist.copy()
    h["__d"] = _dev(h)
    return {"hand": h.groupby(["batter_id", "pitcher_hand"])["__d"].agg(["sum", "size"]),
            "bat": h.groupby("batter_id")["__d"].agg(["sum", "size"])}


def apply_bplatoon(df, state):
    x = pd.DataFrame(index=df.index)
    bh, bo = state["hand"], state["bat"]
    idx = pd.MultiIndex.from_arrays([df["batter_id"].values, df["pitcher_hand"].values])
    hv = df["pitcher_hand"].values
    other = np.where(hv == hv.max(), hv.min(), hv.max())
    oidx = pd.MultiIndex.from_arrays([df["batter_id"].values, other])

    s_h, n_h = bh["sum"].reindex(idx).values, bh["size"].reindex(idx).values
    s_o, n_o = bh["sum"].reindex(oidx).values, bh["size"].reindex(oidx).values
    s_b, n_b = bo["sum"].reindex(df["batter_id"].values).values, \
        bo["size"].reindex(df["batter_id"].values).values
    n_h0, n_o0, n_b0 = (np.nan_to_num(v) for v in (n_h, n_o, n_b))
    for K in BKS:
        k = int(K)
        d_h = np.nan_to_num(s_h) / (n_h0 + K)
        d_o = np.nan_to_num(s_o) / (n_o0 + K)
        d_b = np.nan_to_num(s_b) / (n_b0 + K)
        x[f"bpl_dev{k}"] = d_h.astype("float32")
        x[f"bpl_resid{k}"] = (d_h - d_b).astype("float32")
        x[f"bpl_split{k}"] = (d_h - d_o).astype("float32")
    x["bpl_n"] = n_h0.astype("float32")
    x["bpl_log_n"] = np.log1p(n_h0).astype("float32")
    x["bpl_missing"] = (~np.isfinite(n_h)).astype("float32")
    return x


def add_bplatoon_features(df, label_df):
    parts, cols = [], None
    for s in np.sort(df["season"].unique()):
        hist = label_df[label_df["season"] < s]
        sub = df[df["season"] == s]
        if hist.empty:
            parts.append(pd.DataFrame(np.nan, index=sub.index, columns=cols or ["bpl_dev200"]))
            continue
        out = apply_bplatoon(sub, build_bplatoon_state(hist))
        cols = list(out.columns)
        parts.append(out)
    return pd.concat(parts).reindex(df.index)[cols]


# --- 배포 경로 -----------------------------------------------------------
# 학습과 추론이 같은 정의/같은 컬럼 순서를 쓰도록 여기 한 곳에서만 만든다.
# 테이블은 (합, 개수) 두 열만 저장하고 수축은 추론 시점에 계산한다 — K 를 바꿔도
# 테이블을 다시 만들 필요가 없다.
_TABLES = ("pl_hand", "pl_pit", "pl_count", "bpl_hand", "bpl_bat")


def build_state(hist):
    """과거 시즌 로그 하나로 투수측·타자측 테이블을 모두 만든다."""
    a, b = build_platoon_state(hist), build_bplatoon_state(hist)
    return {"pl_hand": a["hand"], "pl_pit": a["pit"], "pl_count": a["count"],
            "bpl_hand": b["hand"], "bpl_bat": b["bat"]}


def save_state(state, outdir):
    for k in _TABLES:
        state[k].to_csv(f"{outdir}/platoon_{k}.csv")


def load_state(outdir):
    idx = {"pl_hand": ["pitcher_id", "batter_hand"], "pl_pit": ["pitcher_id"],
           "pl_count": ["pitcher_id", "__cs"], "bpl_hand": ["batter_id", "pitcher_hand"],
           "bpl_bat": ["batter_id"]}
    return {k: pd.read_csv(f"{outdir}/platoon_{k}.csv", index_col=idx[k])
            for k in _TABLES}


def plt_attach(df, state, batter_side=True):
    """학습·추론 공용. 컬럼 순서는 matrix_platoon + matrix_bplatoon 과 같다."""
    p = apply_platoon(df, {"hand": state["pl_hand"], "pit": state["pl_pit"],
                           "count": state["pl_count"]})
    if not batter_side:
        return p
    b = apply_bplatoon(df, {"hand": state["bpl_hand"], "bat": state["bpl_bat"]})
    return pd.concat([p, b], axis=1)


# ==========================================================================
# featureset.py
#   학습과 추론이 공유하는 피처 조립기. 순서가 어긋나지 않도록 한 곳에서만 정의한다.
#
#   여기서 만드는 것은 134개뿐이다. 나머지는 호출부(train.py / script.py)에서 이 뒤에
#   같은 순서로 이어붙인다 — 트랙맨 29 (trackman_feats.attach) + 플래툰 12
#   (platoon.attach) = 총 175. 순서의 정본은 meta.json["features"] 다.
# ==========================================================================
def build_all(df, state, proj):
    """base + within-season + 역할 + extra + marcel 134개를 학습과 동일한 순서로 조립.

    트랙맨/플래툰은 여기 포함되지 않는다 (모듈 docstring 참조).
    """
    base = build_features(df).drop(columns=["pitcher_id", "batter_id"])
    sf = apply_carry_state(df, state)
    ex = extra_season_features(df, sf)
    mrc = apply_projection(df, proj, "mrc_pit")
    return pd.concat([base, sf, ex, mrc], axis=1)


# ==========================================================================
# configs.py
#   CatBoost 앙상블 스펙. 검증과 최종 학습이 같은 정의를 공유한다.
#
#   주의: 아래 l2_leaf_reg / rsm 은 v3 시절 값이다. v7 학습은 이걸
#   --l2 1000 --rsm 0.6 으로 덮어쓴다 (train.py 참조). 두 검증연도 모두 +3.5.
#
#   2024 홀드아웃에서 반복 횟수 곡선을 뽑아 각 설정의 정점을 고정했다. 정규화 강도를
#   서로 다르게 준 CatBoost 5종을 균등 평균한다 (자유 가중 최적화 결과가 균등과 동일했음).
#   LightGBM 은 새 피처셋에서 858 로 CatBoost(908) 에 크게 못 미치고 블렌드 가중치가
#   0 으로 수렴해 제외했다.
# ==========================================================================
GROUPS = ["base", "within", "role", "extra", "marcel"]

ENSEMBLE = [
    # (이름, depth, iterations, lr, l2_leaf_reg, rsm)
    ("d7", 7, 750, 0.03, 20.0, 1.0),
    ("d6", 6, 1000, 0.03, 20.0, 1.0),
    ("d7reg", 7, 750, 0.03, 100.0, 0.8),
    ("d8", 8, 500, 0.03, 100.0, 0.8),
    ("slow", 7, 1250, 0.015, 100.0, 0.8),
]

# v2 제출본(리더보드 1052)과 동일한 설정 — 비교 기준
BASELINE_V2 = [("v2", 7, 500, 0.03, 20.0, 1.0)]
BASELINE_V2_GROUPS = ["base", "within", "role"]


def make_model(spec, seed, thread_count=24):
    from catboost import CatBoostClassifier
    _, depth, iters, lr, l2, rsm = spec
    return CatBoostClassifier(iterations=iters, learning_rate=lr, depth=depth,
                              l2_leaf_reg=l2, rsm=rsm, loss_function="Logloss",
                              random_seed=seed, verbose=0, thread_count=thread_count)


# ==========================================================================
# mono.py
#   단조 제약 — 도메인상 방향이 확실한 피처에만 부호를 강제한다.
#
#   시즌 간 일반화를 돕는다. 2024 홀드아웃에서 단독 +9.0.
# ==========================================================================
# (피처, 방향) — 방향이 도메인상 확실한 것만
MONO = {
    "pit_season_pitcher_success": 1,
    "pit_season_pitcher_success_shrunk": 1,
    "asof_pitcher_success_rate": 1,
    "asof_pitcher_success_rate_shrunk": 1,
    "mrc_pit_dev": 1,
    "asof_batter_success_rate": 1,
    "batter_success_shrunk": 1,
    "pit_season_pitcher_middle": -1,
    "asof_pitcher_middle_rate": -1,
    "asof_pitcher_reverse_rate": -1,
    "bat_season_batter_success": 1,
}


# ==========================================================================
# train.py
# ==========================================================================
import argparse
import json
import os
import shutil
import time

from catboost import Pool


OUT = "submit/model"
SRC_DIR = os.path.dirname(os.path.abspath(__file__))   # 번들에 복사할 모듈이 있는 곳
SEEDS = [42, 7, 2024]
TARGET_YEAR = 2025
MRC_REG = 1500.0

# v3 기준점 (리더보드 1073.73 에서 실측 확정)
REF = dict(mean=0.487700, logit_std=0.122990, lam=0.4017, scale=1.0869)
DELTA_ABS = 0.0134812          # r_2024 - r_hat. λ 1단위당 확률 이동량


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v5", help="model_version. 제출 이름 접두사")
    ap.add_argument("--min-season", type=int, default=2020)
    ap.add_argument("--drop-pre-regime-f", action="store_true",
                    help="2022 이하 game_type=F 행을 학습에서 제외")
    ap.add_argument("--halflife", type=float, default=0.0,
                    help="시즌 단위 recency 반감기. 0=균등")
    ap.add_argument("--mono", action="store_true", help="단조 제약 적용")
    ap.add_argument("--trackman", action="store_true",
                    help="트랙맨 물리 프로파일 29개 추가 (과거 시즌 로그로만 생성)")
    ap.add_argument("--legacy-norm", action="store_true",
                    help="리그 정규화를 v3 기존 동작으로. 검증 매트릭스가 이 설정으로 "
                         "만들어졌으므로 홀드아웃 결과를 그대로 옮기려면 켠다")
    ap.add_argument("--platoon", action="store_true",
                    help="투수x타자좌우 플래툰 + 투수x카운트국면 (12개)")
    ap.add_argument("--bplatoon", action="store_true",
                    help="타자x투수좌우 플래툰 (9개). --platoon 과 함께 쓴다")
    ap.add_argument("--l2", type=float, default=None, help="l2_leaf_reg 덮어쓰기")
    ap.add_argument("--rsm", type=float, default=None, help="rsm 덮어쓰기")
    ap.add_argument("--seeds", default="42,7,2024")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.legacy_norm:
        global SF_LEAGUE_KEYS, MRC_LEAGUE_KEYS
        SF_LEAGUE_KEYS = []
        MRC_LEAGUE_KEYS = ["season"]
        print("리그 정규화: v3 기존 동작 (role=전체통합, marcel=시즌만)")

    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    full = pd.read_parquet("work/train.parquet")
    df = full[full.season >= args.min_season]
    n0 = len(df)
    if args.drop_pre_regime_f:
        dead = (df.season <= 2022) & (df.game_type == "F")
        print(f"레짐 이전 F행 제외: {int(dead.sum()):,}행 "
              f"(라벨평균 {df.loc[dead, TARGET_COL].mean():.4f})")
        df = df[~dead]
    df = df.reset_index(drop=True)
    y = df[TARGET_COL].values.astype("float32")
    print(f"학습 {n0:,} -> {len(df):,}행,  라벨평균 {y.mean():.4f}")

    # 시즌 가중 (평균 1 로 정규화해 l2_leaf_reg 의 실효 강도를 보존)
    w = None
    if args.halflife > 0:
        age = df["season"].max() - df["season"].values
        w = np.power(2.0, -age / args.halflife).astype("float64")
        w = w / w.mean()
        print("시즌 가중:", {int(s): round(float(w[df.season.values == s][0]), 4)
                          for s in np.sort(df.season.unique())})

    # ---- 학습 피처: 시즌별 out-of-time ----
    t0 = time.time()
    # game_type 은 marcel 의 (시즌 x 리그) 정규화에 필요하다
    label_cols = ["season", "game_type", "pitcher_id", TARGET_COL]
    plt_cols = ["season", "game_type", "pitcher_id", "batter_id", "pitcher_hand",
                "batter_hand", "balls_before", "strikes_before", TARGET_COL]
    tm = pid2tid = None
    if args.trackman:
        tm, pid2tid = pd.read_parquet("work/trackman.parquet"), load_map()
        print(f"트랙맨 로그 {len(tm):,}행, 투수 대응 {len(pid2tid)}명")
    parts = []
    for s in np.sort(df["season"].unique()):
        hist = full[full.season < s]
        sub = df[df["season"] == s]
        base = build_all(sub, build_carry_state(hist),
                         build_projection(hist[label_cols], ["pitcher_id"],
                                          int(s), reg=MRC_REG))
        if args.trackman:
            # 프로파일도 s 미만 시즌 로그로만 만든다 (out-of-time)
            base = pd.concat([base, tm_attach(sub, build_profile(tm[tm.season < s]),
                                              pid2tid)], axis=1)
        if args.platoon:
            base = pd.concat([base, plt_attach(
                sub, build_state(hist[plt_cols]), args.bplatoon)], axis=1)
        parts.append(base)
    X = pd.concat(parts).reindex(df.index)
    cats = [c for c in CATEGORICAL if c in X.columns]
    feat_names = list(X.columns)
    mono = None
    if args.mono:
        mono = [MONO.get(c, 0) for c in feat_names]
    print(f"피처 {len(feat_names)}개 조립 {time.time() - t0:.0f}s")

    # ---- 추론용 테이블 (2019~2024 전체 기준) ----
    # carry_state 의 n0/r0 는 test 의 asof_* 와 의미가 맞아야 하므로 full 을 그대로 쓴다.
    state = build_carry_state(full)
    state["pitcher"].to_csv(f"{OUT}/carry_pitcher.csv", index_label="pitcher_id")
    state["batter"].to_csv(f"{OUT}/carry_batter.csv", index_label="batter_id")
    state["role"].to_csv(f"{OUT}/carry_role.csv", index_label="pitcher_id")
    build_projection(full[label_cols], ["pitcher_id"], TARGET_YEAR, reg=MRC_REG) \
        .to_csv(f"{OUT}/marcel_pitcher.csv", index_label="pitcher_id")
    if args.platoon:
        save_state(build_state(full[plt_cols]), OUT)
    if args.trackman:
        # 추론용 프로파일은 2019~2024 전체 로그. 2025 트랙맨은 애초에 제공되지 않는다.
        build_profile(tm).to_csv(f"{OUT}/tm_profile.csv", index_label="trackman_id")
        pd.Series(pid2tid, name="trackman_id").to_csv(f"{OUT}/pitcher_map.csv",
                                                      index_label="pitcher_id")

    for c in cats:
        X[c] = X[c].astype("int32").astype("str")
    pool = Pool(X, y, weight=w, cat_features=cats)

    members, preds_last = [], []
    last = df["season"].values == df["season"].max()
    for spec in ENSEMBLE:
        for seed in seeds:
            t = time.time()
            nm, dep, it, lr, l2, rsm = spec
            if args.l2 is not None:
                l2 = args.l2
            if args.rsm is not None:
                rsm = args.rsm
            m = make_model((nm, dep, it, lr, l2, rsm), seed)
            if mono is not None:
                m.set_params(monotone_constraints=mono)
            m.fit(pool)
            name = f"cat_{spec[0]}_{seed}"
            m.save_model(f"{OUT}/{name}.cbm")
            members.append(name)
            preds_last.append(m.predict_proba(X[last])[:, 1])
            print(f"  {name} ({time.time() - t:.0f}s)", flush=True)

    rates = full.groupby("season")[TARGET_COL].mean()
    r_hat = estimate_base_rate(rates.index.values.astype(float), rates.values, TARGET_YEAR)

    # ---- 캘리브레이션: 임시값만 넣고 ref_stats.py 로 이월한다 ----
    # 여기서 재면 안 된다. preds_last 는 시즌별 out-of-time 피처(= 모델이 학습한 그 피처)로
    # 만든 in-sample 예측이라 로짓 분산이 추론 경로보다 60% 가까이 부푼다. 기준값 REF 는
    # 추론 경로로 측정된 값이므로 서로 다른 경로를 비교하면 scale 이 측정 구간 밖으로 튄다.
    #   -> 학습 후 반드시:  python work/ref_stats.py --bundle submit --apply
    p = np.mean(preds_last, axis=0)
    center = float(logit(p).mean())
    lam, scale = REF["lam"], REF["scale"]
    logit_shift = 4.0 * lam * (r_hat - float(rates.iloc[-1]))
    print(f"\n학습경로 참조 std={logit(p).std():.6f} (in-sample, 이월에 쓰지 말 것)"
          f"\n캘리브레이션은 임시값. 다음을 실행하세요:"
          f"\n  python ref_stats.py --bundle submit --apply")

    name = (f"{args.tag}-w100-l{round(lam * 100):03d}-s{round(scale * 100):03d}")
    meta = dict(model_version=args.tag, name=name,
                features=feat_names, cat_features=cats, members=members,
                w_cat=1.0, n_cat=len(members), n_lgb=0,
                logit_shift=round(logit_shift, 6), scale=round(scale, 4),
                center_logit=round(center, 6),
                season_rates={int(k): float(v) for k, v in rates.items()},
                r_hat=r_hat, lambda_=round(lam, 4), target_year=TARGET_YEAR,
                use_season_form=True, use_marcel=True,
                train_rows=len(df), min_season=args.min_season,
                drop_pre_regime_f=args.drop_pre_regime_f, halflife=args.halflife,
                mono=args.mono, trackman=args.trackman, legacy_norm=args.legacy_norm,
                platoon=args.platoon, bplatoon=args.bplatoon,
                l2=args.l2, rsm=args.rsm)
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 제출 번들은 script.py 한 장으로 자족한다 (피처 모듈이 인라인돼 있다)
    shutil.copy(os.path.join(SRC_DIR, "script.py"), "submit/script.py")
    with open("submit/requirements.txt", "w") as f:
        f.write(REQUIREMENTS)
    print(f"\n모델 {len(members)}개 저장 -> {OUT}   제출 이름 {name}")
    print(f"  logit_shift={logit_shift:+.6f}  scale={scale:.4f}  center={center:+.6f}")


# ==========================================================================
# ref_stats (--calibrate).py
#   번들의 참조 예측 통계를 재고, 캘리브레이션 상수를 이월한다.
#
#   λ*/scale* 는 리더보드 실측으로 확정돼 있다 (v3: λ 0.4017, scale 1.0869, LB 1073.73).
#   새 모델은 예측의 평균과 분산이 달라지므로 그만큼만 옮기면 재탐색이 필요 없다.
#
#       λ*_new     = λ*_ref + (m_new - m_ref) / 0.0134812     (최적 평균 이동량은 고정)
#       scale*_new = scale*_ref x (std_ref / std_new)         (최적 '보정 후' 분산은 고정)
#
#   중요: m/std 는 반드시 **추론 경로**로 재야 한다. 학습 경로(시즌별 out-of-time 피처)로
#   재면 모델이 학습한 그 피처라 in-sample 이 되어 분산이 60% 가까이 부풀고, 서로 다른
#   경로로 잰 두 값을 비교하면 이월이 깨진다. 그래서 여기서는 배포용 carry/marcel 테이블을
#   그대로 써서 script.py 와 동일한 경로로 예측한다.
#
#   사용:
#     python ref_stats.py --bundle submit_v3_best          # 기준값 확인
#     python ref_stats.py --bundle submit --apply          # 새 번들에 상수 이월
# ==========================================================================
import argparse
import json
import os
import sys


ROOT = os.getcwd()   # 학습 본체와 동일하게 프로젝트 루트에서 실행한다
N = 200_000
REF_YEAR = 2024
SEED = 0

# v3 기준점 — 리더보드 1073.73 에서 확정 (동일 스크립트, 동일 표본으로 측정)




def calibrate(argv=None):
    ap = argparse.ArgumentParser(prog="--calibrate")
    ap.add_argument("--bundle", required=True, help="submit / submit_v3_best 등")
    ap.add_argument("--apply", action="store_true",
                    help="이월된 λ/scale/center 를 번들 meta.json 에 기록")
    args = ap.parse_args(argv)

    bundle = os.path.join(ROOT, args.bundle)

    md = f"{bundle}/model"
    meta = json.load(open(f"{md}/meta.json"))

    df = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig")
    df = df[df.season == REF_YEAR].reset_index(drop=True)
    df = df.sample(N, random_state=SEED).reset_index(drop=True)

    proj = pd.read_csv(f"{md}/marcel_pitcher.csv", index_col="pitcher_id")
    state = {
        "pitcher": pd.read_csv(f"{md}/carry_pitcher.csv", index_col="pitcher_id"),
        "batter": pd.read_csv(f"{md}/carry_batter.csv", index_col="batter_id"),
        "role": pd.read_csv(f"{md}/carry_role.csv", index_col="pitcher_id"),
    }
    X = build_all(df, state, proj)
    tmp = f"{md}/tm_profile.csv"
    if os.path.exists(tmp):
        # script.py 와 동일한 추론 경로. 빠뜨리면 트랙맨 29개가 NaN 으로 채워져
        # 참조 통계(평균/분산)가 실제 제출본과 달라진다.
        prof = pd.read_csv(tmp, index_col="trackman_id")
        pmap = pd.read_csv(f"{md}/pitcher_map.csv",
                           index_col="pitcher_id")["trackman_id"].to_dict()
        X = pd.concat([X, tm_attach(df, prof, pmap)], axis=1)
    if os.path.exists(f"{md}/platoon_pl_hand.csv"):
        # 트랙맨과 같은 이유로 반드시 붙여야 한다. 빠뜨리면 플래툰 12개가 NaN 이 되어
        # 참조 통계가 실제 제출본과 전혀 다른 값이 된다 (실측: 평균 0.493 -> 0.456).
        #
        # 단, 배포 테이블(2019~2024 전체)을 참조연도 2024 행에 그대로 쓰면 그 해 라벨이
        # 편차에 들어가 in-sample 이 된다 — 실측으로 plt_missing 이 18.5% -> 0%, dev std
        # 가 1.25배로 부풀었다. 부푼 std 로 나눈 scale 이월은 과소 산출된다(0.962).
        # 2025 테스트가 겪을 조건은 '평가연도가 테이블에 없는' 쪽이므로, 참조 측정에서는
        # 테이블을 참조연도 미만으로 다시 만든다.
        raw = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig")
        st = build_state(raw[raw.season < REF_YEAR])
        X = pd.concat([X, plt_attach(df, st, meta.get("bplatoon", False))], axis=1)
    n_missing = [c for c in meta["features"] if c not in X.columns]
    if n_missing:
        print(f"  경고: 추론경로에서 만들어지지 않은 피처 {len(n_missing)}개 "
              f"-> NaN 으로 채움  {n_missing[:5]}")
    for c in meta["features"]:
        if c not in X.columns:
            X[c] = np.nan
    X = X[meta["features"]]
    for c in meta["cat_features"]:
        X[c] = X[c].astype("float64").round().astype("int64").astype("str")

    from catboost import CatBoostClassifier
    acc = np.zeros(len(X))
    for name in meta["members"]:
        m = CatBoostClassifier()
        m.load_model(f"{md}/{name}.cbm")
        acc += m.predict_proba(X)[:, 1]
    p = acc / len(meta["members"])

    lo = logit(p)
    m_new, std_new, center = float(p.mean()), float(lo.std()), float(lo.mean())
    print(f"[{args.bundle}]  {meta.get('name', '?')}   n={len(X):,}")
    print(f"  추론경로 참조통계   mean={m_new:.6f}  logit_std={std_new:.6f}  center={center:+.6f}")

    lam = REF["lam"] + (m_new - REF["mean"]) / DELTA_ABS
    scale = REF["scale"] * REF["logit_std"] / std_new
    print(f"  기준 대비          Δmean={m_new - REF['mean']:+.6f}"
          f"   std비={std_new / REF['logit_std']:.4f}")
    print(f"  이월 상수          λ {REF['lam']:.4f} -> {lam:.4f}"
          f"   scale {REF['scale']:.4f} -> {scale:.4f}")

    if not args.apply:
        return

    last = max(int(k) for k in meta["season_rates"])
    logit_shift = 4.0 * lam * (meta["r_hat"] - meta["season_rates"][str(last)])
    mv = meta.get("model_version", "v5")
    meta.update(lambda_=round(lam, 4), scale=round(scale, 4),
                center_logit=round(center, 6), logit_shift=round(logit_shift, 6),
                ref_mean=round(m_new, 6), ref_logit_std=round(std_new, 6),
                name=f"{mv}-w100-l{round(lam * 100):03d}-s{round(scale * 100):03d}")
    with open(f"{md}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> meta.json 갱신.  제출 이름 {meta['name']}"
          f"  logit_shift={logit_shift:+.6f}")


# ==========================================================================
# 진입점
#   python train_v7.py --tag v7 --trackman --mono --legacy-norm --halflife 2 \
#       --platoon --l2 1000 --rsm 0.6 --seeds 42,7,2024
#   python train_v7.py --calibrate --bundle submit --apply
# ==========================================================================
if __name__ == "__main__":
    import sys as _sys
    if "--calibrate" in _sys.argv:
        _a = [x for x in _sys.argv[1:] if x != "--calibrate"]
        calibrate(_a)
    else:
        main()

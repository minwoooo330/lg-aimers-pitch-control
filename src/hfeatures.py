# -*- coding: utf-8 -*-
"""휴리스틱 파생변수 2차 배치 — features.py(44개)에 없는 야구 휴리스틱.

설계 원칙: 개별 문턱으로 거르지 않는다. domain43의 44개 중 내 단독 문턱
(clean fold 상관 0.0105)을 통과하는 것은 1개(2.3%)뿐인데 그룹으로는
리더보드 +18.81점을 냈다. 즉 약한 신호도 조합으로 값을 낸다.

각 피처는 야구 휴리스틱에서 나오며, 이미 있는 것과 중복되지 않게 골랐다.
리그 기대 실패유형 테이블(J군)은 내 exp45/46 측정 결과를 그대로 피처화한 것으로,
카운트가 실패 유형 구성을 거의 결정한다는 발견을 모델에 직접 전달한다.
"""
import numpy as np
import pandas as pd

# exp46 실측: 1군 카운트별 실패 유형 비율 (한가운데 / 크게벗어남 / 의도반대)
CNT_MID = {"3-0": .1815, "3-1": .1802, "3-2": .1761, "2-0": .1722, "2-1": .1660,
           "1-0": .1630, "1-1": .1551, "0-0": .1541, "2-2": .1520, "0-1": .1407,
           "1-2": .1316, "0-2": .1180}
CNT_BALL = {"3-0": .0830, "3-1": .0832, "3-2": .0968, "2-0": .0889, "2-1": .0979,
            "1-0": .1072, "1-1": .1253, "0-0": .1205, "2-2": .1386, "0-1": .1545,
            "1-2": .1854, "0-2": .2330}
CNT_REV = {"3-0": .2415, "3-1": .2454, "3-2": .2398, "2-0": .2325, "2-1": .2254,
           "1-0": .2154, "1-1": .2006, "0-0": .2090, "2-2": .1980, "0-1": .1783,
           "1-2": .1661, "0-2": .1394}


def _logit(p, lo=1e-4):
    p = np.clip(p, lo, 1 - lo)
    return np.log(p / (1 - p))


def add_hfeatures(d):
    f = pd.DataFrame(index=d.index)
    b = d["balls_before"].astype(float)
    s = d["strikes_before"].astype(float)
    o = d["outs_before"].astype(float)
    inn = d["inning"].astype(float)
    cnt = d["balls_before"].astype(str) + "-" + d["strikes_before"].astype(str)
    n_p = d["asof_pitcher_n"].fillna(0).astype(float)
    n_b = d["asof_batter_n"].fillna(0).astype(float)
    sr = d["asof_pitcher_success_rate"]
    mr = d["asof_pitcher_middle_rate"]
    rr = d["asof_pitcher_reverse_rate"]
    br_ = d["asof_pitcher_ball_rate"]
    st = d["asof_pitcher_strike_rate"]
    bs = d["asof_batter_success_rate"]
    bm = d["asof_batter_middle_rate"]
    fa = d["asof_pitcher_fastball_rate"]
    bk = d["asof_pitcher_breaking_rate"]
    of = d["asof_pitcher_offspeed_rate"]
    LM = 0.52

    # --- A. 카운트 세분 ---
    f["h_balls_left"] = 3.0 - b                      # 볼넷까지 남은 개수
    f["h_strikes_left"] = 2.0 - s                    # 삼진까지 남은 개수
    f["h_pa_progress"] = (b + s) / 5.0               # 타석 진행도
    f["h_cnt_product"] = b * s
    f["h_cnt_ratio"] = (b + 1) / (s + 1)
    f["h_two_way_edge"] = (3.0 - b) - (2.0 - s)      # 볼넷/삼진 임박 비대칭

    # --- B. 리그 기대 실패유형 (exp46 측정 테이블 조인) ---
    f["h_cnt_exp_mid"] = cnt.map(CNT_MID).astype(float)
    f["h_cnt_exp_ball"] = cnt.map(CNT_BALL).astype(float)
    f["h_cnt_exp_rev"] = cnt.map(CNT_REV).astype(float)
    f["h_cnt_exp_fail"] = f.h_cnt_exp_mid + f.h_cnt_exp_ball + f.h_cnt_exp_rev
    f["h_cnt_mid_vs_ball"] = f.h_cnt_exp_mid - f.h_cnt_exp_ball
    # 투수 성향 x 카운트 기대: 한가운데 잘 내는 투수가 한가운데 나기 쉬운 카운트에
    f["h_mid_align"] = mr.fillna(.15) * f.h_cnt_exp_mid
    f["h_ball_align"] = br_.fillna(.35) * f.h_cnt_exp_ball
    f["h_rev_align"] = rr.fillna(.20) * f.h_cnt_exp_rev

    # --- C. 실패 유형 프로필 ---
    tot = (mr.fillna(.15) + rr.fillna(.20) + br_.fillna(.35)).replace(0, np.nan)
    f["h_fail_mid_frac"] = mr.fillna(.15) / tot
    f["h_fail_rev_frac"] = rr.fillna(.20) / tot
    f["h_fail_ball_frac"] = br_.fillna(.35) / tot
    p1, p2, p3 = f.h_fail_mid_frac, f.h_fail_rev_frac, f.h_fail_ball_frac
    f["h_fail_entropy"] = -(p1 * np.log(p1 + 1e-9) + p2 * np.log(p2 + 1e-9) + p3 * np.log(p3 + 1e-9))
    f["h_rev_over_mid"] = np.log((rr.fillna(.20) + .01) / (mr.fillna(.15) + .01))
    f["h_succ_logit"] = _logit(sr.fillna(LM))
    f["h_mid_logit"] = _logit(mr.fillna(.15))
    f["h_rev_logit"] = _logit(rr.fillna(.20))
    f["h_strike_over_ball"] = np.log((st.fillna(.45) + .01) / (br_.fillna(.35) + .01))

    # --- D. 표본 신뢰도 ---
    f["h_se_succ"] = np.sqrt(sr.fillna(LM) * (1 - sr.fillna(LM)) / (n_p + 1))
    f["h_succ_lcb"] = sr.fillna(LM) - 1.96 * f.h_se_succ      # 신뢰구간 하한
    f["h_succ_ucb"] = sr.fillna(LM) + 1.96 * f.h_se_succ
    f["h_sqrt_np"] = np.sqrt(n_p)
    f["h_rate_x_logn"] = sr.fillna(LM) * np.log1p(n_p)
    f["h_np_per_year"] = np.log1p(n_p / (d["season"].astype(float) - 2018.0))

    # --- E. 폼 동역학 ---
    q1 = d["asof_pitcher_prev1_game_success_rate"]
    q3 = d["asof_pitcher_prev3_game_success_rate"]
    q5 = d["asof_pitcher_prev5_game_success_rate"]
    f["h_form_accel"] = (q1 - q3) - (q3 - q5)                 # 폼 가속도
    f["h_form_step13"] = q1 - q3
    f["h_form_step35"] = q3 - q5
    f["h_form_vol"] = (q1 - q3).abs() + (q3 - q5).abs()       # 폼 변동성
    m1 = d["asof_pitcher_prev1_game_middle_rate"]
    m5 = d["asof_pitcher_prev5_game_middle_rate"]
    f["h_midform_vs_career"] = m5 - mr
    f["h_midform_step"] = m1 - m5

    # --- F. 매치업 ---
    f["h_skill_gap"] = sr.fillna(LM) - bs.fillna(LM)          # 투수-타자 실력차
    f["h_mid_gap"] = mr.fillna(.15) - bm.fillna(.15)
    f["h_batter_conf"] = np.log1p(n_b)
    f["h_skill_gap_x_conf"] = f.h_skill_gap * np.log1p(np.minimum(n_p, n_b))

    # --- G. 상황 조합 ---
    runners = d["num_runners_on"].astype(float)
    r3 = d["runner_on_3b"].astype(float)
    sd = d["score_diff_pitcher_team"].astype(float)
    f["h_3ball_risp"] = (b == 3).astype(float) * (runners >= 2).astype(float)
    f["h_2strike_loaded"] = (s == 2).astype(float) * (runners == 3).astype(float)
    f["h_late_close"] = (inn >= 7).astype(float) * (sd.abs() <= 2).astype(float)
    f["h_inn_x_sd"] = inn * sd                                 # 교체 임박 신호
    f["h_dp_chance"] = (o < 2).astype(float) * (d["runner_on_1b"].astype(float))
    f["h_wp_risk"] = r3 * (s == 2).astype(float)               # 3루주자 + 유인구 = 폭투 위험
    f["h_outs_x_runners"] = o * runners

    # --- H. 승부 상황 ---
    we = d["home_win_expectancy"].astype(float)
    f["h_we_dist"] = (we - 50.0).abs()                         # 승부 갈림 정도
    f["h_we_x_li"] = f.h_we_dist * d["li"].astype(float)
    f["h_li_sq"] = d["li"].astype(float) ** 2

    # --- I. 구종 편향 ---
    f["h_br_minus_of"] = bk.fillna(1 / 3) - of.fillna(1 / 3)
    f["h_mix_max"] = pd.concat([fa, bk, of], axis=1).max(axis=1)
    f["h_mix_diversity"] = 1.0 - f.h_mix_max
    f["h_fb_over_br"] = np.log((fa.fillna(1 / 3) + .01) / (bk.fillna(1 / 3) + .01))

    # --- J. 시즌/일정 ---
    gm = d["game_month"].astype(float)
    f["h_season_prog"] = (gm - 3.0) / 7.0
    f["h_early_season"] = (gm <= 4).astype(float)
    f["h_monday"] = (d["game_dayofweek"].astype(float) == 0).astype(float)  # 편성 이상일
    return f.astype(np.float32)

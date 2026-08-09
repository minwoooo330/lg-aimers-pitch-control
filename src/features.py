# -*- coding: utf-8 -*-
"""야구 도메인 피처. 학습/추론 양쪽에서 동일하게 사용한다.
모두 '투구 직전'에 알 수 있는 정보만 쓴다 (미래 정보 누수 없음)."""
import numpy as np
import pandas as pd


def add_features(d):
    f = pd.DataFrame(index=d.index)

    b, s = d["balls_before"], d["strikes_before"]
    # --- 카운트 ---
    f["count_id"] = b * 3 + s
    f["count_diff"] = b - s
    f["is_2strike"] = (s == 2).astype(np.int8)
    f["is_3ball"] = (b == 3).astype(np.int8)
    f["is_full"] = ((b == 3) & (s == 2)).astype(np.int8)
    f["behind"] = (b > s).astype(np.int8)
    f["ahead"] = (s > b).astype(np.int8)
    f["must_strike"] = (b - s >= 2).astype(np.int8)

    # --- 상황/압박 ---
    f["late"] = (d["inning"] >= 7).astype(np.int8)
    f["inning_c"] = d["inning"].clip(1, 12)
    f["abs_diff"] = d["score_diff_pitcher_team"].abs()
    f["close"] = (f["abs_diff"] <= 1).astype(np.int8)
    f["blowout"] = (f["abs_diff"] >= 6).astype(np.int8)
    f["losing"] = (d["score_diff_pitcher_team"] < 0).astype(np.int8)
    f["li_log"] = np.log1p(d["li"].clip(0, 12))
    f["high_li"] = (d["li"] >= 1.5).astype(np.int8)
    f["scoring_pos"] = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype(np.int8)
    f["loaded"] = ((d["runner_on_1b"] == 1) & (d["runner_on_2b"] == 1)
                   & (d["runner_on_3b"] == 1)).astype(np.int8)
    f["risk"] = f["scoring_pos"] * f["close"]
    f["press"] = f["li_log"] * f["must_strike"]

    # --- 표본 신뢰도 ---
    n_p = d["asof_pitcher_n"].fillna(0)
    n_b = d["asof_batter_n"].fillna(0)
    f["log_np"] = np.log1p(n_p)
    f["log_nb"] = np.log1p(n_b)
    f["cold_p"] = (n_p < 50).astype(np.int8)
    f["rookie"] = (n_p < 300).astype(np.int8)

    # --- shrinkage: 표본 적으면 리그평균 쪽으로 ---
    LM, K = 0.52, 300.0
    sr = d["asof_pitcher_success_rate"]
    f["p_succ_shrunk"] = (n_p * sr.fillna(LM) + K * LM) / (n_p + K)
    mr = d["asof_pitcher_middle_rate"]
    f["p_mid_shrunk"] = (n_p * mr.fillna(0.18) + K * 0.18) / (n_p + K)

    # --- 투수 유형: 몰리는 형 vs 엇나가는 형 ---
    rev = d["asof_pitcher_reverse_rate"]
    mid = d["asof_pitcher_middle_rate"]
    bal = d["asof_pitcher_ball_rate"]
    tot_fail = (rev + mid).replace(0, np.nan)
    f["mid_share"] = mid / tot_fail
    f["rev_share"] = rev / tot_fail
    f["mid_vs_ball"] = mid - bal
    f["strike_minus_ball"] = d["asof_pitcher_strike_rate"] - bal

    # --- 최근 폼 추세 ---
    p1 = d["asof_pitcher_prev1_game_success_rate"]
    p3 = d["asof_pitcher_prev3_game_success_rate"]
    p5 = d["asof_pitcher_prev5_game_success_rate"]
    f["form_1_5"] = p1 - p5
    f["form_3_5"] = p3 - p5
    f["form_vs_career"] = p3 - sr
    m1 = d["asof_pitcher_prev1_game_middle_rate"]
    m5 = d["asof_pitcher_prev5_game_middle_rate"]
    f["midform_1_5"] = m1 - m5
    f["form_missing"] = p1.isna().astype(np.int8)

    # --- 매치업 ---
    f["match_succ"] = sr - d["asof_batter_success_rate"]
    f["match_mid"] = mid - d["asof_batter_middle_rate"]
    f["same_hand"] = (d["pitcher_hand"].astype(str)
                      == d["batter_hand"].astype(str)).astype(np.int8)

    # --- 구종 다양성 ---
    fa = d["asof_pitcher_fastball_rate"].fillna(1 / 3).clip(1e-6, 1)
    br = d["asof_pitcher_breaking_rate"].fillna(1 / 3).clip(1e-6, 1)
    of = d["asof_pitcher_offspeed_rate"].fillna(1 / 3).clip(1e-6, 1)
    ssum = fa + br + of
    fa, br, of = fa / ssum, br / ssum, of / ssum
    f["mix_entropy"] = -(fa * np.log(fa) + br * np.log(br) + of * np.log(of))
    f["fb_heavy"] = (fa > 0.6).astype(np.int8)

    # --- 상호작용 ---
    f["shrunk_x_press"] = f["p_succ_shrunk"] * f["must_strike"]
    f["shrunk_x_li"] = f["p_succ_shrunk"] * f["li_log"]
    f["mid_x_behind"] = f["p_mid_shrunk"] * f["behind"]

    return f.astype(np.float32)

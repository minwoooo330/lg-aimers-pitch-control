# -*- coding: utf-8 -*-
"""
실험 02: 점수 올리기
  2019~2023 학습 / 2024 검증 고정. 항상 '최적 밀기' 적용 후 점수 비교.
  A) 원본 그대로 (기준선)
  B) + 야구 도메인 피처
  C) B + 최근 시즌 가중
  D) C + 모델 키우기
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
D = "data/"
ID, TGT = "row_id", "control_success"
CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]

print("로드...")
df = pd.read_csv(D + "train.csv", encoding="utf-8-sig")
print(f"  {len(df):,}행 ({time.time()-t0:.0f}s)")


# ================= 피처 엔지니어링 =================
def add_features(d):
    """모두 '투구 직전'에 알 수 있는 정보만 사용 (누수 없음)"""
    f = pd.DataFrame(index=d.index)

    b, s = d["balls_before"], d["strikes_before"]
    # --- 카운트 ---
    f["count_id"] = b * 3 + s                    # 0~11 카운트 상태
    f["count_diff"] = b - s                      # +면 투수 불리
    f["is_2strike"] = (s == 2).astype(np.int8)
    f["is_3ball"] = (b == 3).astype(np.int8)
    f["is_full"] = ((b == 3) & (s == 2)).astype(np.int8)
    f["behind"] = (b > s).astype(np.int8)        # 투수 불리 카운트
    f["ahead"] = (s > b).astype(np.int8)         # 투수 유리 카운트
    f["must_strike"] = (b - s >= 2).astype(np.int8)  # 스트라이크 넣어야 하는 압박

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
    f["risk"] = f["scoring_pos"] * f["close"]     # 득점권 + 접전
    f["press"] = f["li_log"] * f["must_strike"]   # 압박 x 스트라이크 강요

    # --- 표본 신뢰도 (적은 표본의 비율은 못 믿는다) ---
    n_p = d["asof_pitcher_n"].fillna(0)
    n_b = d["asof_batter_n"].fillna(0)
    f["log_np"] = np.log1p(n_p)
    f["log_nb"] = np.log1p(n_b)
    f["cold_p"] = (n_p < 50).astype(np.int8)      # 데뷔 초반 투수
    f["rookie"] = (n_p < 300).astype(np.int8)

    # --- 리그 평균 쪽으로 당기기(shrinkage): 표본 적으면 평균에 가깝게 ---
    LM = 0.52  # 리그 평균 제구 성공률(대략)
    K = 300.0
    sr = d["asof_pitcher_success_rate"]
    f["p_succ_shrunk"] = (n_p * sr.fillna(LM) + K * LM) / (n_p + K)
    mr = d["asof_pitcher_middle_rate"]
    f["p_mid_shrunk"] = (n_p * mr.fillna(0.18) + K * 0.18) / (n_p + K)

    # --- 투수 유형: 실패할 때 '몰리는 형'인가 '빠지는 형'인가 ---
    rev = d["asof_pitcher_reverse_rate"]
    mid = d["asof_pitcher_middle_rate"]
    bal = d["asof_pitcher_ball_rate"]
    tot_fail = (rev + mid).replace(0, np.nan)
    f["mid_share"] = mid / tot_fail               # 실패 중 '한가운데' 비중
    f["rev_share"] = rev / tot_fail               # 실패 중 '의도 반대' 비중
    f["mid_vs_ball"] = mid - bal
    f["strike_minus_ball"] = d["asof_pitcher_strike_rate"] - bal

    # --- 최근 폼 추세 (좋아지는 중? 나빠지는 중?) ---
    p1 = d["asof_pitcher_prev1_game_success_rate"]
    p3 = d["asof_pitcher_prev3_game_success_rate"]
    p5 = d["asof_pitcher_prev5_game_success_rate"]
    f["form_1_5"] = p1 - p5                       # 최근 1경기 vs 5경기
    f["form_3_5"] = p3 - p5
    f["form_vs_career"] = p3 - sr                 # 최근 3경기 vs 통산
    m1 = d["asof_pitcher_prev1_game_middle_rate"]
    m5 = d["asof_pitcher_prev5_game_middle_rate"]
    f["midform_1_5"] = m1 - m5
    f["form_missing"] = p1.isna().astype(np.int8)  # 최근 등판 이력 없음

    # --- 매치업: 투수 vs 타자 ---
    f["match_succ"] = sr - d["asof_batter_success_rate"]
    f["match_mid"] = mid - d["asof_batter_middle_rate"]
    f["same_hand"] = (d["pitcher_hand"].astype(str)
                      == d["batter_hand"].astype(str)).astype(np.int8)

    # --- 구종 레퍼토리 다양성(entropy) ---
    fa = d["asof_pitcher_fastball_rate"].fillna(1 / 3).clip(1e-6, 1)
    br = d["asof_pitcher_breaking_rate"].fillna(1 / 3).clip(1e-6, 1)
    of = d["asof_pitcher_offspeed_rate"].fillna(1 / 3).clip(1e-6, 1)
    ssum = fa + br + of
    fa, br, of = fa / ssum, br / ssum, of / ssum
    f["mix_entropy"] = -(fa * np.log(fa) + br * np.log(br) + of * np.log(of))
    f["fb_heavy"] = (fa > 0.6).astype(np.int8)

    # --- 상호작용: 압박 x 투수 실력 ---
    f["shrunk_x_press"] = f["p_succ_shrunk"] * f["must_strike"]
    f["shrunk_x_li"] = f["p_succ_shrunk"] * f["li_log"]
    f["mid_x_behind"] = f["p_mid_shrunk"] * f["behind"]

    return f.astype(np.float32)


base_cols = [c for c in df.columns if c not in (ID, TGT)]
Xb = df[base_cols].copy()
for c in CAT:
    Xb[c] = Xb[c].astype(str).astype("category").cat.codes.astype(np.int32)

print("피처 생성...")
Xn = add_features(df)
print(f"  원본 {Xb.shape[1]}개 + 신규 {Xn.shape[1]}개 = {Xb.shape[1]+Xn.shape[1]}개")

y = df[TGT].to_numpy(np.int8)
season = df["season"].to_numpy()
tr_m, va_m = season < 2024, season == 2024
yva = y[va_m]
r_va = yva.mean()
BASE = r_va * (1 - r_va)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def best_shift_score(p):
    """밀기를 최적으로 맞췄을 때의 점수(상한). 모델 자체 성능 비교용."""
    best = -1
    for tgt in np.arange(0.455, 0.520, 0.0025):
        lo, hi = -5.0, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if sig(logit(p) + mid).mean() < tgt:
                lo = mid
            else:
                hi = mid
        ps = sig(logit(p) + (lo + hi) / 2)
        s = max(0.0, 100000 * (1 - np.mean((ps - yva) ** 2) / BASE))
        best = max(best, s)
    raw = max(0.0, 100000 * (1 - np.mean((p - yva) ** 2) / BASE))
    return raw, best


def run(name, X, params, weight=None):
    t = time.time()
    cm = [c in CAT for c in X.columns]
    m = HistGradientBoostingClassifier(categorical_features=cm,
                                       random_state=42, **params)
    sw = None if weight is None else weight[tr_m]
    m.fit(X[tr_m], y[tr_m], sample_weight=sw)
    p = m.predict_proba(X[va_m])[:, 1]
    raw, best = best_shift_score(p)
    print(f"  {name:34s} 밀기전 {raw:7.1f} | 밀기후 {best:7.1f}  "
          f"({m.n_iter_}트리, {time.time()-t:.0f}s)")
    return best


P0 = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
          min_samples_leaf=200, l2_regularization=1.0,
          early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)

print("\n" + "=" * 78)
print("실험 결과  (점수는 '최적 밀기 적용 후' 기준)")
print("=" * 78)

res = {}
res["A"] = run("A. 원본 47개 (현재 제출본)", Xb, P0)

XF = pd.concat([Xb, Xn], axis=1)
res["B"] = run("B. A + 도메인 피처", XF, P0)

# 최근 시즌 가중: 2019=1.0 -> 2023=2.0
w = np.power(1.19, (season - 2019).astype(float))
res["C"] = run("C. B + 최근시즌 가중", XF, P0, weight=w)

P1 = dict(P0, max_iter=1200, learning_rate=0.03, max_leaf_nodes=63,
          min_samples_leaf=300, l2_regularization=3.0, n_iter_no_change=50)
res["D"] = run("D. C + 모델 크게/느리게", XF, P1, weight=w)
res["E"] = run("E. B + 모델 크게 (가중 없음)", XF, P1)

print("\n" + "=" * 78)
bk = max(res, key=res.get)
print(f"최고: {bk} = {res[bk]:.1f}점   (A 대비 {res[bk]-res['A']:+.1f})")
for k in sorted(res, key=res.get, reverse=True):
    print(f"  {k}: {res[k]:8.1f}   (A 대비 {res[k]-res['A']:+7.1f})")
print(f"\n총 {time.time()-t0:.0f}s")

# -*- coding: utf-8 -*-
"""실험 77: 휴리스틱 배치 군별 절제 — 어느 군이 2022를 깎는가.

exp76에서 50개 배치가 2024 +2.99 / 2022 -3.32e-5로 혼재했다.
군을 4개로 묶어 각각 단독 추가해 기여를 분리한다. clean fold만(2022/2024).

G1 카운트세분 + 리그기대실패유형 (가장 신규, exp46 측정 테이블)
G2 실패프로필 + 표본신뢰도
G3 폼동역학 + 매치업
G4 상황조합 + 승부 + 구종 + 시즌
"""
from pathlib import Path
import sys, time, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features
from hfeatures import add_hfeatures

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ID, TARGET = "row_id", "control_success"
CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
PRM = dict(max_iter=200, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=200,
           l2_regularization=1., early_stopping=False, random_state=42)
G1 = ["h_balls_left","h_strikes_left","h_pa_progress","h_cnt_product","h_cnt_ratio",
      "h_two_way_edge","h_cnt_exp_mid","h_cnt_exp_ball","h_cnt_exp_rev","h_cnt_exp_fail",
      "h_cnt_mid_vs_ball","h_mid_align","h_ball_align","h_rev_align"]
G2 = ["h_fail_mid_frac","h_fail_rev_frac","h_fail_ball_frac","h_fail_entropy","h_rev_over_mid",
      "h_succ_logit","h_mid_logit","h_rev_logit","h_strike_over_ball","h_se_succ",
      "h_succ_lcb","h_succ_ucb","h_sqrt_np","h_rate_x_logn","h_np_per_year"]
G3 = ["h_form_accel","h_form_step13","h_form_step35","h_form_vol","h_midform_vs_career",
      "h_midform_step","h_skill_gap","h_mid_gap","h_batter_conf","h_skill_gap_x_conf"]
G4 = ["h_3ball_risp","h_2strike_loaded","h_late_close","h_inn_x_sd","h_dp_chance","h_wp_risk",
      "h_outs_x_runners","h_we_dist","h_we_x_li","h_li_sq","h_br_minus_of","h_mix_max",
      "h_mix_diversity","h_fb_over_br","h_season_prog","h_early_season","h_monday"]
VAR = [("base", None), ("G1_카운트기대", G1), ("G2_실패프로필", G2),
       ("G3_폼매치업", G3), ("G4_상황구종", G4)]


def enc(tr, va, cols, sub):
    a, b = tr[cols].copy(), va[cols].copy()
    for c in CAT:
        v = sorted(tr[c].dropna().astype(str).unique()); m = {x: i for i, x in enumerate(v)}
        a[c] = tr[c].astype(str).map(m).fillna(-1).astype(np.int16)
        b[c] = va[c].astype(str).map(m).fillna(-1).astype(np.int16)
    pa = [a.reset_index(drop=True), add_features(tr).reset_index(drop=True)]
    pb = [b.reset_index(drop=True), add_features(va).reset_index(drop=True)]
    if sub:
        pa.append(add_hfeatures(tr)[sub].reset_index(drop=True))
        pb.append(add_hfeatures(va)[sub].reset_index(drop=True))
    return pd.concat(pa, axis=1), pd.concat(pb, axis=1)


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    res = {}
    for vy in (2022, 2024):
        va = df[df.season == vy].reset_index(drop=True)
        tr = df[df.season < vy].reset_index(drop=True)
        keep = ((va.game_type == "R") | (va.season >= 2023)).to_numpy()
        y = va[TARGET].to_numpy()
        for name, sub in VAR:
            t0 = time.time()
            xa, xb = enc(tr, va, cols, sub)
            m = HistGradientBoostingClassifier(**PRM); m.fit(xa, tr[TARGET])
            p = m.predict_proba(xb)[:, 1]
            res[(vy, name)] = (brier_score_loss(y[keep], p[keep]), roc_auc_score(y[keep], p[keep]))
            print("  %d %-14s n=%3d brier=%.6f auc=%.4f (%ds)" % (
                vy, name, xa.shape[1], res[(vy, name)][0], res[(vy, name)][1],
                round(time.time() - t0)), flush=True)
            del xa, xb, m; gc.collect()
    print("\n=== base 대비 기여 (e-5) ===")
    print("%-14s %10s %10s %10s" % ("군", "2022", "2024", "평균"))
    rows = []
    for name, _ in VAR[1:]:
        g22 = (res[(2022, "base")][0] - res[(2022, name)][0]) * 1e5
        g24 = (res[(2024, "base")][0] - res[(2024, name)][0]) * 1e5
        rows.append((name, g22, g24, (g22 + g24) / 2))
        print("%-14s %+10.2f %+10.2f %+10.2f" % (name, g22, g24, (g22 + g24) / 2))
    pd.DataFrame(rows, columns=["group", "g2022", "g2024", "mean"]).to_csv(
        HERE / "results" / "exp77_group_ablation.csv", index=False, encoding="utf-8-sig")
    good = [r[0] for r in rows if r[1] > 0 and r[2] > 0]
    print("\n두 clean fold 모두 개선한 군: %s" % (good if good else "없음"))


if __name__ == "__main__":
    main()

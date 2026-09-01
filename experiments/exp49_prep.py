# -*- coding: utf-8 -*-
"""실험 49 준비: 2024 fold 잔차 + 역산 라벨 캐시 생성."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent

COLS = ["row_id","season","game_month","inning","top_bottom","game_type",
        "balls_before","strikes_before","outs_before","score_diff_pitcher_team",
        "runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","li",
        "pitcher_id","batter_id","pitcher_hand","batter_hand",
        "asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate","asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate","asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate",
        "asof_pitcher_pitchmix_n","asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate","control_success"]

df = pd.read_csv(HERE/"data"/"train.csv", usecols=COLS)
y4 = np.load(HERE/"results"/"exp24_reconstructed_labels.npy")
assert len(y4) == len(df)
df["cls"] = y4.astype(np.int8)

nm = {0:"성공",1:"의도반대",2:"한가운데",3:"크게벗어남"}
print("=== 역산 라벨 분포 ===")
vc = df.cls.value_counts().sort_index()
for k,v in vc.items():
    print("  %d %-9s %8d  %.4f" % (k, nm[k], v, v/len(df)))

# ---- 잔차: 현행 최고 앙상블 근사 (0.85 tuned_league_role + 0.15 nn_seedavg) ----
o1 = pd.read_csv(HERE/"results"/"exp30_tuned_league_role_oof.csv.gz")
o2 = pd.read_csv(HERE/"results"/"exp37_nn_seedavg_oof.csv.gz")
print("\noof1 cols:", list(o1.columns))
print("oof2 cols:", list(o2.columns))
c1 = [c for c in o1.columns if c in ("prediction","pred","y_pred")][0]
c2 = [c for c in o2.columns if c in ("prediction","pred","y_pred")][0]
o = (o1[["row_id","season","control_success",c1]].rename(columns={c1:"p_gbdt"})
     .merge(o2[["row_id",c2]].rename(columns={c2:"p_nn"}), on="row_id"))
o["ens"] = 0.85*o.p_gbdt + 0.15*o.p_nn
o["resid"] = o.control_success - o.ens
for s in sorted(o.season.unique()):
    z = o[o.season==s]
    print("  %d Brier=%.6f  n=%d" % (s, (z.resid**2).mean(), len(z)))

df.merge(o[["row_id","ens","resid"]], on="row_id", how="left").to_pickle(
    HERE/"results"/"exp49_base.pkl")
print("\n캐시 저장 완료")

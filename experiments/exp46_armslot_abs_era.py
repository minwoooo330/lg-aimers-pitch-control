# -*- coding: utf-8 -*-
"""실험 46: ABS 체제 전환이 팔각도별로 다르게 작용했는가.

공개 자료 확인 사항(2026-08-19 검색):
  - KBO 1군 ABS 정식 도입은 2024년. 퓨처스 전 구장 확대는 2026년.
  - ABS는 인간 심판이 보수적으로 보던 존 상단을 규정대로 잡아준다.
    높은 코스 직구 비율 2017년 30.7% -> 2023년 40.5% -> 2024년 43.8%.
    상단 코너 스트라이크 인정률은 3배 이상 증가.
  - "낮은 릴리스 포인트 때문에 높은 코스 공략이 구조적으로 어려운
    사이드암 투수에게 ABS 존은 가혹한 환경"이라는 현장 평가가 있다.

가설 1: 팔각도가 낮은 투수일수록 2024년(ABS 도입) 하락폭이 크다.
가설 2: 한가운데 실투 증가는 존 상단 공략과 연결되므로 직구에서 두드러진다.

두 가설이 맞다면 팔각도 x ABS체제, 구종 x 연도 상호작용이 필요하다.
지금 모델은 6년을 통으로 학습해 체제 차이를 평균해 버린다.
"""
from pathlib import Path
import sys, gc
import numpy as np
import pandas as pd
from trackman_features import match_exact_games, build_pitcher_mapping

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
P = "pitcher_trackman_id"
MAIN = ["row_id", "season", "game_type", "inning", "top_bottom", "balls_before",
        "strikes_before", "outs_before", "pitcher_id", "pitcher_hand", "batter_hand",
        "asof_pitcher_n", "control_success"]
TMC = ["season", "trackman_game_id", "pitch_no", "inning", "top_bottom", "balls_before",
       "strikes_before", "outs_before", "pitcher_trackman_id", "pitcher_hand",
       "batter_hand", "tagged_pitch_type", "pitch_type_group", "rel_speed", "rel_height",
       "rel_side", "induced_vert_break"]


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", usecols=MAIN)
    df["cls"] = np.load(HERE / "results" / "exp24_reconstructed_labels.npy")
    tm = pd.read_csv(HERE / "data" / "trackman_history.csv", usecols=TMC)
    mg, ts, matches = match_exact_games(df, tm)
    del tm; gc.collect()

    # 팔각도는 체제 전환 이전(2019~2022) 자료로만 정의한다.
    # 2023~2024 자료를 쓰면 결과로 원인을 정의하는 순환이 된다.
    fb = ts[(ts.season <= 2022) & (ts.pitch_type_group == "fastball")]
    slot = fb.groupby(P).rel_height.agg(["mean", "size"])
    slot = slot[slot["size"] >= 300]["mean"]
    mapping = build_pitcher_mapping(mg, ts, matches, 2025)
    m2t = dict(zip(mapping.pitcher_id, mapping.pitcher_trackman_id))

    R = df[df.game_type == "R"].copy()
    R["slot"] = R.pitcher_id.map(m2t).map(slot)
    R = R[R.slot.notna()]
    R["slot_q"] = pd.qcut(R.slot, 4, labels=["Q1 최저(사이드암)", "Q2", "Q3", "Q4 최고(오버핸드)"])
    print("=== 가설 1: 팔각도 분위별 연도 성공률 (1군) ===")
    t = R.pivot_table(index="slot_q", columns="season", values="control_success", observed=True)
    t["23→24"] = t[2024] - t[2023]
    t["19→24"] = t[2024] - t[2019]
    print(t.round(4).to_string())
    print("\n  릴리스 높이 평균:")
    print(R.groupby("slot_q", observed=True).slot.agg(["mean", "size"]).round(3).to_string())

    print("\n=== 가설 1 보조: 팔각도 분위별 '한가운데' 실투 비율 ===")
    R["mid"] = (R.cls == 2).astype(float)
    tm2 = R.pivot_table(index="slot_q", columns="season", values="mid", observed=True)
    tm2["23→24"] = tm2[2024] - tm2[2023]
    print(tm2.round(4).to_string())

    # 행 단위 정렬로 구종별 연도 추이
    mi = mg.groupby("_game_idx", sort=False).indices
    ti = ts.groupby("trackman_game_id", sort=False).indices
    parts = []
    for row in matches.itertuples(index=False):
        a, b = mi[row.main_game_idx], ti[row.trackman_game_id]
        if len(a) != len(b):
            continue
        L = mg.iloc[a][["cls", "season", "game_type"]].reset_index(drop=True)
        Rt = ts.iloc[b][["tagged_pitch_type"]].reset_index(drop=True)
        parts.append(pd.concat([L, Rt], axis=1))
    al = pd.concat(parts, ignore_index=True)
    al = al[(al.cls >= 0) & (al.game_type == "R")]
    print("\n=== 가설 2: 구종별 '한가운데' 실투 비율 연도 추이 (1군) ===")
    al["mid"] = (al.cls == 2).astype(float)
    keep = al.tagged_pitch_type.value_counts()
    keep = keep[keep >= 5000].index
    g = al[al.tagged_pitch_type.isin(keep)]
    t3 = g.pivot_table(index="tagged_pitch_type", columns="season", values="mid")
    t3["23→24"] = t3[2024] - t3[2023]
    t3["19→24"] = t3[2024] - t3[2019]
    print(t3.round(4).sort_values("19→24", ascending=False).to_string())
    print("\n=== 구종 사용 비율 연도 추이 (1군) — 존 공략 방식 변화 확인 ===")
    t4 = pd.crosstab(g.season, g.tagged_pitch_type, normalize="index")
    print(t4.round(4).to_string())


if __name__ == "__main__":
    main()

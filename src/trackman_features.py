# -*- coding: utf-8 -*-
"""공식 train과 Trackman의 동일 경기 투구열을 이용한 안전한 ID 대응 및 과거 프로필."""
from collections import defaultdict
import hashlib
import numpy as np
import pandas as pd

STATE_COLS_MAIN = ["inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
                   "pitcher_hand", "batter_hand"]
MEASURE_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break_aligned",
                "extension", "rel_height", "rel_side_aligned", "zone_speed"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]


def infer_main_games(main):
    """공식 train 행 순서에서 이닝이 1로 되돌아가는 지점을 새 경기로 본다."""
    inning = main["inning"].to_numpy()
    season = main["season"].to_numpy()
    new_game = np.zeros(len(main), dtype=bool)
    new_game[0] = True
    new_game[1:] = (inning[1:] < inning[:-1]) | (season[1:] != season[:-1])
    out = main.copy()
    out["_game_idx"] = np.cumsum(new_game) - 1
    return out


def _pack_state(inning, top_bottom, balls, strikes, outs, pitcher_hand, batter_hand):
    return ((((((inning.astype(np.int32) * 2 + top_bottom) * 4 + balls) * 3 + strikes)
              * 3 + outs) * 3 + pitcher_hand) * 3 + batter_hand).astype(np.int16)


def _hash_groups(ids, codes):
    ids = np.asarray(ids)
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    result = {}
    for start, end in zip(starts, ends):
        digest = hashlib.blake2b(codes[start:end].tobytes(), digest_size=16).digest()
        result[ids[start]] = (end - start, digest)
    return result


def match_exact_games(main, trackman):
    """투구 전 상태의 전체 순서가 동일한 경기만 1:1로 연결한다."""
    main = infer_main_games(main)
    main_code = _pack_state(
        main["inning"].to_numpy(), main["top_bottom"].map({"T": 0, "B": 1}).to_numpy(),
        main["balls_before"].to_numpy(), main["strikes_before"].to_numpy(),
        main["outs_before"].to_numpy(), main["pitcher_hand"].to_numpy(),
        main["batter_hand"].to_numpy(),
    )
    tm = trackman.sort_values(["trackman_game_id", "pitch_no"]).reset_index(drop=True)
    tm_code = _pack_state(
        tm["inning"].to_numpy(), tm["top_bottom"].map({"Top": 0, "Bottom": 1}).to_numpy(),
        tm["balls_before"].to_numpy(), tm["strikes_before"].to_numpy(),
        tm["outs_before"].to_numpy(), tm["pitcher_hand"].map({"Left": 1, "Right": 2}).to_numpy(),
        tm["batter_hand"].map({"Left": 1, "Right": 2}).to_numpy(),
    )
    main_hash = _hash_groups(main["_game_idx"].to_numpy(), main_code)
    tm_hash = _hash_groups(tm["trackman_game_id"].to_numpy(), tm_code)
    reverse = defaultdict(list)
    for game_id, key in tm_hash.items():
        reverse[key].append(game_id)
    rows = []
    game_season = main.groupby("_game_idx", sort=False)["season"].first()
    for game_idx, key in main_hash.items():
        candidates = reverse.get(key, [])
        if len(candidates) == 1:
            rows.append((game_idx, candidates[0], int(game_season.loc[game_idx]), key[0]))
    return main, tm, pd.DataFrame(rows, columns=["main_game_idx", "trackman_game_id", "season", "n_rows"])


def build_pitcher_mapping(main_with_game, tm_sorted, game_matches, cutoff_year,
                          min_votes=20, min_purity=0.99):
    """cutoff 이전의 정확히 일치한 행들에서 반복 투표로 투수 ID 대응을 확정한다."""
    pairs = game_matches[game_matches["season"] < cutoff_year]
    main_indices = main_with_game.groupby("_game_idx", sort=False).indices
    tm_indices = tm_sorted.groupby("trackman_game_id", sort=False).indices
    left, right = [], []
    for row in pairs.itertuples(index=False):
        mi = main_indices[row.main_game_idx]
        ti = tm_indices[row.trackman_game_id]
        if len(mi) != len(ti):
            continue
        left.append(main_with_game.iloc[mi]["pitcher_id"].to_numpy())
        right.append(tm_sorted.iloc[ti]["pitcher_trackman_id"].to_numpy())
    if not left:
        return pd.DataFrame(columns=["pitcher_id", "pitcher_trackman_id", "votes", "total_votes", "purity"])
    vote_rows = pd.DataFrame({"pitcher_id": np.concatenate(left),
                              "pitcher_trackman_id": np.concatenate(right)})
    votes = (vote_rows.groupby(["pitcher_id", "pitcher_trackman_id"]).size()
             .rename("votes").reset_index())
    totals = votes.groupby("pitcher_id")["votes"].sum().rename("total_votes")
    votes = votes.join(totals, on="pitcher_id")
    votes["purity"] = votes["votes"] / votes["total_votes"]
    best = (votes.sort_values(["pitcher_id", "votes"], ascending=[True, False])
            .drop_duplicates("pitcher_id"))
    best = best[(best["votes"] >= min_votes) & (best["purity"] >= min_purity)].copy()
    # 같은 Trackman ID에 둘 이상의 메인 ID가 붙으면 증거가 강한 하나만 남긴다.
    best = (best.sort_values(["purity", "votes"], ascending=False)
            .drop_duplicates("pitcher_trackman_id"))
    return best.sort_values("pitcher_id").reset_index(drop=True)


def build_trackman_profile(trackman, mapping, cutoff_year):
    """cutoff 이전 Trackman으로 투수×구종군 측정치와 과거 역할 프로필을 만든다."""
    tm = trackman[trackman["season"] < cutoff_year].copy()
    sign = tm["pitcher_hand"].map({"Right": 1.0, "Left": -1.0})
    tm["rel_side_aligned"] = tm["rel_side"] * sign
    tm["horz_break_aligned"] = tm["horz_break"] * sign

    base = tm[tm["pitch_type_group"].isin(PITCH_GROUPS)].copy()
    grouped = base.groupby(["pitcher_trackman_id", "pitch_type_group"])
    agg = grouped[MEASURE_COLS].agg(["mean", "std"])
    agg.columns = [f"tm_{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    counts = grouped.size().rename("tm_pitch_group_n").reset_index()
    agg = agg.merge(counts, on=["pitcher_trackman_id", "pitch_type_group"], how="left")
    wide = agg.pivot(index="pitcher_trackman_id", columns="pitch_type_group")
    wide.columns = [f"{name}_{group}" for name, group in wide.columns]
    wide = wide.reset_index()

    total_n = base.groupby("pitcher_trackman_id").size().rename("tm_total_n")
    wide = wide.merge(total_n, on="pitcher_trackman_id", how="outer")
    for group in PITCH_GROUPS:
        ncol = f"tm_pitch_group_n_{group}"
        if ncol in wide:
            wide[f"tm_pitch_share_{group}"] = wide[ncol] / wide["tm_total_n"]

    # 과거 경기별 역할: 팀의 첫 투수이면서 1회 시작이면 선발 후보.
    ordered = tm.sort_values(["trackman_game_id", "pitch_no"])
    first_pitcher = (ordered.groupby(["trackman_game_id", "pitcher_team"], sort=False)
                     .first()["pitcher_trackman_id"].rename("first_pitcher").reset_index())
    app = (tm.groupby(["trackman_game_id", "pitcher_team", "pitcher_trackman_id"])
           .agg(role_pitch_count=("trackman_id", "size"),
                role_min_inning=("inning", "min"),
                role_innings_covered=("inning", "nunique"))
           .reset_index().merge(first_pitcher, on=["trackman_game_id", "pitcher_team"], how="left"))
    app["role_starter"] = ((app["pitcher_trackman_id"] == app["first_pitcher"])
                           & (app["role_min_inning"] == 1)).astype(np.int8)
    app["role_long_relief"] = ((app["role_starter"] == 0)
                               & ((app["role_pitch_count"] > 50)
                                  | (app["role_innings_covered"] >= 3))).astype(np.int8)
    app["role_short_relief"] = ((app["role_starter"] == 0)
                                & (app["role_long_relief"] == 0)).astype(np.int8)
    app["role_pitches_per_inning"] = app["role_pitch_count"] / app["role_innings_covered"].clip(lower=1)
    role = (app.groupby("pitcher_trackman_id")
            .agg(tm_appearances=("trackman_game_id", "size"),
                 tm_starter_share=("role_starter", "mean"),
                 tm_long_relief_share=("role_long_relief", "mean"),
                 tm_short_relief_share=("role_short_relief", "mean"),
                 tm_avg_pitches_per_game=("role_pitch_count", "mean"),
                 tm_avg_innings_per_game=("role_innings_covered", "mean"),
                 tm_avg_pitches_per_inning=("role_pitches_per_inning", "mean"))
            .reset_index())
    wide = wide.merge(role, on="pitcher_trackman_id", how="outer")
    profile = mapping[["pitcher_id", "pitcher_trackman_id", "purity", "votes"]].merge(
        wide, on="pitcher_trackman_id", how="left")
    profile = profile.drop(columns=["pitcher_trackman_id"]).set_index("pitcher_id")
    profile = profile.rename(columns={"purity": "tm_mapping_purity", "votes": "tm_mapping_votes"})
    return profile


def build_trackman_profile_by_league(trackman, mapping, cutoff_year):
    """1군(R)과 퓨처스(F)를 분리한 투수×구종군 평균 및 역할 프로필."""
    tm = trackman[trackman["season"] < cutoff_year].copy()
    tm["game_type"] = np.where(tm["pitcher_team"].astype(str).str.startswith("MIN_"), "F", "R")
    sign = tm["pitcher_hand"].map({"Right": 1.0, "Left": -1.0})
    tm["rel_side_aligned"] = tm["rel_side"] * sign
    tm["horz_break_aligned"] = tm["horz_break"] * sign

    base = tm[tm["pitch_type_group"].isin(PITCH_GROUPS)].copy()
    keys = ["pitcher_trackman_id", "game_type", "pitch_type_group"]
    grouped = base.groupby(keys)
    means = grouped[MEASURE_COLS].mean()
    means.columns = [f"tm_league_{metric}_mean" for metric in means.columns]
    means = means.reset_index()
    counts = grouped.size().rename("tm_league_pitch_group_n").reset_index()
    means = means.merge(counts, on=keys, how="left")
    wide = means.pivot(index=["pitcher_trackman_id", "game_type"], columns="pitch_type_group")
    wide.columns = [f"{name}_{group}" for name, group in wide.columns]
    wide = wide.reset_index()
    total_n = base.groupby(["pitcher_trackman_id", "game_type"]).size().rename("tm_league_total_n").reset_index()
    wide = wide.merge(total_n, on=["pitcher_trackman_id", "game_type"], how="outer")
    for group in PITCH_GROUPS:
        ncol = f"tm_league_pitch_group_n_{group}"
        if ncol in wide:
            wide[f"tm_league_pitch_share_{group}"] = wide[ncol] / wide["tm_league_total_n"]

    ordered = tm.sort_values(["trackman_game_id", "pitch_no"])
    first_pitcher = (ordered.groupby(["trackman_game_id", "pitcher_team"], sort=False)
                     .first()["pitcher_trackman_id"].rename("first_pitcher").reset_index())
    app = (tm.groupby(["trackman_game_id", "pitcher_team", "pitcher_trackman_id", "game_type"])
           .agg(role_pitch_count=("trackman_id", "size"), role_min_inning=("inning", "min"),
                role_innings_covered=("inning", "nunique"))
           .reset_index().merge(first_pitcher, on=["trackman_game_id", "pitcher_team"], how="left"))
    app["role_starter"] = ((app["pitcher_trackman_id"] == app["first_pitcher"])
                           & (app["role_min_inning"] == 1)).astype(np.int8)
    app["role_long_relief"] = ((app["role_starter"] == 0)
                               & ((app["role_pitch_count"] > 50)
                                  | (app["role_innings_covered"] >= 3))).astype(np.int8)
    app["role_short_relief"] = ((app["role_starter"] == 0)
                                & (app["role_long_relief"] == 0)).astype(np.int8)
    role = (app.groupby(["pitcher_trackman_id", "game_type"])
            .agg(tm_league_appearances=("trackman_game_id", "size"),
                 tm_league_starter_share=("role_starter", "mean"),
                 tm_league_long_relief_share=("role_long_relief", "mean"),
                 tm_league_short_relief_share=("role_short_relief", "mean"),
                 tm_league_avg_pitches_per_game=("role_pitch_count", "mean"),
                 tm_league_avg_innings_per_game=("role_innings_covered", "mean"))
            .reset_index())
    role["tm_league_starter_like_share"] = (role["tm_league_starter_share"]
                                             + role["tm_league_long_relief_share"])
    wide = wide.merge(role, on=["pitcher_trackman_id", "game_type"], how="outer")
    profile = mapping[["pitcher_id", "pitcher_trackman_id"]].merge(
        wide, on="pitcher_trackman_id", how="left").drop(columns="pitcher_trackman_id")
    return profile.set_index(["pitcher_id", "game_type"])

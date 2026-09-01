"""Step 1 베이스라인용 최소 피처 정의.

원본 47개 입력 컬럼을 그대로 쓰되, 범주형/수치형만 구분한다.
파생 피처는 만들지 않는다 (Step 2 EDA&FE에서 가설 기반으로 추가).
"""

from __future__ import annotations

import pandas as pd

ID_COLS = ["row_id"]
TARGET_COL = "control_success"

# 순서 의미가 없는 명목형 범주. sklearn HistGradientBoosting의 native categorical은
# 카디널리티 <=255 제약이 있어, 저카디널리티만 여기 넣는다.
# pitcher_id(792종)/batter_id(830종)는 고카디널리티라 baseline에서는 NUMERIC_COLS로
# 내려 원시 숫자로 취급한다 (순서 의미는 없지만 임계값 분할로나마 활용; Step 2에서
# 빈도/과거통계 인코딩 등으로 개선 예정).
CATEGORICAL_COLS = [
    "game_type",
    "top_bottom",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "game_dayofweek",
]

# 크기/순서에 의미가 있는 수치형
NUMERIC_COLS = [
    "season",
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

# Step 2-4에서 검증된 파생 피처. career-recent 차분은 raw 컬럼 두 개로부터
# 트리가 축 분할만으로 재현하기 어려운 정보(고표본 투수의 이력 staleness)를 담아
# 3시드 x 2fold(main +3.70e-5, ref +4.72e-5, 6/6 방향 일치)로 채택 확정.
DERIVED_NUMERIC_COLS = [
    "trend_gap5_success",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + DERIVED_NUMERIC_COLS


def _add_derived(df: pd.DataFrame, out: pd.DataFrame) -> None:
    out["trend_gap5_success"] = (
        df["asof_pitcher_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    ).astype("float64")


def build_categories(df: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    """전체 데이터(또는 fold의 train 구간)에서 범주형 각 컬럼의 카테고리 집합을 고정한다.
    익명 ID·손잡이 코드·요일처럼 값 자체가 닫힌 집합(통계량 아님)이라 이 단계는 타깃을
    보지 않으며 fold 간 누수가 아니다. train/valid에 동일한 카테고리 매핑을 적용하기 위함.
    """
    cats: dict[str, pd.CategoricalDtype] = {}
    for c in CATEGORICAL_COLS:
        values = sorted(df[c].astype(str).unique().tolist())
        cats[c] = pd.CategoricalDtype(categories=values)
    return cats


def prepare_features(df: pd.DataFrame, categories: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    """범주형 컬럼을 pandas 'category' dtype으로 바꿔 sklearn HGB의
    categorical_features='from_dtype' 자동 인식을 쓸 수 있게 한다.
    `categories`는 build_categories()로 미리 고정한 매핑을 그대로 전달해
    train/valid에서 동일한 카테고리 코드가 나오도록 한다.
    """
    out = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    for c in CATEGORICAL_COLS:
        out[c] = out[c].astype(str).astype(categories[c])
    for c in NUMERIC_COLS:
        out[c] = out[c].astype("float64")
    _add_derived(df, out)
    return out[FEATURE_COLS]

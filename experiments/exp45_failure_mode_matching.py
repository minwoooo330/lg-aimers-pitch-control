# -*- coding: utf-8 -*-
"""실험 45: 피처를 실패 '유형'에 대고 재검증한다.

exp43/44의 검증 설계에 결함이 있었다. 팔각도와 구종 구성을 집계된 이진
타깃(control_success)에 대고 쟀는데, 그 타깃 안에는 성격이 전혀 다른 실패
3종이 섞여 있다(의도반대 19.5% / 한가운데 14.9% / 크게벗어남 13.2%).

야구 메커니즘상 각 피처가 설명해야 할 실패 유형은 정해져 있다.
  팔각도 x 좌우      -> 의도 반대  (릴리스 방향 오류)
  구종 구성(커브)     -> 한가운데    (행잉, 실투)
  피로/카운트        -> 크게 벗어남

셋을 합친 타깃에 대면 신호가 희석된다. 여기서는 유형별로 직접 잰다.

측정 방법: 해당 투수의 그 유형 기준선(asof_*_rate)을 뺀 잔차와의 상관.
모델이 필요 없고, "투수 고유 성향"을 이미 제거한 순수 증분만 남는다.

라벨 출처: exp24의 역산 라벨 (0=성공, 1=의도반대, 2=한가운데, 3=크게벗어남)
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
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "control_success"]
TMC = ["season", "trackman_game_id", "pitch_no", "inning", "top_bottom", "balls_before",
       "strikes_before", "outs_before", "pitcher_trackman_id", "pitcher_hand",
       "batter_hand", "tagged_pitch_type", "pitch_type_group", "rel_speed",
       "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side"]


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", usecols=MAIN)
    y4 = np.load(HERE / "results" / "exp24_reconstructed_labels.npy")
    assert len(y4) == len(df), "라벨 길이 불일치"
    df["cls"] = y4

    print("=== 역산 라벨 분포 (전체) ===")
    nm = {0: "성공", 1: "의도반대", 2: "한가운데", 3: "크게벗어남"}
    vc = df.cls.value_counts().sort_index()
    for k, v in vc.items():
        print("  %d %-8s %8d  %.4f" % (k, nm[k], v, v / len(df)))

    print("\n=== 연도별 실패 유형 구성 (1군) ===")
    R = df[df.game_type == "R"]
    t = pd.crosstab(R.season, R.cls, normalize="index").rename(columns=nm)
    print(t.round(4).to_string())

    tm = pd.read_csv(HERE / "data" / "trackman_history.csv", usecols=TMC)
    mg, ts, matches = match_exact_games(df, tm)
    del tm; gc.collect()
    mapping = build_pitcher_mapping(mg, ts, matches, 2024)
    m2t = dict(zip(mapping.pitcher_id, mapping.pitcher_trackman_id))

    cur = df[df.season == 2024].copy()
    cur["tmid"] = cur.pitcher_id.map(m2t)
    # 투수 자신의 유형별 기준선을 뺀 잔차 = 순수 증분 신호
    tgt = {
        "성공(집계)": cur.control_success.to_numpy() - cur.asof_pitcher_success_rate.to_numpy(),
        "의도반대": (cur.cls == 1).to_numpy().astype(float) - cur.asof_pitcher_reverse_rate.to_numpy(),
        "한가운데": (cur.cls == 2).to_numpy().astype(float) - cur.asof_pitcher_middle_rate.to_numpy(),
        "크게벗어남": (cur.cls == 3).to_numpy().astype(float) - (
            cur.asof_pitcher_ball_rate.to_numpy() * 0.0),  # 전용 기준선 없음 -> 원값 사용
    }

    h = ts[ts.season < 2024].copy()
    sgn = h.pitcher_hand.map({"Right": 1.0, "Left": -1.0})
    h["rs"] = h.rel_side * sgn
    fb = h[h.pitch_type_group == "fastball"]

    def z(s):
        return (s - s.mean()) / s.std()

    slot_h = z(fb.groupby(P).rel_height.mean())
    slot_s = z(fb.groupby(P).rs.mean())
    sh = h.groupby([P, "tagged_pitch_type"]).size().rename("n").reset_index()
    sh["p"] = sh.n / sh.groupby(P).n.transform("sum")
    piv = sh.pivot(index=P, columns="tagged_pitch_type", values="p").fillna(0.0)

    ph = cur.pitcher_hand.to_numpy(); bh = cur.batter_hand.to_numpy()
    opp = (ph != bh).astype(float)
    v_h = cur.tmid.map(slot_h).to_numpy()
    v_s = cur.tmid.map(slot_s).to_numpy()

    feats = {
        "팔각도 x 반대손": v_h * opp,
        "팔각도 x 같은손": v_h * (1 - opp),
        "릴리스좌우 x 반대손": v_s * opp,
        "커브 비중": cur.tmid.map(piv.get("Curveball")).to_numpy() if "Curveball" in piv else None,
        "슬라이더 비중": cur.tmid.map(piv.get("Slider")).to_numpy() if "Slider" in piv else None,
        "커터 비중": cur.tmid.map(piv.get("Cutter")).to_numpy() if "Cutter" in piv else None,
        "직구 비중": cur.tmid.map(piv.get("Fastball")).to_numpy() if "Fastball" in piv else None,
        "체인지업 비중": cur.tmid.map(piv.get("ChangeUp")).to_numpy() if "ChangeUp" in piv else None,
    }

    print("\n=== 피처 x 실패유형 잔차 상관 ===")
    print("(각 유형의 투수 기준선 asof_*_rate를 뺀 뒤의 순수 증분)")
    hdr = "%-22s" % "피처"
    for k in tgt:
        hdr += "%12s" % k
    print(hdr); print("-" * len(hdr))
    for fname, v in feats.items():
        if v is None:
            continue
        v = np.asarray(v, float)
        line = "%-22s" % fname
        for k, y in tgt.items():
            ok = np.isfinite(v) & np.isfinite(y)
            c = np.corrcoef(v[ok], y[ok])[0, 1] if ok.sum() > 5000 else np.nan
            line += "%+12.5f" % c
        print(line)

    print("\n=== 참고: 유형별 예측 난이도 (투수 기준선만으로 설명되는 정도) ===")
    for k, col, cls in [("의도반대", "asof_pitcher_reverse_rate", 1),
                        ("한가운데", "asof_pitcher_middle_rate", 2)]:
        act = (cur.cls == cls).to_numpy().astype(float)
        base = cur[col].to_numpy()
        ok = np.isfinite(base)
        print("  %-8s 실제비율 %.4f  기준선평균 %.4f  상관 %+.4f"
              % (k, act.mean(), np.nanmean(base), np.corrcoef(base[ok], act[ok])[0, 1]))


if __name__ == "__main__":
    main()

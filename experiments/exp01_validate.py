# -*- coding: utf-8 -*-
"""
실험 01: 2019~2023 학습 -> 2024 검증
- 베이스라인(상수) 대비 모델이 실제로 점수를 내는지
- '밀기'(drift 보정)가 효과 있는지 정량 확인
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
D = "data/"
ID, TGT = "row_id", "control_success"
CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]

print("데이터 로드 중...")
df = pd.read_csv(D + "train.csv", encoding="utf-8-sig")
print(f"  {len(df):,}행 x {df.shape[1]}컬럼  ({time.time()-t0:.0f}s)")

# ---------- 피처 준비 ----------
feat_cols = [c for c in df.columns if c not in (ID, TGT)]
X_all = df[feat_cols].copy()
for c in CAT:
    X_all[c] = X_all[c].astype("category").cat.codes.astype(np.int32)
cat_mask = [c in CAT for c in feat_cols]

y_all = df[TGT].values.astype(np.int8)
season = df["season"].values

tr_m, va_m = season < 2024, season == 2024
Xtr, ytr = X_all[tr_m], y_all[tr_m]
Xva, yva = X_all[va_m], y_all[va_m]
print(f"  학습 {len(Xtr):,} / 검증(2024) {len(Xva):,}")

# ---------- 평가 함수 ----------
r_va = yva.mean()
BASE = r_va * (1 - r_va)
print(f"\n2024 실제 성공률 r={r_va:.4f}, 기준선 Brier={BASE:.6f}")


def brier(p, y=yva):
    return np.mean((p - y) ** 2)


def score(p, y=yva):
    return max(0.0, 100000 * (1 - brier(p, y) / BASE))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def shift_to_mean(p, target):
    """로짓 공간에서 평균이 target이 되도록 이동 (0~1 범위 자동 보장)"""
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(logit(p) + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return sigmoid(logit(p) + (lo + hi) / 2)


# ---------- 0) 상수 베이스라인 ----------
print("\n" + "=" * 62)
print("[0] 상수 예측 (참고용)")
print("=" * 62)
r_tr = ytr.mean()
for nm, p in [("학습기간 평균으로 찍기", r_tr), ("2024 실제평균(정답)", r_va)]:
    pv = np.full(len(yva), p)
    print(f"  {nm:24s} p={p:.4f}  Brier={brier(pv):.6f}  점수={score(pv):8.1f}")

# ---------- 1) 모델 학습 ----------
print("\n" + "=" * 62)
print("[1] HistGradientBoosting 학습")
print("=" * 62)
t1 = time.time()
model = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
    min_samples_leaf=200, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
    categorical_features=cat_mask, random_state=42,
)
model.fit(Xtr, ytr)
print(f"  학습 완료 {time.time()-t1:.0f}s, 트리 {model.n_iter_}개")

t2 = time.time()
p_raw = model.predict_proba(Xva)[:, 1]
print(f"  추론 {len(Xva):,}건 {time.time()-t2:.1f}s "
      f"(24.6만건 환산 {(time.time()-t2)*245789/len(Xva):.1f}s)")
print(f"\n  예측 평균={p_raw.mean():.4f}  (실제 {r_va:.4f}, "
      f"차이 {p_raw.mean()-r_va:+.4f})")
print(f"  예측 범위: {p_raw.min():.3f} ~ {p_raw.max():.3f}, 표준편차 {p_raw.std():.4f}")
print(f"\n  ★ 밀기 없음     Brier={brier(p_raw):.6f}  점수={score(p_raw):8.1f}")

# ---------- 2) 밀기 스캔 ----------
print("\n" + "=" * 62)
print("[2] '밀기' 효과 — 예측 평균을 목표값으로 이동")
print("=" * 62)
print(f"  {'목표평균':>8s} {'실제평균':>9s} {'Brier':>10s} {'점수':>9s}")
best = (None, -1)
for tgt in np.arange(0.44, 0.545, 0.005):
    ps = shift_to_mean(p_raw, tgt)
    s = score(ps)
    mark = ""
    if s > best[1]:
        best = (tgt, s)
    print(f"  {tgt:>8.3f} {ps.mean():>9.4f} {brier(ps):>10.6f} {s:>9.1f}{mark}")
print(f"\n  ★ 최적 목표평균 = {best[0]:.3f} -> {best[1]:.1f}점")
print(f"  ★ (2024 실제평균은 {r_va:.4f})")
print(f"  ★ 밀기로 얻은 이득: {best[1]-score(p_raw):+.1f}점")

# ---------- 3) 연도별 성공률 추세 -> 2025 외삽 ----------
print("\n" + "=" * 62)
print("[3] 2025 성공률 추정 (제출 시 밀기 목표값)")
print("=" * 62)
g = df.groupby("season")[TGT].mean()
print(g.round(4).to_string())
x, yv = g.index.values.astype(float), g.values
a, b = np.polyfit(x, yv, 1)
print(f"\n  전체 6년 선형추세 -> 2025 = {a*2025+b:.4f}")
a2, b2 = np.polyfit(x[-3:], yv[-3:], 1)
print(f"  최근 3년 추세     -> 2025 = {a2*2025+b2:.4f}")
print(f"  2024값 그대로     -> 2025 = {yv[-1]:.4f}")

print(f"\n총 소요 {time.time()-t0:.0f}s")

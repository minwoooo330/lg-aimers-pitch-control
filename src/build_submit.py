# -*- coding: utf-8 -*-
"""
개선판(B: 원본 + 도메인 피처) 제출본 생성
- features.py 소스를 script.py 안에 통째로 삽입 -> 서버에서도 동일 피처 생성
- shift 값만 다른 제출본 3종
"""
import os
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import add_features

t0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, OUT = os.path.join(ROOT, "data"), os.path.join(ROOT, "submits")
ID, TGT = "row_id", "control_success"
CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]

print("로드...")
df = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
base_cols = [c for c in df.columns if c not in (ID, TGT)]

cat_maps = {}
for c in CAT:
    vals = sorted(df[c].dropna().astype(str).unique().tolist())
    cat_maps[c] = {v: i for i, v in enumerate(vals)}


def build_X(d):
    Xb = d[base_cols].copy()
    for c in CAT:
        Xb[c] = Xb[c].astype(str).map(cat_maps[c]).fillna(-1).astype(np.int32)
    return pd.concat([Xb, add_features(d)], axis=1)


X = build_X(df)
y = df[TGT].to_numpy(np.int8)
feat_cols = list(X.columns)
cat_mask = [c in CAT for c in feat_cols]
print(f"  X={X.shape} (원본 {len(base_cols)} + 신규 {X.shape[1]-len(base_cols)})")

print("\n전체 기간 학습 (설정 B)...")
t1 = time.time()
model = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
    min_samples_leaf=200, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
    categorical_features=cat_mask, random_state=42,
)
model.fit(X, y)
print(f"  완료 {time.time()-t1:.0f}s, 트리 {model.n_iter_}개")

m24 = df["season"].to_numpy() == 2024
raw_mean = float(model.predict_proba(X[m24])[:, 1].mean())
print(f"  2024구간 예측평균={raw_mean:.4f} (실제 {y[m24].mean():.4f})")

os.makedirs(OUT, exist_ok=True)
joblib.dump({"model": model, "base_cols": base_cols, "feat_cols": feat_cols,
             "cat_maps": cat_maps, "cat_cols": CAT},
            os.path.join(OUT, "model.pkl"), compress=3)
print(f"모델 저장 {os.path.getsize(f'{OUT}/model.pkl')/1e6:.1f}MB")

FEATURES_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.py"), encoding="utf-8").read()

SCRIPT_TMPL = '''# -*- coding: utf-8 -*-
"""제구 성공 확률 예측 - 추론 스크립트 (도메인 피처 포함)"""
import os
import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
SHIFT = {shift}          # 드리프트 보정 (예측 확률에 더할 값)

TEST_DIR, MODEL_DIR, OUT_DIR = "./data", "./model", "./output"


# ===================== 피처 정의 (학습과 동일) =====================
{features_src}
# ==================================================================


def main():
    b = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    model, base_cols = b["model"], b["base_cols"]
    feat_cols, cat_maps, cat_cols = b["feat_cols"], b["cat_maps"], b["cat_cols"]

    test = pd.read_csv(os.path.join(TEST_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(TEST_DIR, "sample_submission.csv"),
                      encoding="utf-8-sig")
    print("test=%d sub=%d" % (len(test), len(sub)))

    Xb = test.reindex(columns=base_cols).copy()
    for c in cat_cols:
        Xb[c] = Xb[c].astype(str).map(cat_maps[c]).fillna(-1).astype(np.int32)
    X = pd.concat([Xb, add_features(test)], axis=1)
    X = X.reindex(columns=feat_cols)
    print("features=%d" % X.shape[1])

    p = model.predict_proba(X)[:, 1]
    print("raw mean=%.5f" % p.mean())
    if SHIFT != 0.0:
        p = np.clip(p + SHIFT, 1e-6, 1 - 1e-6)
        print("shifted mean=%.5f (SHIFT=%+.4f)" % (p.mean(), SHIFT))

    pred = dict(zip(test[ID_COL].tolist(), p))
    sub[TARGET_COL] = [pred.get(r, 0.5) for r in sub[ID_COL]]
    os.makedirs(OUT_DIR, exist_ok=True)
    sub.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False,
               encoding="utf-8")
    print("saved %d rows mean=%.5f" % (len(sub), sub[TARGET_COL].mean()))


if __name__ == "__main__":
    main()
'''

REQ = "scikit-learn==1.8.0\njoblib==1.5.3\npandas==2.3.3\nnumpy==2.4.6\n"


def make_zip(name, shift):
    src = SCRIPT_TMPL.format(shift=repr(float(shift)),
                             features_src=FEATURES_SRC)
    path = os.path.join(OUT, name + ".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("script.py", src)
        z.writestr("requirements.txt", REQ)
        z.write(os.path.join(OUT, "model.pkl"), "model/model.pkl")
    return os.path.getsize(path) / 1e6


print(f"\n제출본 생성 (기준 예측평균 {raw_mean:.4f})")
print("=" * 60)
for nm, tgt, desc in [("v2A_noshift", None, "보정 없음"),
                      ("v2B_4747", 0.4747, "목표 0.4747 (6년 추세)"),
                      ("v2C_4622", 0.4622, "목표 0.4622 (최근3년 추세)")]:
    sh = 0.0 if tgt is None else round(tgt - raw_mean, 4)
    mb = make_zip(nm, sh)
    print(f"  {nm:14s} SHIFT={sh:+.4f}  {desc:24s} {mb:.1f}MB")

print(f"\n총 {time.time()-t0:.0f}s   위치: {os.path.abspath(OUT)}")

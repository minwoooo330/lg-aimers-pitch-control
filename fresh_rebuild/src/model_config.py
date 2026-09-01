"""Step 2 전체에서 공유하는 기본 GBDT 설정.

early_stopping=False로 고정한 이유: HistGradientBoostingClassifier는 n>10000이면
early_stopping='auto'가 기본값이라 내부적으로 검증 분할을 무작위 수행 -> 시드 간 잡음이
불필요하게 커진다(04_noise_floor.py 실측: main SD 6.6e-5 -> 3.7e-5, ref 3.6e-5 -> 1.9e-5).
비교 실험에서는 반복 횟수를 고정하고 이 잡음원을 끄는 게 맞다.
"""

DEFAULT_HGB_PARAMS = dict(
    max_iter=200,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    early_stopping=False,
    random_state=42,
)

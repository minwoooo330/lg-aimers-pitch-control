# LG Aimers 9기 - 투구 제구 성공 확률 예측

[대회 페이지](https://dacon.io/competitions/official/236743/overview/description)

투구 단위로 제구 성공 확률(`control_success`)을 예측한다.

## 대회 개요

| 항목 | 내용 |
| --- | --- |
| 평가지표 | Brier Skill Score |
| 리더보드 | Public = Private (전체 테스트 100%) |
| 학습 데이터 | 2019~2024, 1,475,092행 x 49컬럼 |
| 평가 데이터 | 2025, 245,789행 (비공개) |
| 제출 | 코드 제출(zip), 하루 5회 |
| 실행 제한 | 추론 10분 이내, 오프라인, Python |
| 외부 데이터 | 사용 불가 |

### 일정

| 날짜 | 내용 |
| --- | --- |
| 08.26 | 팀 병합 마감 |
| 09.01 | 리더보드 제출 마감 |
| 09.07 | 코드 및 PPT 제출 마감 |

### 제구 성공 기준

아래 세 가지는 제구 실패, 나머지는 제구 성공으로 간주한다.

1. 스트라이크존 가운데 부근으로 들어간 공
2. 스트라이크존에서 크게 벗어난 공
3. 포수의 요구 방향과 반대로 들어간 공

## 폴더 구조

```
.
├── requirements.txt
├── baseline/
│   └── baseline_submit.zip     대회 제공 베이스라인
├── docs/
│   └── data_description.md     대회 제공 데이터 설명서
├── data/
│   ├── test.csv                형식 확인용 5건 (대회 제공)
│   ├── sample_submission.csv   제출 양식 (대회 제공)
│   ├── train.csv               용량 문제로 git 제외
│   └── trackman_history.csv    용량 문제로 git 제외
├── src/
│   ├── features.py             파생 피처 정의
│   └── build_submit.py         학습 및 제출 zip 생성
└── submits/                    제출 파일 생성 위치 (git 제외)
```

## 실행

`train.csv`, `trackman_history.csv`는 용량 때문에 저장소에 없다.
대회 페이지에서 `open.zip`을 받아 압축 해제 후 `data/`에 넣는다.

```bash
pip install -r requirements.txt

python src/build_submit.py   # submits/ 에 제출 zip 생성
```

## 제출 파일 구조

zip 최상단에 아래 구조로 압축한다.

```
submit.zip
├── model/
│   └── model.pkl
├── requirements.txt
└── script.py
```

평가 서버는 `./data/test.csv`를 실제 평가 데이터로 교체한 뒤 `script.py`를 실행하고,
`./output/submission.csv`를 채점한다.

## 주의

- `data/`, `*.csv`, `*.pkl`, `*.zip`은 `.gitignore`로 제외되어 있다. 커밋 전 `git status`로 확인한다.
- 평가 데이터의 각 행은 독립적으로 예측해야 한다. `test.csv` 내부의 다른 행을 이용한
  누적·빈도·rolling·target encoding 피처는 대회 규정상 사용할 수 없다.
- 2025년 Trackman 데이터는 제공되지 않는다.

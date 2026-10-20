# Ensemble Benchmark — Baseline 및 Evaluation

4·5파트 합주 음원을 악기 종류별로 분리하는 SynthSOD X-UMX 기반 프로젝트입니다.

- 홈페이지: https://www.chopss.kr/ensemble/index.html
- 개인 GitHub: https://github.com/c-peace/ensemble-separation-challenge-2026
- [소용량 샘플 ZIP · 6.02 MB](https://raw.githubusercontent.com/c-peace/ensemble-separation-challenge-2026/main/samples/sample-dataset.zip) — GitHub에 포함, 각 데이터셋 5초의 혼합음과 정답
- [코드 ZIP](https://www.chopss.kr/ensemble/downloads/code.zip)
- [학습·평가용 데이터 ZIP](https://www.chopss.kr/ensemble/downloads/dataset.zip)
- [코드와 데이터 전체 ZIP](https://www.chopss.kr/ensemble/downloads/all-downloads.zip)

## 홈페이지

`index.html`을 브라우저에서 열면 대회 소개·데이터·참가 및 평가·리더보드를 볼 수 있습니다. CSS와 이미지를 포함한 단일 HTML입니다. `submission/Team_최평화_202126857.html`은 Challenge 저장소에 제출하는 동일 내용의 파일입니다.

## 포함 파일

- `baseline.ipynb`: 학습 코드
- `evaluate.ipynb`: 평가 코드
- `competition-kit/`: 모델·전처리·채점 소스, manifest 및 라이선스
- `competition-kit.zip`: 원본 배포 키트
- `DATASET.md`: 데이터 출처·라이선스·가공 내역
- `samples/sample-dataset.zip`: 청취·형식 확인용 소용량 샘플
- `tools/make_sample.py`: 전체 데이터 ZIP에서 샘플을 재생성하는 코드

## 소용량 샘플

샘플은 SynthSOD의 `sod_756_native` 81–86초와 URMP의 `urmp_43` 30–35초를 포함합니다. 혼합음 2개와 악기별 정답 7개, 총 WAV 9개이며 44,100 Hz · mono · float32입니다. 같은 장면의 모든 트랙을 같은 위치에서 잘라 길이·정렬을 유지했습니다. 원본 출처·라이선스와 SHA256 manifest도 ZIP 안에 있습니다.

샘플은 청취·파일 형식 확인용이며 학습 데이터가 아닙니다. `baseline.ipynb`와 `evaluate.ipynb` 실행에는 전체 `dataset.zip`이 필요합니다. 샘플 점수는 공식 leaderboard와 비교할 수 없습니다.

재생성: `python tools/make_sample.py /path/to/dataset.zip samples/sample-dataset.zip`

## 데이터와 실행

학습 데이터는 SynthSOD 363곡입니다. 공개 평가에는 validation의 SynthSOD 24개와 URMP 6개, 총 30개 혼합음 및 정답 stem을 사용합니다. 별도 test 음원은 배포하지 않습니다. 대용량 음원은 위 서버의 데이터 ZIP으로 제공합니다.

1. 이 폴더의 파일과 내려받은 `dataset.zip`을 같은 폴더에 둡니다. Colab에서는 `/content`를 사용합니다.
2. `baseline.ipynb`를 위에서부터 실행합니다. 필요한 패키지는 노트북에서 설치하며 데이터 ZIP도 자동으로 해제합니다.
3. 같은 `RUN_NAME`을 지정한 `evaluate.ipynb`를 실행합니다. 학습한 체크포인트가 필요합니다.
4. 연결 확인 시 두 노트북 모두 `SMOKE=True`, 전체 점수 계산 시 모두 `False`로 설정합니다. 작은 smoke 실행 점수는 전체 평가와 비교하지 않습니다.

학습 가중치는 포함하지 않습니다. 데이터 ZIP과 해제 데이터만 약 41.3 GB이며 모델·결과 공간이 추가로 필요합니다.

## 평가 및 비교

BSS Eval v4로 SDR·SIR·SAR를 계산합니다. 점수가 높을수록 좋습니다. 순위 기준은 `SynthSOD SDR × 0.5 + URMP SDR × 0.5`입니다. 프레임 → 부모 녹음 → 작품 → 악기 순으로 집계하며 상세 구현은 `competition-kit/ensemble_benchmark/scoring.py`와 `evaluation_metrics.py`에 있습니다.

현재 홈페이지에 기록된 전체 validation 결과는 X-UMX baseline SDR **1.469 dB**입니다. SIR·SAR 종합값은 기록되지 않았습니다. 이 묶음을 만들면서 재학습·재측정하지 않았으며, 추가 비교 실험 결과를 만들지 않았습니다. 다른 실험은 별도 `ENSEMBLE_RUN`으로 실행하고 같은 데이터·채점 방법을 사용하세요.

코드 라이선스와 원본 모델 출처는 `competition-kit/LICENSE`, `competition-kit/README.md`, `competition-kit/ensemble_benchmark/vendor/LICENSE.asteroid`를 유지합니다. 데이터 라이선스는 `DATASET.md`를 따릅니다.

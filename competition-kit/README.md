# X-UMX baseline

baseline.ipynb: 기존 손실, seed 42, 1 epoch, batch 16, 6초, 곡당 64개 샘플.
evaluate.ipynb: 선택한 run의 검증 SDR 및 보조 SIR/SAR.
네 배포 파일을 같은 폴더에 두고 노트북을 위에서부터 실행합니다. Colab은 /content에 업로드합니다.
dataset.zip은 train/validation만 포함하며, 자동 해제 위치는 prepared/입니다. 기존 dataset/prepared도 인식합니다.
데이터 해제와 학습 저장에 충분한 여유 공간이 필요합니다. 실행 시작 시 음원 무결성을 검사합니다.
평가에는 먼저 학습한 체크포인트가 필요합니다. 모델 가중치는 이 배포본에 없습니다.
반복 학습은 같은 run에서 재개합니다. 다른 실험은 ENSEMBLE_RUN을 바꿉니다.
SMOKE=True는 연결 확인용이며 전체 점수와 비교하지 않습니다.
SIR/SAR은 유효 프레임이 부족하면 None입니다. 학습 실험 비교용 log-mel 노트북/설정/결과는 포함하지 않습니다.

코드: SynthSOD-Baseline https://github.com/repertorium/SynthSOD-Baseline 기반, AGPL-3.0.
vendored Asteroid X-UMX는 별도 MIT 고지를 포함합니다.
데이터 출처와 가공 내역은 dataset.zip의 README를 확인하세요.

# 공개 학습·검증 데이터

압축을 프로젝트 폴더에 해제하면 dataset/prepared/train 및 validation이 생성됩니다.
- 학습: SynthSOD 363곡. 원래 선정 목록 중 sod_507과 sod_968은 전체 무음으로 제외됐습니다. 원본 manifest의 complete=false 표시는 그대로 보존했습니다.
- 검증: SynthSOD 24개, URMP 6개 장면(총 30개).
- 학습 혼합음은 정답 stem을 합쳐 생성하며, 검증에는 mixture와 정답 stem이 포함됩니다.
- 모노 44,100 Hz로 통일했고, 출력 악기 범주별 stem 병합·선정된 파트의 부분 혼합·공통 gain 처리를 적용했습니다. 원본 전체 길이를 유지합니다. 학습 FLAC 저장용 감쇠는 manifest gain에 반영됩니다.
- manifest.json과 record.json에 경로·길이·가공 버전·gain·SHA256이 있습니다. 원본 파일과 메타데이터 바이트는 변경하지 않았습니다.

## 출처와 라이선스
- SynthSOD: Garcia-Martinez 등, https://doi.org/10.5281/zenodo.13759492 . CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ . 해당 가공 데이터도 동일 라이선스입니다. mono 변환, stem 병합, subset 선정, 공통 gain 및 패키징 변경이 적용됐습니다. 재배포 시 저자·원본 링크·라이선스·변경 내역을 유지하세요.
- URMP: Li 등, https://doi.org/10.5061/dryad.ng3r749 . CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/ . 원본 출처를 명시한 Kaggle 미러 버전 1에서 검증용 26개 stem을 확보했습니다: https://www.kaggle.com/datasets/alonhaviv/multi-modal-music-performance-urmp . 접근 불가한 원본과 전체 바이트 동일성은 확인하지 않았으며, 확보 파일의 ZIP CRC·SHA256·48 kHz mono 24-bit 규격을 검증한 자료를 가공했습니다.

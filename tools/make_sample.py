"""Create aligned 5-second listening samples from the published validation ZIP.

Usage: python tools/make_sample.py /path/to/dataset.zip samples/sample-dataset.zip
Requires numpy and scipy, already used by the baseline.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.io import wavfile

SCENES = [('SynthSOD', 'sod_756_native'), ('URMP', 'urmp_43')]
RATE = 44100
SECONDS = 5

def create_sample(source, output):
    files = {}
    manifest = {'purpose': 'listening_and_format_preview_only', 'official_evaluation': False,
                'sample_rate': RATE, 'channels': 1, 'encoding': 'WAV float32', 'scenes': []}
    with zipfile.ZipFile(source) as archive:
        files['sample-dataset/SOURCES.md'] = archive.read('dataset/README.md')
        for dataset, scene in SCENES:
            prefix = f'dataset/prepared/validation/{dataset}/{scene}/'
            record = json.loads(archive.read(prefix + 'record.json'))
            names = ['mixture'] + record['target_classes']
            tracks = {}
            source_hashes = {}
            for name in names:
                raw = archive.read(prefix + name + '.wav')
                expected = record['sha256'][f'{dataset}/{scene}/{name}.wav']
                digest = hashlib.sha256(raw).hexdigest()
                if digest != expected:
                    raise ValueError(f'Source hash mismatch: {scene}/{name}')
                rate, audio = wavfile.read(io.BytesIO(raw))
                if rate != RATE or audio.ndim != 1 or audio.dtype != np.float32 or not np.isfinite(audio).all():
                    raise ValueError(f'Unexpected source format: {scene}/{name}')
                tracks[name] = audio
                source_hashes[name + '.wav'] = digest
            if len({len(a) for a in tracks.values()}) != 1:
                raise ValueError('Unaligned source lengths')
            frame_count = len(tracks['mixture']) // RATE
            energies = np.stack([
                np.mean(tracks[name][:frame_count * RATE].reshape(frame_count, RATE).astype(np.float64)**2, axis=1)
                for name in record['target_classes']
            ])
            # Pick a shared interval whose quietest one-second target frame is strongest.
            # This changes only the crop, never gain, sample rate or alignment.
            scores = [float(energies[:, s:s + SECONDS].min()) for s in range(frame_count - SECONDS + 1)]
            start = int(np.argmax(scores))
            if scores[start] <= 1e-12:
                raise ValueError('No fully active sample interval')
            start_sample, end_sample = start * RATE, (start + SECONDS) * RATE
            sample = {'dataset': dataset, 'source_scene_id': scene, 'source_split': 'validation',
                      'start_seconds': start, 'duration_seconds': SECONDS, 'num_samples': SECONDS * RATE,
                      'target_classes': record['target_classes'], 'source_sha256': source_hashes,
                      'mixture': f'{dataset}/{scene}/mixture.wav', 'references': {}, 'sha256': {}}
            slices = {}
            for name, audio in tracks.items():
                segment = audio[start_sample:end_sample]
                slices[name] = segment
                buffer = io.BytesIO()
                wavfile.write(buffer, RATE, segment)
                relative = f'{dataset}/{scene}/{name}.wav'
                data = buffer.getvalue()
                files['sample-dataset/' + relative] = data
                sample['sha256'][relative] = hashlib.sha256(data).hexdigest()
                if name != 'mixture': sample['references'][name] = relative
            error = float(np.max(np.abs(slices['mixture'].astype(np.float64) - sum(slices[n].astype(np.float64) for n in record['target_classes']))))
            if error > 1e-6: raise ValueError('Mixture/reference sum mismatch')
            sample['max_sum_error'] = error
            manifest['scenes'].append(sample)
    files['sample-dataset/manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    files['sample-dataset/README.md'] = '''# Ensemble Benchmark 소용량 샘플

SynthSOD·URMP의 공개 validation 음원에서 각각 5초씩 추출했습니다.
각 장면의 mixture.wav와 악기별 정답 WAV를 함께 제공합니다.
44,100 Hz · mono · WAV float32이며 모든 트랙은 같은 구간·길이입니다.
추가 gain 조정, 리샘플링, 개별 악기 정규화는 하지 않았습니다.

- SynthSOD: sod_756_native — Cello, Viola, Violin (바이올린 2파트 병합)
- URMP: urmp_43 — Horn, Trombone, Trumpet, Tuba (트럼펫 2파트 병합)
- 정확한 원본 구간·파일 SHA256은 manifest.json에서 확인할 수 있습니다.

이 ZIP은 청취·입출력 형식 확인용입니다. baseline.ipynb/evaluate.ipynb가 사용하는
전체 dataset.zip을 대체하지 않습니다. 학습에 사용하거나 공식 leaderboard 점수를
이 샘플로 계산하지 마세요. 전체 평가용 30개 장면은 서버의 dataset.zip에 있습니다.

## 출처 및 라이선스

원본 배포의 출처·가공·라이선스 안내는 SOURCES.md에 그대로 포함했습니다.
이 샘플의 추가 가공은 모든 트랙에 동일한 5초 구간 자르기와 WAV 재포장입니다.
SynthSOD 부분: CC BY-SA 4.0 (샘플도 동일), https://doi.org/10.5281/zenodo.13759492
URMP 부분: CC0 1.0, https://doi.org/10.5061/dryad.ng3r749
SOURCES.md의 전체 데이터 규모 설명은 원본 배포에 대한 것으로 이 샘플의 크기가 아닙니다.
'''.encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compresslevel=6)
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = create_sample(args.source, args.output)
    print(json.dumps({'bytes': args.output.stat().st_size, 'scenes': result['scenes']}, ensure_ascii=False, indent=2))

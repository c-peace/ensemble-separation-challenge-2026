"""Package a saved run for transfer from a Colab server to a PC."""
from pathlib import Path
import os
import tempfile
import zipfile


def export_run(run_dir, include_predictions=False):
    """Create an atomic ZIP next to the run. Source files are never modified.

    Run training/evaluation to completion (or interrupt it) before exporting.
    Only saved checkpoints, run metadata, logs and scores are included by default.
    """
    root = Path(run_dir).resolve()
    required = [root/'checkpoints/latest.pt', root/'run.json']
    if not all(p.is_file() for p in required):
        raise FileNotFoundError('먼저 학습 체크포인트와 run.json을 저장하세요.')
    files = required + [root/'train-log.jsonl']
    folders = ['results', 'local-manifests']
    if include_predictions:
        folders += ['predictions', 'predictions-smoke']
    for folder in folders:
        files.extend(p for p in (root/folder).rglob('*') if p.is_file())
    files = sorted(set(p for p in files if p.is_file()))
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError('Run export path escapes run directory')
    destination = root.parent/(root.name+'-export.zip')
    before = {p:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    descriptor, temporary = tempfile.mkstemp(prefix=root.name+'-',suffix='.zip.tmp',dir=root.parent)
    os.close(descriptor)
    try:
        # Store avoids recompressing large checkpoints and keeps memory bounded.
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for path in files:
                archive.write(path, str(Path(root.name)/path.relative_to(root)))
        if any((p.stat().st_size,p.stat().st_mtime_ns)!=before[p] for p in files):
            raise RuntimeError('학습 결과가 변경되었습니다. 학습·평가를 멈춘 뒤 다시 내보내세요.')
        os.replace(temporary,destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination

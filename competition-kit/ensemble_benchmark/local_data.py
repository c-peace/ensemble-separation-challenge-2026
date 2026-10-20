"""Find prepared data or safely unpack a local prepared.zip (stdlib only)."""

from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def _ready(root):
    return all(
        (root / split / "manifest.json").is_file() for split in ("train", "validation")
    )


def resolve_prepared(project_root, explicit=None, archive_root=None):
    """Prefer dataset/prepared, then prepared; explicit paths never fall back.

    Extraction is staged and only published after both manifests exist. Existing
    directories and the archive are never removed or overwritten.
    """
    project = Path(project_root).expanduser().resolve()
    if explicit is not None:
        target = Path(explicit).expanduser().resolve()
        if not _ready(target):
            raise FileNotFoundError(
                f"ENSEMBLE_PREPARED 경로의 train/validation manifest를 확인하세요: {target}"
            )
        return target
    for target in (project / "dataset/prepared", project / "prepared"):
        if _ready(target):
            return target
    base = Path(archive_root).resolve() if archive_root is not None else project
    # Colab code may be in a nested project while uploaded data is in /content.
    if base != project:
        for target in (base / "dataset/prepared", base / "prepared"):
            if _ready(target):
                return target
    archive = base / "dataset.zip"
    if not archive.is_file():
        archive = base / "prepared.zip"
    target = base / "prepared"
    if not archive.is_file():
        raise FileNotFoundError(
            f"dataset/prepared 또는 prepared에서 train/validation manifest를 찾지 못했습니다. "
            f"폴더를 준비하거나 {archive}를 업로드하세요."
        )
    if target.exists():
        raise FileExistsError(
            f"불완전한 폴더가 이미 있습니다: {target}. 내용을 확인하고 다른 위치로 옮긴 뒤 다시 실행하세요."
        )
    with zipfile.ZipFile(archive) as zf:
        entries = []
        names = set()
        for item in zf.infolist():
            name = item.filename
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or ":" in name
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError(f"허용하지 않는 ZIP 경로: {name}")
            if "__MACOSX" in path.parts or path.name == ".DS_Store":
                continue
            normal = str(path)
            if not item.is_dir():
                if normal in names:
                    raise ValueError(f"중복 ZIP 파일 경로: {normal}")
                names.add(normal)
            entries.append((item, normal))
        prefixes = [
            p
            for p in ("", "prepared/", "dataset/prepared/")
            if all(p + s + "/manifest.json" in names for s in ("train", "validation"))
        ]
        if len(prefixes) != 1:
            raise ValueError(
                "ZIP 안에는 train/validation 또는 prepared/train·validation 구조가 하나 있어야 합니다."
            )
        prefix = prefixes[0]
        selected = [
            (i, n[len(prefix) :])
            for i, n in entries
            if n.startswith(prefix) and n[len(prefix) :] and not i.is_dir()
        ]
        total = sum(i.file_size for i, _ in selected)
        if shutil.disk_usage(base).free < total:
            raise OSError(
                f"압축 해제 공간 부족: 추가 {total / 2**30:.2f}GiB가 필요합니다."
            )
        print(f"압축 해제: {archive} → {target} ({total / 2**30:.2f}GiB)", flush=True)
        with tempfile.TemporaryDirectory(prefix=".prepared-extract-", dir=base) as temp:
            staging = Path(temp) / "prepared"
            staging.mkdir()
            done, next_percent = 0, 10
            for info, relative in selected:
                dest = staging / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                done += info.file_size
                percent = int(100 * done / max(total, 1))
                if percent >= next_percent:
                    print(f"압축 해제 {percent}%", flush=True)
                    next_percent = (percent // 10 + 1) * 10
            if not _ready(staging):
                raise ValueError("압축 해제 후 train/validation manifest가 없습니다.")
            if target.exists():
                raise FileExistsError(
                    f"대상 폴더가 생성되어 덮어쓰기를 중단합니다: {target}"
                )
            staging.rename(target)
    print("데이터 준비 완료:", target, flush=True)
    return target

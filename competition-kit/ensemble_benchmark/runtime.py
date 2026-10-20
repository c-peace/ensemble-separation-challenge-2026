"""Shared data acquisition, checkpoint layout and evaluation for both notebooks."""

from pathlib import Path
import gc, hashlib, json, os, shutil, stat, urllib.request, zipfile, warnings
import numpy as np
import soundfile as sf
from .baseline import Trainer
from .losses import loss_config
from tqdm.auto import tqdm
from .scoring import score_dataset


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(8 * 2**20):
            h.update(b)
    return h.hexdigest()


def read_release(distribution):
    if str(distribution).startswith(("https://", "http://")):
        with urllib.request.urlopen(
            str(distribution).rstrip("/") + "/release.json", timeout=60
        ) as f:
            return json.load(f)
    return json.loads((Path(distribution) / "release.json").read_text())


def validate_artifact(artifact):
    key = artifact["id"]
    if not key or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in key
    ):
        raise ValueError("Invalid artifact ID")
    if Path(artifact["file"]).name != artifact["file"] or not artifact["file"].endswith(
        ".zip"
    ):
        raise ValueError("Invalid artifact filename")
    if len(artifact["sha256"]) != 64 or any(
        c not in "0123456789abcdef" for c in artifact["sha256"]
    ):
        raise ValueError("Invalid SHA256")
    if artifact["archive_bytes"] <= 0 or artifact["unpacked_bytes"] <= 0:
        raise ValueError("Invalid archive size")


def extracted_valid(root):
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        records = manifest.get("tracks", manifest.get("scenes", []))
        if not records:
            return False
        for record in records:
            checks = record.get("sha256", {})
            if not checks:
                return False
            required = set(record.get("stems", record.get("references", {})).values())
            if "mixture" in record:
                required.add(record["mixture"])
            if not required <= set(checks):
                return False
            for rel in required:
                path = (root / rel).resolve()
                if (
                    not path.is_relative_to(root.resolve())
                    or not path.is_file()
                    or sha256(path) != checks[rel]
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def acquire(artifact, distribution, cache, reserve_bytes=10 * 2**30):
    """Only one training shard cached at a time; verify bytes/hash on every fetch."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    validate_artifact(artifact)
    key = artifact["id"]
    dest = cache / key
    marker = dest / ".verified.json"
    if (
        marker.exists()
        and json.loads(marker.read_text())["sha256"] == artifact["sha256"]
        and extracted_valid(dest)
    ):
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    needed = artifact["archive_bytes"] + artifact["unpacked_bytes"] + reserve_bytes
    if shutil.disk_usage(cache).free < needed:
        raise RuntimeError(f"디스크 부족: {needed / 2**30:.1f}GiB필요")
    archive = cache / (key + ".zip.part")
    try:
        if str(distribution).startswith(("https://", "http://")):
            request = urllib.request.urlopen(
                str(distribution).rstrip("/") + "/" + artifact["file"], timeout=120
            )
        else:
            request = (Path(distribution) / artifact["file"]).open("rb")
        total = 0
        with request as source, archive.open("wb") as f:
            while chunk := source.read(8 * 2**20):
                total += len(chunk)
                if total > artifact["archive_bytes"]:
                    raise ValueError("Archiveexceedsdeclaredsize")
                f.write(chunk)
        if total != artifact["archive_bytes"] or sha256(archive) != artifact["sha256"]:
            raise ValueError("Archivesize/hashmismatch")
        dest.mkdir()
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) != artifact["unpacked_bytes"]:
                raise ValueError("Unpackedsize mismatch")
            for i in z.infolist():
                if not (dest / i.filename).resolve().is_relative_to(
                    dest.resolve()
                ) or stat.S_ISLNK(i.external_attr >> 16):
                    raise ValueError("UnsafeZIPmember")
            z.extractall(dest)
        marker.write_text(json.dumps({"sha256": artifact["sha256"]}))
        return dest
    except BaseException:
        if dest.exists():
            shutil.rmtree(dest)
        raise
    finally:
        archive.unlink(missing_ok=True)


def signature(release, settings):
    return hashlib.sha256(
        json.dumps({"release": release, "training": settings}, sort_keys=True).encode()
    ).hexdigest()


def train(
    release,
    distribution,
    cache,
    run_dir,
    device="cpu",
    epochs=1,
    tiny=False,
    steps_per_shard=None,
    batch_size=1,
    samples_per_track=64,
    segment_seconds=6.0,
    save_every=50,
    acquire_data=None,
    persistent_data=False,
    audio_root=None,
    loss_type="baseline",
    log_mel_weight=0.1,
):
    provider = acquire_data or acquire
    if save_every < 1 or epochs < 1:
        raise ValueError("epochs/save_every mustpositive")
    settings = dict(
        tiny=tiny,
        steps_per_shard=steps_per_shard,
        batch_size=batch_size,
        samples_per_track=samples_per_track,
        segment_seconds=segment_seconds,
    )
    objective = loss_config(loss_type, log_mel_weight)
    # Preserve legacy baseline signatures; only optional objectives extend them.
    if loss_type != "baseline":
        settings.update(objective)
    sig = signature(release, settings)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "checkpoints" / "latest.pt"
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    trainer = (
        Trainer.load(path, device)
        if path.exists()
        else Trainer(
            device=device,
            tiny=tiny,
            **(
                dict(loss_type=loss_type, log_mel_weight=log_mel_weight)
                if loss_type != "baseline"
                else {}
            ),
        )
    )
    if getattr(trainer, "loss_settings", loss_config()) != objective:
        raise ValueError("Checkpoint loss settings differ; use a new run directory")
    progress = trainer.progress
    if progress and progress["signature"] != sig:
        raise ValueError("다른 데이터/학습 설정입니다. 새 RUN_NAME을 사용하세요.")

    def position(epoch, shard, batch):
        return {
            "signature": sig,
            "epoch": epoch,
            "shard_index": shard,
            "batch_index": batch,
            "settings": settings,
        }

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "release_id": release["release_id"],
                "settings": settings,
                "loss_settings": objective,
                "signature": sig,
                "checkpoint": "checkpoints/latest.pt",
            },
            indent=2,
        )
    )
    if not progress:
        trainer.save(path, position(0, 0, 0))
        progress = trainer.progress
    # Remove stale interrupted training caches only inside the dedicated cache.
    for art in release["train"]:
        if acquire_data is None:
            validate_artifact(art)
        elif not art["id"] or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in art["id"]
        ):
            raise ValueError("Invalid artifact ID")
        stale = cache / art["id"]
        if not persistent_data and stale.is_dir():
            shutil.rmtree(stale)
    start_epoch = progress["epoch"]
    for epoch in range(start_epoch, epochs):
        first = progress["shard_index"] if epoch == start_epoch else 0
        for index in range(first, len(release["train"])):
            art = release["train"][index]
            start_batch = (
                progress["batch_index"]
                if epoch == start_epoch and index == first
                else 0
            )
            data = provider(art, distribution, cache)
            print(
                f"Epoch{epoch + 1}/{epochs} shard{index + 1}/{len(release['train'])} start_batch={start_batch}",
                flush=True,
            )

            def checkpoint_step(info, losses):
                if (
                    info["batch_index"] % save_every == 0
                    or info["batch_index"] == info["num_steps"]
                ):
                    trainer.save(path, position(epoch, index, info["batch_index"]))
                    with (run_dir / "train-log.jsonl").open("a") as f:
                        f.write(
                            json.dumps(
                                dict(
                                    epoch=epoch,
                                    shard=index,
                                    **info,
                                    losses=losses,
                                    loss_components=getattr(
                                        trainer, "loss_components", {}
                                    ),
                                )
                            )
                            + "\n"
                        )
                    tqdm.write(
                        f"Saved batch {info['batch_index']} / {info['num_steps']}"
                    )

            try:
                trainer.train_shard(
                    data / "manifest.json",
                    steps=steps_per_shard,
                    batch_size=batch_size,
                    seed=42 + epoch * 10000 + index,
                    samples_per_track=samples_per_track,
                    segment_seconds=segment_seconds,
                    start_step=start_batch,
                    on_step=checkpoint_step,
                    progress_desc=f"Epoch {epoch + 1}/{epochs} shard {index + 1}/{len(release['train'])}",
                    **({"root": audio_root} if audio_root is not None else {}),
                )
                next_epoch, next_shard = (
                    (epoch + 1, 0)
                    if index + 1 == len(release["train"])
                    else (epoch, index + 1)
                )
                trainer.save(path, position(next_epoch, next_shard, 0))
            finally:
                if not persistent_data:
                    shutil.rmtree(data, ignore_errors=True)
    return path


def evaluate(
    release,
    distribution,
    cache,
    run_dir,
    device="cpu",
    chunk_seconds=20.0,
    limit_scenes=None,
    max_seconds=None,
    allow_partial=False,
    acquire_data=None,
    audio_root=None,
    reuse_predictions=False,
    trust_legacy_predictions=False,
    include_metrics=False,
):
    run_dir = Path(run_dir)
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            "baseline.ipynb 학습을 먼저 실행하세요: " + str(checkpoint)
        )
    run = json.loads((run_dir / "run.json").read_text())
    if run["release_id"] != release["release_id"]:
        raise ValueError("모델과 검증 데이터 release가 다릅니다.")
    # Prepare audio before loading the model to reduce peak host memory.
    provider = acquire_data or acquire
    root = provider(release["validation"], distribution, cache)
    from .evaluation_metrics import (
        prediction_provenance,
        validate_reuse,
        write_json,
        add_supplementary,
    )

    full_manifest = json.loads((root / "manifest.json").read_text())
    result_path = run_dir / "results" / "validation.json"
    provenance_path = run_dir / "results" / "prediction-provenance.json"
    predictions_path = run_dir / "predictions"
    if reuse_predictions and (result_path.exists() or predictions_path.exists()):
        if allow_partial or max_seconds is not None or limit_scenes is not None:
            raise ValueError("Saved prediction reuse requires full evaluation")
        if not result_path.exists():
            raise ValueError(
                "Predictions exist without completed SDR report; rerun inference explicitly"
            )
        report = json.loads(result_path.read_text())
        provenance = validate_reuse(
            report,
            full_manifest,
            predictions_path,
            checkpoint,
            release["release_id"],
            provenance_path,
            trust_legacy=trust_legacy_predictions,
            inference_settings=dict(
                chunk_seconds=chunk_seconds,
                wiener=True,
                max_seconds=None,
                limit_scenes=None,
                diagnostic=False,
            ),
        )
        if include_metrics:
            references = Path(audio_root) if audio_root is not None else root
            report = add_supplementary(
                report,
                full_manifest,
                references,
                predictions_path,
                run_dir / "results" / "metric-cache",
            )
            report["prediction_provenance"] = provenance
            result_path = run_dir / "results" / "validation-extra.json"
            write_json(result_path, report)
        return report, result_path
    inference_checkpoint_hash = sha256(checkpoint)
    trainer = Trainer.load(checkpoint, device)
    if trainer.tiny and not allow_partial:
        raise ValueError("Tiny smoke모델은 공식 평가할 수 없습니다.")
    if (limit_scenes is not None or max_seconds is not None) and not allow_partial:
        raise ValueError("부분 평가는 진단모드만 가능합니다.")
    manifest = json.loads((root / "manifest.json").read_text())
    root = Path(audio_root) if audio_root is not None else root
    if not manifest.get("complete") and not allow_partial:
        raise ValueError("선정된 검증 데이터가 완성되지 않았습니다.")
    scenes = manifest["scenes"][:limit_scenes] if limit_scenes else manifest["scenes"]
    predictions = run_dir / ("predictions-smoke" if allow_partial else "predictions")
    references = root
    if max_seconds:
        # Diagnostic excerpts cannot produce a competition overall score.
        references = Path(cache) / "diagnostic-references"
        references.mkdir(exist_ok=True)
    processed = []
    with tqdm(
        total=len(scenes),
        desc="Separating",
        unit="scene",
        mininterval=1.0,
        dynamic_ncols=True,
    ) as progress:
        for scene in scenes:
            progress.set_postfix_str(scene["scene_id"], refresh=False)
            info = sf.info(root / scene["mixture"])
            if (
                info.samplerate != 44100
                or info.channels != 1
                or info.frames != scene["num_samples"]
            ):
                raise ValueError("Invalidevaluationmixture")
            frames = (
                min(info.frames, int(max_seconds * info.samplerate))
                if max_seconds
                else -1
            )
            mixture, sr = sf.read(
                root / scene["mixture"], dtype="float32", frames=frames
            )
            current = dict(scene)
            if max_seconds:
                mixture = mixture[: int(max_seconds * sr)]
                current["num_samples"] = len(mixture)
                current["references"] = {}
                for target, relative in scene["references"].items():
                    ref, _ = sf.read(
                        root / relative, dtype="float32", frames=len(mixture)
                    )
                    dest = references / scene["scene_id"] / (target + ".wav")
                    dest.parent.mkdir(exist_ok=True, parents=True)
                    sf.write(dest, ref, sr, subtype="FLOAT")
                    current["references"][target] = str(dest.relative_to(references))
            outputs = trainer.separate(
                mixture,
                scene["target_classes"],
                chunk_seconds=chunk_seconds,
                wiener=True,
                show_progress=True,
            )
            dest = predictions / scene["scene_id"]
            dest.mkdir(parents=True, exist_ok=True)
            for target, audio in outputs.items():
                sf.write(dest / (target + ".wav"), audio, sr, subtype="FLOAT")
            processed.append(current)
            progress.update(1)
            del mixture, outputs, audio
    del trainer
    gc.collect()
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    use_manifest = dict(manifest, scenes=processed)
    if allow_partial:
        use_manifest["complete"] = False
    report = score_dataset(
        use_manifest,
        references,
        predictions,
        split="validation",
        allow_partial=allow_partial,
        show_progress=True,
    )
    if sha256(checkpoint) != inference_checkpoint_hash:
        raise ValueError(
            "Checkpoint changed during evaluation; predictions cannot be attributed safely"
        )
    report["checkpoint_sha256"] = inference_checkpoint_hash
    report["release_id"] = release["release_id"]
    report["diagnostic"] = allow_partial
    results = run_dir / "results"
    results.mkdir(exist_ok=True)
    file = results / ("validation-smoke.json" if allow_partial else "validation.json")
    provenance = prediction_provenance(
        use_manifest,
        predictions,
        checkpoint,
        release["release_id"],
        dict(
            chunk_seconds=chunk_seconds,
            wiener=True,
            max_seconds=max_seconds,
            limit_scenes=limit_scenes,
            diagnostic=allow_partial,
        ),
    )
    report["prediction_provenance"] = provenance
    write_json(
        results
        / (
            "prediction-provenance-smoke.json"
            if allow_partial
            else "prediction-provenance.json"
        ),
        provenance,
    )
    write_json(file, report)
    if include_metrics:
        report = add_supplementary(
            report, use_manifest, references, predictions, results / "metric-cache"
        )
        file = results / (
            "validation-smoke-extra.json" if allow_partial else "validation-extra.json"
        )
        write_json(file, report)
    return report, file


_SELECTION_PATH = Path(__file__).resolve().parents[1] / "manifests" / "selection.json"
_KNOWN_SILENT_TRAIN_IDS = {"sod_507", "sod_968"}


def _local_manifests(prepared_root, smoke, excluded_train_ids):
    """Validate saved manifests without downloading, rewriting or deleting audio."""
    root = Path(prepared_root).resolve()
    selection = json.loads(_SELECTION_PATH.read_text())
    excluded = set(excluded_train_ids)
    if excluded - _KNOWN_SILENT_TRAIN_IDS:
        raise ValueError("Only the known silent training tracks may be excluded")
    manifests = {
        split: json.loads((root / split / "manifest.json").read_text())
        for split in ("train", "validation")
    }
    expected_train = {"sod_" + str(x) for x in selection["train"]["allowed_sod_ids"]}
    expected_validation = {
        x["scene_id"]
        for x in selection["synthsod_scenes"]
        if x["split"] == "validation"
    }
    expected_validation.update(
        "urmp_" + str(x["id"]) for x in selection["urmp"] if x["split"] == "validation"
    )
    files = []
    for split, key, id_key, expected in (
        ("train", "tracks", "track_id", expected_train),
        ("validation", "scenes", "scene_id", expected_validation),
    ):
        manifest = manifests[split]
        records = manifest[key]
        ids = [record[id_key] for record in records]
        if manifest.get("split") != split or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid {split} split or duplicate IDs")
        missing = expected - set(ids)
        if set(ids) - expected or (missing - excluded if split == "train" else missing):
            raise ValueError(
                f"Unexpected or missing {split} IDs: {sorted(missing | (set(ids) - expected))}"
            )
        if split == "validation" and not manifest.get("complete"):
            raise ValueError("Validation manifest is incomplete")
        if (
            split == "validation"
            and set(manifest.get("expected_scene_ids", [])) != expected
        ):
            raise ValueError("Validation expected_scene_ids differ from selection")
        for record in records:
            if (
                record.get("split") != split
                or record.get("sample_rate") != 44100
                or record.get("num_samples", 0) <= 0
            ):
                raise ValueError(f"Invalid audio metadata: {record[id_key]}")
            paths = list(
                record.get("stems" if split == "train" else "references", {}).values()
            )
            if not paths:
                raise ValueError(f"Missing audio references: {record[id_key]}")
            if split == "validation":
                if set(record["references"]) != set(record["target_classes"]):
                    raise ValueError("Validation targets differ from references")
                paths.append(record["mixture"])
            for relative in paths:
                path = (root / split / relative).resolve()
                if Path(relative).is_absolute() or not path.is_relative_to(
                    root / split
                ):
                    raise ValueError(f"Audio path escapes prepared split: {relative}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                digest = record.get("sha256", {}).get(relative, "")
                if len(digest) != 64 or any(
                    c not in "0123456789abcdef" for c in digest
                ):
                    raise ValueError(f"Missing or invalid SHA256 metadata: {relative}")
                info = sf.info(path)
                if (
                    info.samplerate != 44100
                    or info.channels != 1
                    or info.frames != record["num_samples"]
                ):
                    raise ValueError(f"Audio header differs from manifest: {relative}")
                if sha256(path) != digest:
                    raise ValueError(f"Audio SHA256 differs from manifest: {relative}")
                files.append((split, relative, path.stat().st_size, digest))
    actual_excluded = sorted(
        expected_train - {x["track_id"] for x in manifests["train"]["tracks"]}
    )
    fingerprint = signature(
        manifests,
        {
            "files": files,
            "selection": selection,
            "excluded": sorted(excluded),
            "smoke": smoke,
        },
    )
    if smoke:
        manifests["train"]["tracks"] = manifests["train"]["tracks"][:1]
        manifests["validation"]["scenes"] = [
            x for x in manifests["validation"]["scenes"] if x["scene_id"] == "urmp_43"
        ]
        manifests["validation"]["complete"] = False
    if not manifests["train"]["tracks"] or not manifests["validation"]["scenes"]:
        raise ValueError("Local dataset is empty")
    return manifests, fingerprint, actual_excluded


def build_local_release(
    prepared_root, smoke=False, excluded_train_ids=("sod_507", "sod_968")
):
    """Describe local train/validation data; Test is never read.

    The fingerprint includes manifest content and verified audio SHA256/size,
    so moving or copying unchanged data preserves checkpoint compatibility.
    Startup streams every referenced audio file to check its recorded hash.
    """
    manifests, fingerprint, excluded = _local_manifests(
        prepared_root, smoke, excluded_train_ids
    )
    if excluded:
        warnings.warn(
            "Known silent training tracks omitted: " + ", ".join(excluded),
            UserWarning,
            stacklevel=2,
        )
    return {
        "release_id": "local-" + fingerprint,
        "fingerprint": fingerprint,
        "smoke": bool(smoke),
        "allowed_missing_train_ids": sorted(set(excluded_train_ids)),
        "excluded_train_ids": excluded,
        "train_count": len(manifests["train"]["tracks"]),
        "validation_count": len(manifests["validation"]["scenes"]),
        "train": [{"id": "local_train"}],
        "validation": {"id": "local_validation"},
    }


def _local_provider(release, prepared_root, run_dir):
    root, run = Path(prepared_root).resolve(), Path(run_dir).resolve()
    if root.is_relative_to(run) or run.is_relative_to(root):
        raise ValueError("RUN_DIR and PREPARED_ROOT must not overlap")
    manifests, fingerprint, _ = _local_manifests(
        root, release["smoke"], release["allowed_missing_train_ids"]
    )
    if (
        fingerprint != release["fingerprint"]
        or release["release_id"] != "local-" + fingerprint
    ):
        raise ValueError("Local data changed; rebuild release and use a new RUN_NAME")

    def provider(artifact, distribution, cache):
        split = {"local_train": "train", "local_validation": "validation"}[
            artifact["id"]
        ]
        snapshot = run / "local-manifests" / split
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "manifest.json").write_text(
            json.dumps(manifests[split], ensure_ascii=False, indent=2)
        )
        return snapshot

    return provider


def train_local(
    release,
    prepared_root,
    run_dir,
    device="cpu",
    epochs=1,
    tiny=False,
    steps_per_shard=None,
    batch_size=1,
    samples_per_track=64,
    segment_seconds=6.0,
    save_every=50,
    loss_type="baseline",
    log_mel_weight=0.1,
):
    """Train against permanent stems, retaining every prepared input file."""
    provider = _local_provider(release, prepared_root, run_dir)
    return train(
        release,
        prepared_root,
        Path(run_dir) / "local-cache",
        run_dir,
        device=device,
        epochs=epochs,
        tiny=tiny,
        steps_per_shard=steps_per_shard,
        batch_size=batch_size,
        samples_per_track=samples_per_track,
        segment_seconds=segment_seconds,
        save_every=save_every,
        acquire_data=provider,
        persistent_data=True,
        audio_root=Path(prepared_root).resolve() / "train",
        loss_type=loss_type,
        log_mel_weight=log_mel_weight,
    )


def evaluate_local(
    release,
    prepared_root,
    run_dir,
    device="cpu",
    chunk_seconds=20.0,
    allow_partial=False,
    max_seconds=None,
    reuse_predictions=False,
    trust_legacy_predictions=False,
    include_metrics=False,
):
    """Evaluate saved validation audio, keeping diagnostics under RUN_DIR."""
    if release["smoke"] and not allow_partial:
        raise ValueError("Smoke release requires allow_partial=True")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    provider = _local_provider(release, prepared_root, run_dir)
    return evaluate(
        release,
        prepared_root,
        Path(run_dir) / "local-cache",
        run_dir,
        device=device,
        chunk_seconds=chunk_seconds,
        max_seconds=max_seconds,
        allow_partial=allow_partial,
        acquire_data=provider,
        audio_root=Path(prepared_root).resolve() / "validation",
        reuse_predictions=reuse_predictions,
        trust_legacy_predictions=trust_legacy_predictions,
        include_metrics=include_metrics,
    )

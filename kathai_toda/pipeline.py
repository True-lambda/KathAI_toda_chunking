from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import logging
import platform
import shutil
import tempfile
import tomllib

from . import __version__, audio
from .export import dump_json, export_all, sentence_records, uncovered_speech, write_csv
from .ingest import find_inputs, read_rows, normalize
from .ml import Models

LOG = logging.getLogger(__name__)
DEFAULTS = dict(device="cpu", threads=2, window_seconds=10.0, context_seconds=1.0,
                gap_penalty=2.0, max_alignment_cells=100_000_000,
                vad_threshold=0.5, vad_silence_ms=250, vad_max_seconds=30.0,
                padding_seconds=0.08, snap_seconds=0.20, low_score_threshold=0.30)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.suffix in {".py", ".html"}:
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def configuration(path=None, device=None):
    config = tomllib.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    settings = dict(DEFAULTS)
    unknown = set(config.get("settings", {})) - set(settings)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    settings.update(config.get("settings", {}))
    if device:
        settings["device"] = device
    if settings["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not 0 < settings["vad_threshold"] < 1 or not 0 <= settings["low_score_threshold"] <= 1:
        raise ValueError("Invalid VAD/score thresholds")
    if settings["threads"] < 1 or settings["window_seconds"] < 1 or settings["context_seconds"] < 0.02:
        raise ValueError("threads >= 1, window_seconds >= 1, context_seconds >= 0.02 required")
    if min(settings[x] for x in ("gap_penalty", "padding_seconds", "snap_seconds")) < 0:
        raise ValueError("Penalties and padding must be nonnegative")
    if settings["vad_silence_ms"] < 1 or settings["vad_max_seconds"] < 1 or settings["max_alignment_cells"] < 1:
        raise ValueError("VAD durations and max_alignment_cells must be positive")
    return config, settings


def read_corrections(path, folder):
    if not path:
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    result = []
    for r in records:
        if r.get("folder") and r["folder"] != folder:
            continue
        reviewed = str(r.get("reviewed", "")).lower() in {"true", "1", "yes"}
        skip = str(r.get("skip", "")).lower() in {"true", "1", "yes"}
        if not reviewed and not skip:
            continue
        patch = {"sheet": r["sheet"], "row": int(r["row"]), "skip": skip}
        if not skip:
            patch.update(start=float(r["start"]), end=float(r["end"]))
        result.append(patch)
    if not result:
        raise ValueError(f"No reviewed corrections for {folder!r}. Use review.html to confirm/export rows.")
    return result


def merge_patches(first, second):
    result = {}
    for patch in [*first, *second]:
        key = (patch.get("sheet", "Sheet1"), int(patch["row"]))
        result[key] = {**result.get(key, {}), **patch}
    return list(result.values())


def alignment_blocks(rows):
    """Fixed times are real anchors. Untimed rows between anchors align together."""
    pending, previous_end = [], 0.0
    blocks = []
    def flush(limit=None):
        nonlocal pending
        if not pending:
            return
        first = pending[0]
        start = max(first.scope_start, previous_end)
        end = min(first.scope_end, limit) if limit is not None else first.scope_end
        if start >= end:
            raise ValueError(f"No time left for {first.key}; check neighboring anchors")
        blocks.append((list(pending), start, end))
        pending = []
    for row in rows:
        if row.skip:
            continue
        if pending and (row.sheet, row.group, row.scope_start, row.scope_end) != (
                pending[0].sheet, pending[0].group, pending[0].scope_start, pending[0].scope_end):
            flush()
        if row.start is not None:
            flush(row.start)
            previous_end = row.end
        else:
            if pending and row.row <= pending[-1].row:
                raise ValueError("Transcript rows are out of order")
            pending.append(row)
    flush()
    return blocks


def run_one(folder, output_root, config, settings, models=None, *, overwrite=False,
            corrections=None, ignore_existing=False, vad_only=False):
    folder, output_root = Path(folder), Path(output_root)
    options = config.get("folders", {}).get(folder.name, {})
    wav, workbook = find_inputs(folder, options)
    recording = audio.load(wav)
    patches = merge_patches(options.get("rows", []), read_corrections(corrections, folder.name))
    rows = read_rows(workbook, recording.duration, patches, ignore_existing)
    for row in rows:
        if not row.skip and row.start is None:
            normalize(row.alignment_text if row.alignment_text is not None else row.transcript)
    hashes = {"audio": sha256(wav), "workbook": sha256(workbook)}
    source_hash = code_hash()
    stamp = {"version": __version__, "code_hash": source_hash, "hashes": hashes, "settings": settings, "patches": patches,
             "ignore_existing": ignore_existing, "vad_only": vad_only}
    fingerprint = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / folder.name
    if destination.is_symlink():
        raise ValueError("Output destination must not be a symlink")
    if destination.exists() and not overwrite:
        report_file = destination / "report.json"
        report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
        expected = report.get("output_files", [])
        if report.get("fingerprint") == fingerprint and expected and all((destination / p).is_file() for p in expected):
            LOG.info("Unchanged inputs and complete output: %s", destination)
            return report
        raise FileExistsError(f"Output already exists or changed: {destination}. Use --overwrite to regenerate it.")
    models = models or Models(settings)
    LOG.info("%s: %.2fs, %s transcript rows", folder.name, recording.duration, len(rows))
    vad = models.vad(recording)
    words, gaps, blocks = [], [], []
    if not vad_only:
        for block_rows, start, end in alignment_blocks(rows):
            LOG.info("Aligning %s rows in absolute %.3f–%.3fs", len(block_rows), start, end)
            block_words, block_gaps = models.align(recording, block_rows, start, end,
                                                   output_root / ".cache", hashes["audio"])
            words.extend(block_words)
            gaps.extend(block_gaps)
            blocks.append({"start": start, "end": end, "rows": [r.key for r in block_rows]})
        records = sentence_records(rows, words, vad, settings, recording)
    else:
        records = []
    report = {"folder": folder.name, "fingerprint": fingerprint,
              "created_utc": datetime.now(timezone.utc).isoformat(), "pipeline_version": __version__,
              "pipeline_code_sha256": source_hash,
              "python_version": platform.python_version(), "input_audio": wav.name,
              "input_workbook": workbook.name, "input_hashes": hashes,
              "duration_seconds": recording.duration, "sample_rate": recording.sample_rate,
              "channels": recording.samples.shape[1], "settings": settings, "vad_only": vad_only,
              "transcript_rows": len(rows), "sentence_clips": sum(bool(r["Path"]) for r in records),
              "vad_clips": len(vad), "needs_review": sum(r["needs_review"] for r in records),
              "flagged_rows": sum(bool(r["flags"]) and r["status"] != "skipped" for r in records),
              "alignment_blocks": blocks,
              "unlabelled_speech_seconds": sum(x["end"] - x["start"] for x in uncovered_speech(vad, records)),
              "limitations": ["Acoustic scores are not calibrated correctness probabilities.",
                              "Every new ML alignment is auto_unverified until reviewed.",
                              "Provided source timestamps are reused, not independently validated.",
                              "VAD chunks are speech activity regions, not sentence labels."],
              "model": "torchaudio.pipelines.MMS_FA (CC-BY-NC-4.0)",
              "vad_model": "silero-vad 6.0.0 (MIT)"}
    # Stage a complete folder. Failed reruns preserve the previous output.
    temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=output_root))
    try:
        if vad_only:
            vad_records = []
            for i, segment in enumerate(vad, 1):
                path = f"vad_chunks/vad_{i:04d}.wav"
                start, end = audio.write_clip(recording, temporary / path, segment["start"], segment["end"])
                vad_records.append({"path": path, "start_time": start, "end_time": end, "duration": end - start})
            write_csv(temporary / "vad_manifest.csv", vad_records, ["path", "start_time", "end_time", "duration"])
        else:
            export_all(temporary, folder.name, recording, workbook, rows, words, vad, gaps, records, report)
        report["output_files"] = sorted(str(p.relative_to(temporary)).replace("\\", "/")
                                        for p in temporary.rglob("*") if p.is_file() and p.name != "report.json")
        dump_json(temporary / "report.json", report)
        backup = destination.with_name(".backup-" + destination.name)
        if backup.exists():
            raise RuntimeError(f"Previous interrupted replacement backup exists: {backup}. Inspect it before retrying.")
        if destination.exists():
            destination.rename(backup)
        try:
            temporary.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    LOG.info("Wrote %s sentence clips + %s VAD clips to %s", report["sentence_clips"], len(vad), destination)
    return report


def benchmark(folder, output_root, config, settings, models, overwrite=False):
    folder = Path(folder)
    options = config.get("folders", {}).get(folder.name, {})
    wav, workbook = find_inputs(folder, options)
    recording = audio.load(wav)
    truth = read_rows(workbook, recording.duration)
    if not any(r.start is not None for r in truth):
        raise ValueError("Benchmark needs existing sentence timestamps, e.g. the completed 16-07 reference")
    # Exclude time/skip/scope overrides: benchmark must not leak labels into alignment.
    benchmark_config = {"folders": {folder.name: {k: v for k, v in options.items() if k in ("audio", "workbook")}}}
    report = run_one(folder, Path(output_root) / "benchmarks", benchmark_config, settings, models,
                     overwrite=overwrite, ignore_existing=True)
    destination = Path(output_root) / "benchmarks" / folder.name
    predictions = json.loads((destination / "alignment.json").read_text(encoding="utf-8"))["sentences"]
    by_key = {f"{r['sheet']}:{r['source_row']}": r for r in predictions}
    comparison = []
    for row in truth:
        if row.start is None:
            continue
        p = by_key[row.key]
        comparison.append({"sheet": row.sheet, "row": row.row, "reference_start": row.start,
                           "reference_end": row.end, "predicted_start": p["start_time"],
                           "predicted_end": p["end_time"],
                           "start_error": p["start_time"] - row.start,
                           "end_error": p["end_time"] - row.end,
                           "flags": p["flags"]})
    import numpy as np
    errors = np.array([abs(r[k]) for r in comparison for k in ("start_error", "end_error")])
    metrics = {"folder": folder.name, "rows": len(comparison), "boundary_count": len(errors),
               "mean_absolute_boundary_error_seconds": float(errors.mean()),
               "median_absolute_boundary_error_seconds": float(np.median(errors)),
               "p90_absolute_boundary_error_seconds": float(np.quantile(errors, 0.9)),
               "max_absolute_boundary_error_seconds": float(errors.max()),
               "fraction_boundaries_within_0_5_seconds": float((errors <= 0.5).mean()),
               "warning": "One recording is not validation on the remaining speakers/topics. "
                          "References include clip padding; predictions here also include configured padding."}
    dump_json(destination / "benchmark.json", metrics)
    write_csv(destination / "benchmark_rows.csv", comparison)
    return metrics

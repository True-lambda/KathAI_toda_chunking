from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from . import audio
from .ingest import find_inputs, read_rows
from .ml import Models
from .pipeline import configuration, run_one, benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kathai Toda: transcript alignment + separate VAD audio clips")
    parser.add_argument("command", choices=["inspect", "run", "benchmark"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=None, help="Default: <data-root>/output")
    parser.add_argument("--folder", help="Exact immediate subfolder name under data-root")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--corrections", type=Path, help="CSV exported from review.html; requires --folder")
    parser.add_argument("--vad-only", action="store_true")
    parser.add_argument("--strict-review", action="store_true", help="Return exit code 2 if auto_unverified rows remain")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config, settings = configuration(args.config, args.device)
        if not args.data_root.is_dir():
            raise ValueError(f"Missing data root: {args.data_root}. Place recording folders under data/.")
        output = args.output_root or args.data_root / "output"
        if output.resolve() == args.data_root.resolve():
            raise ValueError("output-root must not equal data-root")
        if args.corrections and not args.folder:
            raise ValueError("--corrections requires --folder to identify the recording")
        if args.folder:
            if Path(args.folder).name != args.folder or args.folder in {".", "..", "output"}:
                raise ValueError("--folder must be one immediate recording folder name")
            folders = [args.data_root / args.folder]
        else:
            folders = sorted(p for p in args.data_root.iterdir()
                             if p.is_dir() and not p.name.startswith(".") and p.name != "output"
                             and p.resolve() != output.resolve())
        if not folders:
            raise ValueError("No recording folders under data-root")
        if args.command == "benchmark" and len(folders) != 1:
            raise ValueError("benchmark requires --folder naming the completed reference")
        models = Models(settings)
        results, failures = [], []
        for folder in folders:
            try:
                if not folder.is_dir() or folder.is_symlink():
                    raise ValueError(f"Not a regular recording folder: {folder}")
                if folder.resolve() == output.resolve() or output.resolve().is_relative_to(folder.resolve()):
                    raise ValueError("Output must not be inside a source recording folder")
                if args.command == "inspect":
                    wav, workbook = find_inputs(folder, config.get("folders", {}).get(folder.name, {}))
                    recording = audio.load(wav)
                    rows = read_rows(workbook, recording.duration, config.get("folders", {}).get(folder.name, {}).get("rows", []))
                    results.append({"folder": folder.name, "duration_seconds": recording.duration,
                                    "sample_rate": recording.sample_rate, "rows": [r.to_dict() for r in rows]})
                elif args.command == "benchmark":
                    results.append(benchmark(folder, output, config, settings, models, args.overwrite))
                else:
                    report = run_one(folder, output, config, settings, models, overwrite=args.overwrite,
                                     corrections=args.corrections, vad_only=args.vad_only)
                    results.append({k: report[k] for k in ("folder", "sentence_clips", "vad_clips", "needs_review", "flagged_rows")}
                                   | {"output": str(output / folder.name)})
            except Exception as e:
                logging.error("%s: %s", folder.name, e)
                failures.append({"folder": folder.name, "error": str(e)})
        print(json.dumps({"results": results, "failures": failures}, ensure_ascii=False, indent=2))
        if failures:
            return 1
        if args.strict_review and any(r.get("needs_review", 0) for r in results):
            return 2
        return 0
    except (ValueError, OSError) as e:
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

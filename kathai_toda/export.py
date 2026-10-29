"""Portable exports and surgical OOXML annotations without rewriting source cells."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import posixpath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .audio import write_clip
from .ingest import header_key

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKGREL = "http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("", NS)
ET.register_namespace("r", REL)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, records, fields=None):
    fields = fields or list(records[0]) if records else fields or []
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            # Source text must remain text when CSV is opened in Excel.
            safe = {k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
                    for k, v in record.items()}
            writer.writerow(safe)


def overlap(start, end, intervals):
    return sum(max(0.0, min(end, x["end"]) - max(start, x["start"])) for x in intervals)


def uncovered_speech(vad, sentences, min_seconds=0.3):
    """VAD portions not covered by any labelled clip. Includes speech outside coarse scopes."""
    coverage = sorted((x["start_time"], x["end_time"]) for x in sentences if x.get("start_time") is not None)
    result = []
    for segment in vad:
        cursor = segment["start"]
        for start, end in coverage:
            if end <= cursor:
                continue
            if start >= segment["end"]:
                break
            if start - cursor >= min_seconds:
                result.append({"start": cursor, "end": min(start, segment["end"]), "reason": "unlabelled_speech"})
            cursor = max(cursor, end)
        if segment["end"] - cursor >= min_seconds:
            result.append({"start": cursor, "end": segment["end"], "reason": "unlabelled_speech"})
    return result


def sentence_records(rows, words, vad, settings, audio):
    by_key = defaultdict(list)
    for word in words:
        by_key[word["key"]].append(word)
    result = []
    for i, row in enumerate(rows, 1):
        record = {"S.no": i, "Chunk ID": f"chunk_{i:04d}", "Path": "", "start_time": None,
                  "end_time": None, "duration": None, "Romanized Script": row.transcript,
                  "English Translation": row.translation, "Word level (roman script -> english translation)": row.gloss,
                  "sheet": row.sheet, "source_row": row.row, "group": row.group,
                  "scope_start": row.scope_start, "scope_end": row.scope_end,
                  "alignment_text": row.alignment_text if row.alignment_text is not None else row.transcript,
                  "speech_start": None, "speech_end": None, "acoustic_score": None,
                  "status": "skipped" if row.skip else "auto_unverified", "needs_review": not row.skip,
                  "flags": "", "vad_overlap_ratio": None}
        flags = list(row.flags)
        if row.skip:
            record["flags"] = "explicitly_skipped"
            result.append(record)
            continue
        aligned = by_key[row.key]
        if row.start is not None:
            start, end = row.start, row.end
            record["status"] = "human_reviewed" if row.manual else "provided_timestamps"
            record["needs_review"] = False  # Source times trusted operationally; not newly validated.
        else:
            if not aligned:
                raise ValueError(f"No alignment for {row.key}")
            start, end = aligned[0]["start"], aligned[-1]["end"]
            record["speech_start"], record["speech_end"] = start, end
            score = float(np.mean([w["score"] for w in aligned]))
            record["acoustic_score"] = round(score, 6)
            if score < settings["low_score_threshold"]:
                flags.append("low_acoustic_score")
            chars = sum(len(w["word"]) for w in aligned)
            speed = chars / max(0.01, end - start)
            if speed > 35 or speed < 1:
                flags.append("implausible_character_rate")
            if end - start > 30:
                flags.append("long_sentence_check_repetitions")
            if any(b["start"] - a["end"] > 1.5 for a, b in zip(aligned, aligned[1:])):
                flags.append("large_internal_word_gap")
            # Only expand toward a nearby VAD edge. Do not cut a transcript at every pause.
            close_starts = [x["start"] for x in vad if 0 <= start - x["start"] <= settings["snap_seconds"]]
            close_ends = [x["end"] for x in vad if 0 <= x["end"] - end <= settings["snap_seconds"]]
            start = min(close_starts + [start]) - settings["padding_seconds"]
            end = max(close_ends + [end]) + settings["padding_seconds"]
            start, end = max(row.scope_start, start), min(row.scope_end, end)
        record["start_time"], record["end_time"] = start, end
        ratio = min(1.0, overlap(start, end, vad) / max(0.001, end - start))
        record["vad_overlap_ratio"] = round(ratio, 4)
        if ratio < 0.5:
            flags.append("little_detected_speech")
        record["flags"] = ";".join(dict.fromkeys(flags))
        result.append(record)
    # Padding must not duplicate neighboring utterances. Raw word times remain unchanged.
    active = [r for r in result if r["start_time"] is not None]
    for prev, cur in zip(active, active[1:]):
        if cur["start_time"] < prev["start_time"]:
            raise ValueError("Transcript times are not chronological; check manual/source times and scopes")
        if prev["end_time"] > cur["start_time"]:
            prev_fixed = prev["status"] != "auto_unverified"
            cur_fixed = cur["status"] != "auto_unverified"
            if prev_fixed and cur_fixed:
                raise ValueError("Overlapping source/manual sentence times; correct them explicitly")
            lo = prev["speech_end"] if not prev_fixed else prev["end_time"]
            hi = cur["speech_start"] if not cur_fixed else cur["start_time"]
            if lo > hi:
                raise ValueError("Overlapping acoustic/source sentence boundaries")
            boundary = (lo + hi) / 2
            if not prev_fixed:
                prev["end_time"] = min(prev["end_time"], boundary if not cur_fixed else cur["start_time"])
            if not cur_fixed:
                cur["start_time"] = max(cur["start_time"], boundary if not prev_fixed else prev["end_time"])
    for r in active:
        # Quantize manifests to the exact sample boundaries used by the audio files.
        rate = audio.sample_rate
        r["start_time"] = round(r["start_time"] * rate) / rate
        r["end_time"] = round(r["end_time"] * rate) / rate
        if r["end_time"] <= r["start_time"]:
            raise ValueError(f"Empty sentence clip at {r['sheet']}:{r['source_row']}")
        r["duration"] = r["end_time"] - r["start_time"]
        r["Path"] = f"chunks/{r['Chunk ID']}.wav"
    return result


def col_name(index):
    name = ""
    while index:
        index, r = divmod(index - 1, 26)
        name = chr(65 + r) + name
    return name


def annotate_workbook(source, target, records):
    """Copy XLSX archive; patch only timestamps, clip links and new pipeline columns.

    The XLSX is a generated codebase output. All transcript/gloss XML and other ZIP
    members are retained. No alternate workbook authoring dependency is required.
    """
    by_sheet = defaultdict(list)
    for record in records:
        by_sheet[record["sheet"]].append(record)
    with zipfile.ZipFile(source) as src:
        strings = []
        if "xl/sharedStrings.xml" in src.namelist():
            tree = ET.fromstring(src.read("xl/sharedStrings.xml"))
            strings = ["".join(x.itertext()) for x in tree]
        workbook = ET.fromstring(src.read("xl/workbook.xml"))
        rels = ET.fromstring(src.read("xl/_rels/workbook.xml.rels"))
        mapping = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        changes = {}
        for sheet in workbook.find(f"{{{NS}}}sheets"):
            name = sheet.attrib["name"]
            if name not in by_sheet:
                continue
            rel_target = mapping[sheet.attrib[f"{{{REL}}}id"]]
            member = rel_target.lstrip("/") if rel_target.startswith("/") else posixpath.normpath("xl/" + rel_target)
            tree = ET.fromstring(src.read(member))
            data = tree.find(f"{{{NS}}}sheetData")
            row_map = {int(r.attrib["r"]): r for r in data}
            def text(c):
                v = c.find(f"{{{NS}}}v")
                if c.get("t") == "s" and v is not None:
                    return strings[int(v.text)]
                if c.get("t") == "inlineStr":
                    return "".join(c.find(f"{{{NS}}}is").itertext())
                return v.text if v is not None else ""
            header = next(r for r in data if any(header_key(text(c)) == "romanizedscript" for c in r))
            columns = {header_key(text(c)): c.attrib["r"].rstrip("0123456789") for c in header}
            max_col = max((column_number(c.attrib["r"].rstrip("0123456789")) for r in data for c in r), default=0)
            def put(row, column, value, formula=False):
                coordinate = f"{column}{row.attrib['r']}"
                c = next((c for c in row if c.get("r") == coordinate), None)
                if c is None:
                    c = ET.SubElement(row, f"{{{NS}}}c", {"r": coordinate})
                for child in list(c):
                    c.remove(child)
                c.attrib.pop("t", None)
                c.set("s", "0")  # Explicit seconds, not Excel's fractional-day time display.
                if value is None:
                    return
                if formula:
                    ET.SubElement(c, f"{{{NS}}}f").text = value
                elif isinstance(value, (int, float, bool)):
                    ET.SubElement(c, f"{{{NS}}}v").text = str(int(value) if isinstance(value, bool) else value)
                else:
                    c.set("t", "inlineStr")
                    ET.SubElement(ET.SubElement(c, f"{{{NS}}}is"), f"{{{NS}}}t").text = str(value)
            def ensure(label, alternate=None):
                nonlocal max_col
                key = header_key(label)
                if alternate and alternate in columns:
                    return columns[alternate]
                if key not in columns:
                    max_col += 1
                    columns[key] = col_name(max_col)
                    put(header, columns[key], label)
                return columns[key]
            fields = {"start_time": ensure("Start time for sentences", "starttime"),
                      "end_time": ensure("End time for sentences", "endtime"),
                      "Path": ensure("Pipeline chunk path"), "status": ensure("Pipeline status"),
                      "flags": ensure("Pipeline review flags"), "acoustic_score": ensure("Acoustic score"),
                      "alignment_text": ensure("Alignment text used")}
            changed_cells = set()
            for record in by_sheet[name]:
                row = row_map[record["source_row"]]
                for field, column in fields.items():
                    put(row, column, record[field])
                if "duration" in columns:
                    put(row, columns["duration"], record["duration"])
                if "chunkid" in columns:
                    put(row, columns["chunkid"], record["Chunk ID"])
                if "path" in columns:
                    column = columns["path"]
                    put(row, column, f'HYPERLINK("{record["Path"]}","Open chunk")' if record["Path"] else None,
                        formula=bool(record["Path"]))
                    changed_cells.add(f"{column}{record['source_row']}")
            links = tree.find(f"{{{NS}}}hyperlinks")
            if links is not None:
                for link in list(links):
                    if link.attrib.get("ref") in changed_cells:
                        links.remove(link)
            for row in data:
                row[:] = sorted(row, key=lambda c: column_number(c.attrib["r"].rstrip("0123456789")))
            dimension = tree.find(f"{{{NS}}}dimension")
            if dimension is not None:
                dimension.set("ref", f"A1:{col_name(max_col)}{max(row_map)}")
            changes[member] = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
            for item in src.infolist():
                out.writestr(item, changes.get(item.filename, src.read(item.filename)))


def column_number(name):
    total = 0
    for c in name:
        total = total * 26 + ord(c) - 64
    return total


def srt_time(t):
    ms = round(t * 1000)
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def export_all(out, folder, audio, workbook, rows, words, vad, gaps, records, report):
    for record in records:
        if record["Path"]:
            write_clip(audio, out / record["Path"], record["start_time"], record["end_time"])
    vad_records = []
    for i, segment in enumerate(vad, 1):
        path = f"vad_chunks/vad_{i:04d}.wav"
        start, end = write_clip(audio, out / path, segment["start"], segment["end"])
        vad_records.append({"chunk_id": f"vad_{i:04d}", "path": path, "start_time": start,
                            "end_time": end, "duration": end - start, "label": "unlabelled_speech_activity"})
    write_csv(out / "sentences.csv", records)
    write_csv(out / f"{folder}_vad_manifest.csv", records)
    write_csv(out / "vad_manifest.csv", vad_records,
              ["chunk_id", "path", "start_time", "end_time", "duration", "label"])
    write_csv(out / "words.csv", words, ["key", "word", "start", "end", "score"])
    write_csv(out / "review_queue.csv", [r for r in records if r["needs_review"]], list(records[0]))
    write_csv(out / "unlabelled_speech.csv", uncovered_speech(vad, records), ["start", "end", "reason"])
    dump_json(out / "alignment.json", {"sentences": records, "words": words, "wildcard_gaps": gaps, "vad": vad})
    dump_json(out / "report.json", report)
    annotate_workbook(workbook, out / "timestamped_transcription.xlsx", records)
    for language, field in (("toda", "Romanized Script"), ("english", "English Translation")):
        active = [r for r in records if r["Path"]]
        content = "\n\n".join(f"{i}\n{srt_time(r['start_time'])} --> {srt_time(r['end_time'])}\n{r[field]}"
                              for i, r in enumerate(active, 1))
        (out / f"{language}.srt").write_text(content + "\n", encoding="utf-8")
    # A local full recording permits hearing context even when a predicted clip is wrong.
    write_clip(audio, out / "review_audio.wav", 0, audio.duration)
    mono = audio.samples.mean(axis=1)
    stride = max(1, len(mono) // 5000)
    envelope = [float(np.max(np.abs(mono[i:i + stride]))) for i in range(0, len(mono), stride)]
    payload = json.dumps({"folder": folder, "duration": audio.duration, "sentences": records,
                          "envelope": envelope}, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("review_template.html").read_text(encoding="utf-8")
    (out / "review.html").write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")

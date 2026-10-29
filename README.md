# Kathai Toda alignment pipeline

Put a recording folder under `data/`, run one command, and find its timestamps and audio clips under `data/output/<foldername>/`. Repeat with the next folder.

## 1. Install

Use **Python 3.11 or 3.12** in a separate environment. The CPU setup is the default and is suitable for a 16 GB RAM laptop. No API key, paid service, FFmpeg installation, or Hugging Face login is required for the supplied PCM WAV files.

Windows PowerShell, from this repository's root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. folders
Place one folder in data/

## 3. Run

Windows:

```powershell
.\.venv\Scripts\python.exe -m kathai_toda inspect
.\.venv\Scripts\python.exe -m kathai_toda run
```

`inspect` reads the workbook and prints the detected rows, coarse windows, and existing times without loading models. `run` processes every immediate recording folder under `data/` except `output/`. For a single folder:

```bash
python -m kathai_toda run --folder "06-07-2026 EMRS ENGLISH TODA RECAP"
```

After a run, you can move that source folder out, put the next folder in `data/`, and run the same command. Old outputs remain under their own names. Unchanged inputs with complete outputs are skipped on rerun. Changed inputs or corrections require `--overwrite`, which replaces **only that recording's output folder** after a successful staged run.

## 4. Use the outputs

Each `data/output/<foldername>/` contains:

| File or folder | Meaning |
|---|---|
| `timestamped_transcription.xlsx` | A copy with sentence start/end columns populated, status/flags added, original transcript/translation/gloss retained |
| `sentences.csv` | Main manifest, one row per transcript row, absolute times and relative clip path |
| `<foldername>_vad_manifest.csv` | Sentence manifest with the reference's leading six columns; this historical filename is kept for compatibility |
| `chunks/chunk_0001.wav`, etc. | One sentence clip per non-skipped spreadsheet row, matching the reference naming convention |
| `vad_manifest.csv` | Raw speech-activity intervals, without transcript labels |
| `vad_chunks/vad_0001.wav`, etc. | Separate VAD audio chunks from the full original recording |
| `words.csv` | Normalized alignment words and raw acoustic times for newly aligned rows |
| `alignment.json` | Sentence records, word timings, wildcard gaps, and VAD intervals |
| `review_queue.csv` | All newly inferred, unverified sentence rows |
| `unlabelled_speech.csv` | Detected speech not covered by labelled clips, including outside selected coarse windows |
| `review.html` + `review_audio.wav` | Offline listening, time editing, and correction export |
| `toda.srt`, `english.srt` | Subtitles using the corresponding source text and the same sentence intervals |
| `report.json` | Input hashes, settings, model/version provenance, counts, output inventory and limitations |

`start_time` and `end_time` refer to the **full original recording**, in seconds. They match the exact audio sample boundaries exported. `speech_start`/`speech_end` retain raw acoustic word boundaries before padding/VAD adjustment. Sentence clips retain the original sample rate and channels and use 16-bit PCM WAV. The model receives a separate 16 kHz mono view.

## Reference benchmark

Put the completed reference folder under `data/`, then run:

```bash
python -m kathai_toda benchmark --folder "<foldername>"
```

This hides all existing sentence timestamps from the aligner and compares its predictions against them afterward. It also excludes row/time overrides from the benchmark so those labels cannot leak into alignment. Results, row errors, chunks, and the review page go to `data/output/benchmarks/<foldername>/`. Repeating a benchmark after a relevant code/config change requires `--overwrite`.


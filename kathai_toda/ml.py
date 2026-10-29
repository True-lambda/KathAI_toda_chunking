"""Pinned, local ML inference. No ASR or translation generation."""
from __future__ import annotations

from pathlib import Path
import hashlib
import logging
import json

import numpy as np

from .ingest import normalize
from .ctc import align as ctc_align

LOG = logging.getLogger(__name__)


class Models:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.vad_model = None

    def runtime(self):
        try:
            import torch
            import torchaudio
        except (ImportError, OSError) as e:
            raise RuntimeError("Install the matching torch==2.7.1 and torchaudio==2.7.1 wheels from README. "
                               f"Original error: {e}") from e
        if torch.__version__.split("+")[0] != "2.7.1" or torchaudio.__version__.split("+")[0] != "2.7.1":
            raise RuntimeError("This backend requires the pinned torch/torchaudio 2.7.1 MMS_FA bundle.")
        torch.set_num_threads(int(self.settings["threads"]))
        return torch, torchaudio

    def vad(self, audio):
        torch, _ = self.runtime()
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
        except ImportError as e:
            raise RuntimeError("Install silero-vad==6.0.0 (pip install -r requirements.txt).") from e
        if self.vad_model is None:
            self.vad_model = load_silero_vad(onnx=False)
        ts = get_speech_timestamps(
            torch.from_numpy(audio.mono16()), self.vad_model, sampling_rate=16000,
            threshold=self.settings["vad_threshold"],
            min_speech_duration_ms=150, min_silence_duration_ms=self.settings["vad_silence_ms"],
            speech_pad_ms=0, max_speech_duration_s=self.settings["vad_max_seconds"],
            return_seconds=False,
        )
        return [{"start": x["start"] / 16000, "end": min(audio.duration, x["end"] / 16000)} for x in ts]

    def emissions(self, audio, start, end, cache_dir: Path, audio_hash):
        torch, torchaudio = self.runtime()
        # Do not mix frame caches across inference parameters or backend versions.
        signature = json.dumps(["mms-fa-2.7.1-grid-v1", audio_hash, start, end,
                                self.settings["window_seconds"], self.settings["context_seconds"],
                                self.settings["device"]])
        key = hashlib.sha256(signature.encode()).hexdigest()
        path = cache_dir / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                return f["log_probs"], f["centers"]
        device = self.settings["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")
        if self.model is None:
            LOG.info("Loading MMS_FA (315M parameters; first use downloads ~1.2 GB)")
            self.model = torchaudio.pipelines.MMS_FA.get_model(with_star=False).to(device).eval()
        waveform = audio.mono16(start, end)
        # MMS convolution: receptive field 400 samples, stride 320, center 199.5.
        # Window starts and context align to the same stride grid. No VAD time compression.
        core = max(320, round(self.settings["window_seconds"] * 16000 / 320) * 320)
        context = max(320, round(self.settings["context_seconds"] * 16000 / 320) * 320)
        emissions, centers = [], []
        for i in range(0, len(waveform), core):
            j = min(len(waveform), i + core)
            lo, hi = max(0, i - context), min(len(waveform), j + context)
            if hi - lo < 400:
                continue
            LOG.info("Acoustic inference %.1f–%.1fs", start + i / 16000, start + j / 16000)
            try:
                with torch.inference_mode():
                    logits, _ = self.model(torch.from_numpy(waveform[lo:hi]).unsqueeze(0).to(device))
                    logp = logits[0].log_softmax(-1).cpu().numpy()
            except torch.OutOfMemoryError as e:
                raise RuntimeError("GPU memory exhausted. Rerun --device cpu; or reduce window_seconds.") from e
            frame_centers = lo + 199.5 + np.arange(len(logp)) * 320
            keep = (frame_centers >= i) & (frame_centers < j)
            emissions.append(logp[keep])
            centers.append(start + frame_centers[keep] / 16000)
        if not emissions:
            raise ValueError("Alignment window is too short for the acoustic model")
        logp, times = np.concatenate(emissions), np.concatenate(centers)
        if len(times) > 1 and np.max(np.abs(np.diff(times) - 0.02)) > 1e-5:
            raise RuntimeError("Unexpected MMS frame grid; refusing to emit shifted timestamps")
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, log_probs=logp, centers=times)
        temp.replace(path)
        return logp, times

    def align(self, audio, rows, start, end, cache_dir, audio_hash):
        torch, torchaudio = self.runtime()
        logp, centers = self.emissions(audio, start, end, cache_dir, audio_hash)
        normalized = []
        for row in rows:
            words, flags = normalize(row.alignment_text if row.alignment_text is not None else row.transcript)
            row.flags.extend(flags)
            normalized.append(words)
        for i, row in enumerate(rows):
            if any(flag in row.flags for flag in ("numbers_need_spoken_form", "alternative_or_slash_in_transcript",
                                                  "parenthetical_text_may_not_be_spoken", "possible_unspoken_heading")):
                for j in (i - 1, i + 1):
                    if 0 <= j < len(rows):
                        rows[j].flags.append("adjacent_transcript_ambiguity")
        dictionary = torchaudio.pipelines.MMS_FA.get_dict(star=None)
        star = logp.shape[1]
        # A penalized wildcard between rows can absorb extra utterances/repetitions.
        # It never permits omitting supplied letters. Missing transcript words still need review.
        speech_gap = np.logaddexp.reduce(logp[:, 1:], axis=1) - self.settings["gap_penalty"]
        wildcard = np.logaddexp(logp[:, 0], speech_gap)
        emission = np.column_stack([logp, wildcard]).astype(np.float32)
        words_flat, owners = ["*"], [None]
        for row, words in zip(rows, normalized):
            words_flat.extend(words)
            owners.extend([row.key] * len(words))
            words_flat.append("*")
            owners.append(None)
        token_lists = [[star] if w == "*" else [dictionary[c] for c in w] for w in words_flat]
        targets = [c for word in token_lists for c in word]
        repeats = sum(a == b for a, b in zip(targets, targets[1:]))
        if len(targets) + repeats > len(emission):
            raise ValueError("Transcript cannot fit this time window. Check coarse times and spoken text.")
        try:
            spans = ctc_align(emission, targets, blank=0, max_cells=self.settings["max_alignment_cells"])
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"CTC alignment failed for {start}–{end}s: {e}") from e
        if [int(s.token) for s in spans] != targets:
            raise RuntimeError("CTC token accounting mismatch; refusing to fabricate word times")
        aligned, gaps, index = [], [], 0
        for word, owner, tokens in zip(words_flat, owners, token_lists):
            parts = spans[index:index + len(tokens)]
            index += len(tokens)
            left = max(start, float(centers[parts[0].start] - 0.01))
            right = min(end, float(centers[parts[-1].end - 1] + 0.01))
            if owner is None:
                if right - left >= 0.3:
                    gaps.append({"start": left, "end": right, "reason": "wildcard_untranscribed_audio"})
            else:
                # Scores quantify acoustic token fit, not P(correct sentence boundary).
                score = sum(float(s.score) * (s.end - s.start) for s in parts) / sum(s.end - s.start for s in parts)
                aligned.append({"key": owner, "word": word, "start": left, "end": right, "score": score})
        return aligned, gaps

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import gcd
import wave

import numpy as np
from scipy.signal import resample_poly


@dataclass
class Audio:
    samples: np.ndarray  # [samples, channels], original rate
    sample_rate: int

    @property
    def duration(self):
        return len(self.samples) / self.sample_rate

    def mono16(self, start=0.0, end=None):
        i, j = sample_bounds(start, self.duration if end is None else end, self.sample_rate, len(self.samples))
        mono = self.samples[i:j].mean(axis=1)
        if self.sample_rate != 16000:
            g = gcd(self.sample_rate, 16000)
            mono = resample_poly(mono, 16000 // g, self.sample_rate // g)
        return np.ascontiguousarray(mono, dtype=np.float32)


def sample_bounds(start, end, rate, length):
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise ValueError(f"Invalid audio interval {start}, {end}")
    i, j = max(0, round(start * rate)), min(length, round(end * rate))
    if j <= i:
        raise ValueError(f"Empty/out-of-range audio interval {start}, {end}")
    return i, j


def load(path: Path):
    try:
        with wave.open(str(path), "rb") as f:
            if f.getcomptype() != "NONE":
                raise ValueError("Only uncompressed PCM WAV is supported")
            rate, channels, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
            raw = f.readframes(f.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"Cannot read {path.name}: {e}. Convert to PCM WAV first.") from e
    if width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width in (2, 4):
        data = np.frombuffer(raw, dtype=f"<i{width}").astype(np.float32) / 2 ** (8 * width - 1)
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        v = (v ^ 0x800000) - 0x800000
        data = v.astype(np.float32) / 8388608
    else:
        raise ValueError(f"Unsupported PCM width {width}")
    if not len(data):
        raise ValueError("Empty recording")
    return Audio(data.reshape(-1, channels), rate)


def write_clip(audio: Audio, path: Path, start, end):
    """Keep original rate/channels; export portable 16-bit PCM, with no concatenation."""
    i, j = sample_bounds(start, end, audio.sample_rate, len(audio.samples))
    data = (np.clip(audio.samples[i:j], -1, 32767 / 32768) * 32768).round().astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(audio.samples.shape[1])
        f.setsampwidth(2)
        f.setframerate(audio.sample_rate)
        f.writeframes(data.tobytes())
    return i / audio.sample_rate, j / audio.sample_rate

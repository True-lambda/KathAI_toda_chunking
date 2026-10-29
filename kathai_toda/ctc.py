"""CTC Viterbi alignment with explicit transcript-position backtracking.

The state path, not collapsed token identities, owns each character. This keeps
repeated letters and repeated wildcard symbols distinct. NumPy vectorizes states;
the frame loop stores one byte per state for backtracking.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Span:
    token: int
    start: int
    end: int
    score: float


def align(log_probs, targets, *, blank=0, max_cells=100_000_000):
    log_probs = np.asarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    if log_probs.ndim != 2 or not len(log_probs) or not len(targets):
        raise ValueError("Nonempty [frames, labels] emissions and target tokens required")
    if not np.isfinite(log_probs).all():
        raise ValueError("Emissions contain non-finite values")
    if np.any(targets == blank) or np.any(targets < 0) or np.any(targets >= log_probs.shape[1]):
        raise ValueError("Targets must contain valid nonblank token IDs")
    frames = len(log_probs)
    states = 2 * len(targets) + 1
    if frames * states > max_cells:
        raise ValueError("Alignment exceeds max_alignment_cells; add checked sentence anchors")
    if len(targets) + np.sum(targets[1:] == targets[:-1]) > frames:
        raise ValueError("Transcript has more required CTC steps than audio frames")
    labels = np.full(states, blank, dtype=np.int64)
    labels[1::2] = targets
    can_skip = np.zeros(states, bool)
    can_skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    previous = np.full(states, -np.inf, np.float32)
    previous[0] = 0
    back = np.empty((frames, states), dtype=np.uint8)
    options = np.empty((3, states), dtype=np.float32)
    for t in range(frames):
        options[0] = previous
        options[1, 0] = -np.inf
        options[1, 1:] = previous[:-1]
        options[2, :2] = -np.inf
        options[2, 2:] = previous[:-2]
        options[2, ~can_skip] = -np.inf
        moves = np.argmax(options, axis=0).astype(np.uint8)
        previous = np.take_along_axis(options, moves[None, :], axis=0)[0] + log_probs[t, labels]
        back[t] = moves
    state = states - 2 if previous[-2] >= previous[-1] else states - 1
    if not np.isfinite(previous[state]):
        raise ValueError("No valid CTC alignment path")
    path = np.empty(frames, np.int32)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    if state != 0:
        raise RuntimeError("Invalid CTC backtracking origin")
    # Monotonic state path makes each token's frame span contiguous.
    starts = np.searchsorted(path, 2 * np.arange(len(targets)) + 1, side="left")
    ends = np.searchsorted(path, 2 * np.arange(len(targets)) + 1, side="right")
    spans = []
    for token, start, end in zip(targets, starts, ends):
        if start == end:
            raise RuntimeError("A required transcript character was not aligned")
        spans.append(Span(int(token), int(start), int(end), float(np.exp(log_probs[start:end, token]).mean())))
    return spans

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

def forced_align(log_probs: np.ndarray, targets: Sequence[int],
                 blank: int = 0) -> List[Tuple[int, int, int]]:
    """
    Viterbi alignment of `targets` to `log_probs` under the CTC topology.

    log_probs : (T, V) log-softmax from the acoustic model
    targets   : token ids, no blanks
    returns   : [(token_id, start_frame, end_frame)] with end exclusive

    This is what makes the whole product work. Because the reference text is
    known (the user told us which ayah they are reciting, or we retrieved it),
    we never run open-vocabulary decoding. Forced alignment is O(T * 2L), it is
    exact, and it hands us the frame boundaries that the tajweed grader needs.
    Free decoding would give us neither the boundaries nor the robustness.
    """
    T, V = log_probs.shape
    L = len(targets)
    if L == 0:
        raise RuntimeError("empty target")
    # extended sequence: blank, t0, blank, t1, ..., blank
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = np.asarray(targets, dtype=np.int64)
    S = len(ext)
    if T < L:
        raise RuntimeError(f"audio too short: {T} frames for {L} tokens")

    NEG = -1e30
    dp = np.full((T, S), NEG, dtype=np.float32)
    bp = np.zeros((T, S), dtype=np.int8)       # 0 stay, 1 from s-1, 2 from s-2

    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        emit = log_probs[t, ext]                       # (S,)
        stay = dp[t - 1]
        prev1 = np.concatenate(([NEG], dp[t - 1, :-1]))
        # skip is only legal into a non-blank whose predecessor label differs
        prev2 = np.concatenate(([NEG, NEG], dp[t - 1, :-2]))
        allow_skip = np.zeros(S, dtype=bool)
        allow_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
        prev2 = np.where(allow_skip, prev2, NEG)

        cand = np.stack([stay, prev1, prev2])           # (3, S)
        best = np.argmax(cand, axis=0)
        dp[t] = cand[best, np.arange(S)] + emit
        bp[t] = best.astype(np.int8)

    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])

    # collapse to per-target-token spans
    out: List[Tuple[int, int, int]] = []
    for i in range(L):
        s_idx = 2 * i + 1
        frames = np.nonzero(path == s_idx)[0]
        if len(frames) == 0:
            # token absorbed entirely by blanks: zero-width span at its position
            prev_end = out[-1][2] if out else 0
            out.append((int(targets[i]), prev_end, prev_end))
        else:
            out.append((int(targets[i]), int(frames[0]), int(frames[-1]) + 1))
    return out


def greedy_decode(log_probs: np.ndarray, blank: int = 0) -> List[int]:
    """Best-path CTC decode, for the ayah-identification stage."""
    best = log_probs.argmax(axis=1)
    out, prev = [], -1
    for k in best:
        k = int(k)
        if k != prev and k != blank:
            out.append(k)
        prev = k
    return out


def alignment_confidence(log_probs: np.ndarray,
                         spans: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """Mean per-frame posterior of each aligned token. Feeds `token_accuracy`."""
    conf = np.zeros(len(spans), dtype=np.float32)
    for i, (tok, s, e) in enumerate(spans):
        if e > s:
            conf[i] = float(np.exp(log_probs[s:e, tok]).mean())
    return conf
#!/usr/bin/env python3
"""
Test a trained Tajweed-QuartzNet checkpoint against a real audio file.

Two modes:

  KNOWN AYAH (you know what was recited -- the common case: testing against
  a training clip, or a new recording of a specific ayah):

      python run_inference.py --ckpt exp/full/best.pt --wav clip.wav \\
          --surah 1 --ayah 1

  IDENTIFY (you don't know which ayah it is): greedy-decodes the model's own
  output and matches it against a phoneme index of the whole Quran built
  from TajweedG2P (cached to assets/phoneme_index.json after the first run,
  since building it calls the phonemizer 6,236 times):

      python run_inference.py --ckpt exp/full/best.pt --wav clip.wav --identify

PIPELINE
----------
  audio -> log_mel -> model forward pass -> CTC log-probs
                                                  |
                        known ayah? -----yes------+------no------ greedy decode
                              |                                        |
                    TajweedG2P(surah,ayah)                    phoneme n-gram index
                    -> target tokens + spans                  -> best-matching ayah
                              |                                        |
                    forced_align(log_probs, targets)  <-----------------
                              |
                    TajweedGrader (verifier probs if the checkpoint has
                    trained verifier weights, else duration-only grading)
                              |
                    printed report + optional --json

A checkpoint from train.py's CTC phase (with_verifier=False, which is what
--light/--full both currently produce) has no verifier weights, so grading
falls back to the deterministic duration/tolerance checks in
TajweedGrader -- still real tajweed feedback, just without the learned
"does this actually sound right" signal a trained verifier head would add.
This script detects that automatically from the checkpoint's own state dict
rather than needing to be told.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data.tajweed_g2p import G2PResult, TajweedG2P
from data.tokens import TOKEN2ID, VOCAB, VOCAB_SIZE
from model.tajweed_quartznet import build_model
from runtime.frontend import N_MELS, SR, forced_align, greedy_decode, log_mel
from tajweed.grader import FRAME_MS, TajweedGrader, format_report

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_PHONEME_INDEX_PATH = _ASSETS_DIR / "phoneme_index.json"


# ===========================================================================
# 1. Load the model
# ===========================================================================
def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, bool]:
    ck = torch.load(ckpt_path, map_location=device)
    ckpt_vocab = ck.get("vocab_size")
    if ckpt_vocab is not None and ckpt_vocab != VOCAB_SIZE:
        sys.exit(
            f"checkpoint was trained with vocab_size={ckpt_vocab}, but the "
            f"currently loaded assets/vocab.json gives VOCAB_SIZE={VOCAB_SIZE}. "
            f"These must match -- the token ids mean different things under "
            f"each vocabulary. Either retrain against the current vocab, or "
            f"regenerate assets/vocab.json (scripts/build_vocab.py) to match "
            f"whatever quranic-phonemizer version produced this checkpoint.")

    with_verifier = any(k.startswith("verifier.") for k in ck["model"])
    model = build_model(N_MELS, VOCAB_SIZE, with_verifier=with_verifier)
    missing = model.load_state_dict(ck["model"], strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        print(f"  (state_dict mismatch -- missing: {missing.missing_keys}, "
             f"unexpected: {missing.unexpected_keys})", file=sys.stderr)
    model.to(device).eval()

    per = ck.get("per")
    per_s = f"{per*100:.1f}%" if per is not None else "?"
    print(f"loaded {ckpt_path} (epoch {ck.get('epoch', '?')}, "
         f"train/val PER at save time: {per_s}, "
         f"verifier weights: {'yes' if with_verifier else 'no -- duration-only grading'})")
    return model, with_verifier


# ===========================================================================
# 2. Run the model on one clip
# ===========================================================================
def run_model(model: torch.nn.Module, wav: np.ndarray, sr: int,
              device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (log_probs (T, V), enc (C, T)) as numpy arrays, unpadded."""
    feats = log_mel(wav, sr)
    x = torch.from_numpy(feats)[None].to(device)
    fl = torch.tensor([feats.shape[1]], dtype=torch.long, device=device)
    with torch.no_grad():
        logp, enc, ol = model(x, fl)
    t_out = int(ol[0])
    return logp[0, :t_out].cpu().numpy(), enc[0, :, :t_out].cpu().numpy()


def run_verifier(model: torch.nn.Module, enc: np.ndarray, align: List[tuple],
                 spans: List, device: torch.device,
                 max_len: int = 64) -> Optional[np.ndarray]:
    """(n_spans, n_classes) softmax, or None if this checkpoint has no verifier."""
    if model.verifier is None or not spans:
        return None
    from quran_asr.tajweed.grader import estimate_harakah_ms
    h = estimate_harakah_ms(align)
    C, _T = enc.shape
    slices, extras, lens = [], [], []
    for sp in spans:
        s, e = sp.start, min(sp.end, len(align))
        if s >= len(align) or e <= s:
            f0, f1 = 0, 1
        else:
            f0, f1 = align[s][1], align[e - 1][2]
        pad = 4
        f0 = max(0, f0 - pad)
        f1 = min(enc.shape[1], max(f0 + 1, f1 + pad), f0 + max_len)
        slices.append(enc[:, f0:f1].T)
        meas = (f1 - f0) * FRAME_MS / max(h, 1e-6)
        exp = sp.expected_harakat or 1.0
        extras.append([meas, exp, meas / exp, (f1 - f0) * FRAME_MS / 1000.0,
                      h / 1000.0, float(hash(sp.rule) % 97) / 97.0])
        lens.append(f1 - f0)
    L = max(lens) if lens else 1
    span_t = torch.zeros(len(slices), L, C, device=device)
    for i, s in enumerate(slices):
        span_t[i, :s.shape[0]] = torch.from_numpy(s).to(device)
    extra_t = torch.tensor(extras, dtype=torch.float32, device=device)
    lens_t = torch.tensor(lens, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model.verifier(span_t, extra_t, lens_t)
        probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


# ===========================================================================
# 3. Ayah identification (--identify), for when you don't already know it
# ===========================================================================
def _build_phoneme_index(n: int = 4) -> Tuple[List[Tuple[int, int]], List[List[int]],
                                               Dict[Tuple[int, ...], List[int]]]:
    """
    n-gram inverted index over every ayah's target token-id sequence, plus
    exact Levenshtein re-ranking of the top candidates -- mirrors the design
    used elsewhere in this project for the same reason: the search space is
    exactly 6,236 fixed strings, so an inverted index is exact and cannot
    hallucinate an ayah that does not exist, unlike a learned language model.
    """
    from quran_asr.data.data_translator import iter_all_ayat

    print("building phoneme index (first run only; cached after this) ...")
    t0 = time.time()
    g2p = TajweedG2P()
    keys: List[Tuple[int, int]] = []
    seqs: List[List[int]] = []
    post: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for row in iter_all_ayat():
        s, a = row["surah"], row["ayah"]
        try:
            r = g2p(s, a)
        except Exception:                                   # noqa: BLE001
            continue
        ids = [TOKEN2ID[t] for t in r.tokens if t in TOKEN2ID]
        i = len(seqs)
        keys.append((s, a))
        seqs.append(ids)
        for j in range(max(0, len(ids) - n + 1)):
            post[tuple(ids[j:j + n])].append(i)
    print(f"  built in {time.time()-t0:.0f}s ({len(seqs)} ayat)")

    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _PHONEME_INDEX_PATH.write_text(json.dumps({
        "n": n, "keys": keys, "seqs": seqs,
        "post": {",".join(map(str, k)): v for k, v in post.items()},
    }), encoding="utf-8")
    return keys, seqs, post


def _load_or_build_phoneme_index(n: int = 4):
    if _PHONEME_INDEX_PATH.exists():
        data = json.loads(_PHONEME_INDEX_PATH.read_text(encoding="utf-8"))
        if data.get("n") == n:
            keys = [tuple(k) for k in data["keys"]]
            seqs = data["seqs"]
            post = {tuple(map(int, k.split(","))): v for k, v in data["post"].items()}
            return keys, seqs, post
    return _build_phoneme_index(n)


def _lev(a: List[int], b: List[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def identify_ayah(hyp_ids: List[int], n: int = 4,
                  topk: int = 3) -> List[Tuple[float, Tuple[int, int]]]:
    keys, seqs, post = _load_or_build_phoneme_index(n)
    votes = Counter()
    for j in range(max(0, len(hyp_ids) - n + 1)):
        for i in post.get(tuple(hyp_ids[j:j + n]), ()):
            votes[i] += 1
    candidates = [i for i, _ in votes.most_common(30)] or list(range(min(30, len(seqs))))
    scored = []
    for i in candidates:
        ref = seqs[i]
        d = _lev(hyp_ids, ref)
        scored.append((1.0 - d / max(len(ref), 1), keys[i]))
    scored.sort(reverse=True)
    return scored[:topk]


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="e.g. exp/full/best.pt")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--surah", type=int)
    ap.add_argument("--ayah", type=int)
    ap.add_argument("--identify", action="store_true",
                    help="don't assume you know the ayah -- identify it from "
                         "the model's own decoded output")
    ap.add_argument("--identify-threshold", type=float, default=0.55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verifier-threshold", type=float, default=0.60)
    ap.add_argument("--major-only", action="store_true")
    ap.add_argument("--muttasil-harakat", type=float, default=None,
                    help="override the default 4-harakat madd_wajib_muttasil "
                         "target (Hafs via Shatibiyyah also permits 5)")
    ap.add_argument("--munfasil-harakat", type=float, default=None)
    ap.add_argument("--json", help="also write the full report as JSON here")
    args = ap.parse_args()

    if not args.identify and args.surah is None:
        ap.error("give --surah/--ayah, or pass --identify")

    device = torch.device(args.device)
    model, has_verifier = load_checkpoint(args.ckpt, device)

    import soundfile as sf
    wav, sr = sf.read(args.wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        idx = np.linspace(0, len(wav) - 1, int(len(wav) * SR / sr))
        wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)
        sr = SR
    audio_sec = len(wav) / sr

    t0 = time.time()
    log_probs, enc = run_model(model, wav, sr, device)
    t_model = time.time() - t0

    hyp_ids = greedy_decode(log_probs)
    print(f"\ngreedy decode ({len(hyp_ids)} tokens): "
         f"{' '.join(VOCAB[i] for i in hyp_ids)}")

    if args.identify:
        hits = identify_ayah(hyp_ids)
        if not hits or hits[0][0] < args.identify_threshold:
            print("\nCould not identify the ayah with enough confidence.")
            for score, (s, a) in hits[:3]:
                print(f"   candidate {s}:{a}  score {score:.2f}")
            sys.exit(2)
        score, (surah, ayah) = hits[0]
        print(f"identified: {surah}:{ayah}  (confidence {score:.2f})")
    else:
        surah, ayah = args.surah, args.ayah

    g2p = TajweedG2P()
    ref: G2PResult = g2p(surah, ayah)
    target_ids = [TOKEN2ID[t] for t in ref.tokens if t in TOKEN2ID]

    # a quick, alignment-independent sanity number: how close was the raw
    # greedy decode to the true token sequence, before any forced alignment
    from train import edit_distance
    raw_per = edit_distance(hyp_ids, target_ids) / max(len(target_ids), 1)
    print(f"greedy-decode PER against {surah}:{ayah}: {raw_per*100:.1f}%")

    try:
        segs = forced_align(log_probs, target_ids)
    except RuntimeError as e:
        sys.exit(f"forced alignment failed: {e}\n"
                 f"(the audio may be too short for this ayah, or this is a "
                 f"different ayah than {surah}:{ayah})")
    align = [(VOCAB[t], s, e) for (t, s, e) in segs]

    probs = None
    t_verifier = 0.0
    if has_verifier and ref.spans:
        t0 = time.time()
        probs = run_verifier(model, enc, align, ref.spans, device)
        t_verifier = time.time() - t0

    spans_dicts = [dict(rule=sp.rule, start=sp.start, end=sp.end,
                        letters=sp.letters, word_idx=sp.word_idx,
                        expected_harakat=sp.expected_harakat) for sp in ref.spans]
    overrides = {}
    if args.muttasil_harakat is not None:
        overrides["madd_wajib_muttasil"] = args.muttasil_harakat
    if args.munfasil_harakat is not None:
        overrides["madd_jaiz_munfasil"] = args.munfasil_harakat

    grader = TajweedGrader(verifier_threshold=args.verifier_threshold,
                           report_minor=not args.major_only,
                           nominal_overrides=overrides or None)
    token_accuracy = 1.0 - raw_per
    rep = grader.grade(surah, ayah, align, spans_dicts, probs, token_accuracy)

    print()
    print(ref.text)
    print()
    print(format_report(rep, arabic_words=ref.text.split()))

    total = time.time() - t0 + t_model
    print(f"\n  timing: model {t_model*1000:.0f}ms | verifier {t_verifier*1000:.0f}ms | "
         f"audio {audio_sec:.1f}s | RTF {(t_model+t_verifier)/max(audio_sec,1e-6):.2f}")

    if args.json:
        from dataclasses import asdict
        Path(args.json).write_text(json.dumps({
            "surah": surah, "ayah": ayah, "score": rep.score,
            "harakah_ms": rep.harakah_ms, "greedy_decode_per": raw_per,
            "verdicts": [asdict(v) for v in rep.verdicts],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()

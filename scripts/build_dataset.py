#!/usr/bin/env python3
"""
Build per-clip CTC forced-alignment JSON files from a locally-extracted
tarteel-ai/everyayah dataset (the output of extract_everyayah.py).

Expects, for every clip:
    data/everyayah_extracted/<split>/<reciter>-<idx>.wav
    data/everyayah_extracted/<split>/<reciter>-<idx>.txt   (imlaei text, e.g.
                                                            "بِسْمِ اللَّهِ
                                                            الرَّحْمَنِ
                                                            الرَّحِيمِ")

Writes, alongside each pair:
    data/everyayah_extracted/<split>/<reciter>-<idx>.json

PIPELINE, per clip
--------------------
  1. RECOVER THE AYAH. The .txt file holds imlaei text, not the Uthmani rasm
     the tajweed rules need -- match_text() (quran_asr/data/data_translator.py)
     content-matches it against quranic-phonemizer's own built-in Hafs text
     (never by filename or row order) and recovers (surah, ayah).
  2. APPLY TAJWEED RULES. TajweedG2P(surah, ayah) returns the authoritative
     phoneme token sequence and every tajweed RuleSpan it contains.
  3. CTC FORCED ALIGNMENT. A pretrained Arabic wav2vec2 CTC model, bridged
     onto our phoneme vocabulary via the one-to-many character map in
     quran_asr/data/alignment.py, gives a frame boundary for every token.

The written JSON is what train.py's verifier phase and augment.py's negative
synthesis both read: `tokens`+`audio` alone would be enough for plain CTC
acoustic-model training (which needs no alignment at all -- CTCLoss computes
its own), but grading and negative synthesis need the frame boundaries this
script produces.

USAGE
-----
  pip install quranic-phonemizer soundfile transformers torch

  # light: processes ~20 clips total, to verify the script end-to-end before
  # committing to a run that may take hours over a real dataset
  python scripts\build_dataset.py --data-dir data/everyayah_extracted --light

  # full: every clip in every split (or --splits to restrict)
  python scripts\build_dataset.py --data-dir data/everyayah_extracted --full

Resumable either way: a clip whose .json already exists is skipped, so an
interrupted --full run just picks back up where it left off. --light always
re-picks its sample from the front of the (sorted) file list, so re-running
it also resumes rather than reprocessing already-written clips.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.alignment import SeedModel
from data.data_translator import match_text, match_text_ordered
from data.tajweed_g2p import G2PResult, TajweedG2P
from data.tokens import TOKEN2ID, VOCAB
from src.utils.ctc_force_alignment import forced_align
from utils.tajweed.grader import FRAME_MS, estimate_harakah_ms

SPLITS = ("train", "validation", "test")
_STEM_RE = re.compile(r"^(?P<reciter>.+)-(?P<idx>\d+)$")


# ===========================================================================
# 1. Find (.wav, .txt) pairs
# ===========================================================================
def iter_pairs(data_dir: Path, splits: List[str]) -> Iterator[Tuple[str, Path, Path]]:
    """Yields (split, wav_path, txt_path) for every matched pair, sorted by
    filename within each split so --light and resumed --full runs are
    deterministic."""
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            print(f"  (skipping split {split!r}: {split_dir} does not exist)",
                 file=sys.stderr)
            continue
        for wav_path in sorted(split_dir.glob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists():
                yield split, wav_path, txt_path
            else:
                print(f"  {wav_path.name}: no matching .txt, skipping",
                     file=sys.stderr)


def parse_reciter(stem: str) -> str:
    m = _STEM_RE.match(stem)
    return m.group("reciter") if m else "unknown"


# ===========================================================================
# 2. Per-clip processing
# ===========================================================================
def align_one(seed: SeedModel, wav_path: str,
              tokens: List[str]) -> Optional[List[tuple]]:
    """Returns [(token_str, start_frame, end_frame), ...] or None on failure."""
    log_probs = seed.log_probs(wav_path)
    target_ids = [TOKEN2ID[t] for t in tokens if t in TOKEN2ID]
    if not target_ids:
        return None
    try:
        segs = forced_align(log_probs, target_ids)
    except RuntimeError:
        return None
    return [(VOCAB[t], s, e) for (t, s, e) in segs]


def build_json_row(uid: str, wav_path: Path, reciter: str, surah: int, ayah: int,
                   g2p_result: G2PResult, align: List[tuple]) -> dict:
    """
    Schema matches step1_tajweed_ctc_alignment.py's to_manifest_row, so
    tooling written against one works against the other:
      uid, audio, surah, ayah, reciter, text, tokens, harakah_ms, label,
      spans: [{rule, start, end, letters, word_idx, expected_harakat,
               measured_harakat}, ...]
      align: [[token_str, start_frame, end_frame], ...]  -- align[i]
             corresponds to tokens[i], in order
    """
    h = estimate_harakah_ms(align)
    spans_out = []
    for sp in g2p_result.spans:
        seg = [x for x in align[sp.start:sp.end] if x[2] > x[1]]
        measured = None
        if seg:
            t0, t1 = seg[0][1] * FRAME_MS, seg[-1][2] * FRAME_MS
            measured = (t1 - t0) / max(h, 1e-6)
        spans_out.append({
            "rule": sp.rule, "start": sp.start, "end": sp.end,
            "letters": sp.letters, "word_idx": sp.word_idx,
            "expected_harakat": sp.expected_harakat,
            "measured_harakat": measured,
        })
    return {
        "uid": uid, "audio": str(wav_path), "surah": surah, "ayah": ayah,
        "reciter": reciter, "text": g2p_result.text,
        "tokens": g2p_result.tokens, "harakah_ms": h, "label": "correct",
        "spans": spans_out,
        "align": [[t, s, e] for (t, s, e) in align],
    }


def process_one(seed: SeedModel, g2p: TajweedG2P, split: str, wav_path: Path,
                txt_path: Path, reciter_last: Dict[str, tuple],
                assume_mushaf_order: bool, stats: Dict[str, int]) -> Optional[str]:
    """Returns an error/skip reason string, or None on success."""
    text = txt_path.read_text(encoding="utf-8").strip()
    reciter = parse_reciter(wav_path.stem)

    if assume_mushaf_order:
        m = match_text_ordered(text, reciter_last, reciter)
    else:
        m = match_text(text)

    if m.status == "unmatched":
        stats["unmatched"] += 1
        return "unmatched"
    if m.status == "ambiguous":
        stats["ambiguous"] += 1
        return "ambiguous"
    if m.ayah == 0:
        stats["basmalah_skipped"] += 1
        return "basmalah preamble (no ayah to align against)"

    g2p_result = g2p(m.surah, m.ayah)
    print(f"g2p_result.tokens = {g2p_result.tokens}")
    align = align_one(seed, str(wav_path), g2p_result.tokens)
    if align is None:
        stats["align_failed"] += 1
        return "forced alignment failed"

    row = build_json_row(wav_path.stem, wav_path, reciter, m.surah, m.ayah,
                         g2p_result, align)
    out_path = wav_path.with_suffix(".json")
    out_path.write_text(json.dumps(row, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    stats["written"] += 1
    return None


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/everyayah_extracted")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS),
                    choices=SPLITS, help="which splits to process")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true",
                      help="process every clip in the requested splits")
    mode.add_argument("--light", action="store_true",
                      help="process only --light-n clips total, to verify "
                           "the script before committing to a full run")
    ap.add_argument("--light-n", type=int, default=20)
    ap.add_argument("--seed-model", default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
                    help="pretrained CTC model for round-1 alignment, or a "
                         "path to your own exp/ctc/best.pt after round 1")
    ap.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    ap.add_argument("--overwrite", action="store_true",
                    help="reprocess clips whose .json already exists, "
                         "instead of skipping them")
    ap.add_argument("--assume-mushaf-order", action="store_true",
                    help="resolve repeated ayat / basmalah preambles by "
                         "proximity to the reciter's last resolved ayah -- "
                         "see data_translator.match_text_ordered's docstring")
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    limit = args.light_n if args.light else None

    print(f"loading seed model {args.seed_model} on {args.device} ...")
    seed = SeedModel(args.seed_model, device=args.device)
    g2p = TajweedG2P()
    reciter_last: Dict[str, tuple] = {}

    stats: Dict[str, int] = {
        "written": 0, "skipped_existing": 0, "unmatched": 0,
        "ambiguous": 0, "basmalah_skipped": 0, "align_failed": 0,
    }

    t0 = time.time()
    n_seen = 0
    for split, wav_path, txt_path in iter_pairs(data_dir, args.splits):
        if limit is not None and n_seen >= limit:
            break

        out_path = wav_path.with_suffix(".json")
        if out_path.exists() and not args.overwrite:
            stats["skipped_existing"] += 1
            n_seen += 1
            continue

        reason = process_one(seed, g2p, split, wav_path, txt_path,
                             reciter_last, args.assume_mushaf_order, stats)
        n_seen += 1

        if reason:
            print(f"  [{split}/{wav_path.stem}] {reason}")
        if n_seen % args.progress_every == 0:
            done = stats["written"]
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1e-6)
            print(f"  ... {n_seen} seen, {done} written "
                 f"({rate:.1f} clips/s, {elapsed:.0f}s elapsed)")

    print(f"\n{'='*70}")
    print(f"mode: {'light (n=' + str(args.light_n) + ')' if args.light else 'full'}"
         f"  |  splits: {args.splits}")
    for k, v in stats.items():
        print(f"  {k:<18}{v:>8,}")
    print(f"  {'elapsed':<18}{time.time()-t0:>7.0f}s")
    print(f"{'='*70}")

    if stats["unmatched"] or stats["ambiguous"]:
        print(f"\n{stats['unmatched']} unmatched + {stats['ambiguous']} ambiguous "
             f"clips were skipped, not written. Unmatched rows are worth a "
             f"manual look (scripts/audit_matching.py can give you the full "
             f"picture without re-running alignment). Ambiguous rows are "
             f"expected -- repeated ayat and basmalah preambles that no "
             f"text-matching alone can resolve; re-run with "
             f"--assume-mushaf-order to resolve them by proximity instead, "
             f"only if you've confirmed this dataset preserves mushaf order "
             f"per reciter.")
    if args.light and stats["written"] > 0:
        print(f"\nLight run looks {'good' if stats['align_failed'] == 0 else 'CHECK align_failed above'} "
             f"-- inspect a couple of the written .json files by hand, then "
             f"re-run with --full (same command, swap --light for --full) "
             f"when you're satisfied. Already-written clips will be skipped "
             f"automatically.")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()

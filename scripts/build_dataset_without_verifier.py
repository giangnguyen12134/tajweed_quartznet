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

from data.data_translator import match_text, match_text_ordered
from data.tajweed_g2p import G2PResult, TajweedG2P

SPLITS = ("train", "validation", "test")
_STEM_RE = re.compile(r"^(?P<reciter>.+)-(?P<idx>\d+)$")

# NOTE: this manifest is for CTC encoder/head training only. nn.CTCLoss needs
# (log_probs, targets, input_lengths, target_lengths) and nothing else -- no
# forced alignment, no seed model, no per-frame boundaries. Alignment-derived
# fields (harakah_ms, measured_harakat, per-span durations) are a *separate*
# downstream artifact needed only for the rule-based duration checker / GOP
# calibration, produced later by re-running alignment with this script's own
# trained model as --seed-model, once it exists. See align_dataset.py (or
# whatever this repo names that follow-up script) for that step -- don't
# conflate the two: adding alignment back into this file recreates the
# external-model dependency and vocab-mismatch risk we're deliberately
# avoiding here.


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
def build_json_row(uid: str, wav_path: Path, reciter: str, surah: int, ayah: int,
                   g2p_result: G2PResult) -> dict:
    """
    Encoder-training manifest row -- audio + token ids only, no alignment.

    Schema: uid, audio, surah, ayah, reciter, text, tokens, label
    (no harakah_ms / spans / align -- those are alignment-derived and belong
    to the separate calibration-dataset step, not this one)
    """
    return {
        "uid": uid, "audio": str(wav_path), "surah": surah, "ayah": ayah,
        "reciter": reciter, "text": g2p_result.text,
        "tokens": g2p_result.tokens, "label": "correct",
    }


def process_one(g2p: TajweedG2P, split: str, wav_path: Path, txt_path: Path,
                reciter_last: Dict[str, tuple], assume_mushaf_order: bool,
                stats: Dict[str, int],
                g2p_cache: Dict[Tuple[int, int], G2PResult]) -> Optional[str]:
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

    # G2PResult depends only on (surah, ayah), never on the audio/reciter --
    # everyayah.com has ~dozens of reciters per ayah, so without this cache
    # every ayah gets re-phonemized once per reciter instead of once, ever.
    key = (m.surah, m.ayah)
    g2p_result = g2p_cache.get(key)
    if g2p_result is None:
        g2p_result = g2p(m.surah, m.ayah)
        g2p_cache[key] = g2p_result
        stats["g2p_cache_miss"] += 1
    else:
        stats["g2p_cache_hit"] += 1

    if not g2p_result.tokens:
        stats["empty_tokens"] += 1
        return "g2p produced no tokens"

    row = build_json_row(wav_path.stem, wav_path, reciter, m.surah, m.ayah,
                         g2p_result)
    out_path = wav_path.with_suffix(".json")
    out_path.write_text(json.dumps(row, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    stats["written"] += 1
    return None


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/everyayah_without_verifier")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS),
                    choices=SPLITS, help="which splits to process")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true",
                      help="process every clip in the requested splits")
    mode.add_argument("--light", action="store_true",
                      help="process only --light-n clips total, to verify "
                           "the script before committing to a full run")
    ap.add_argument("--light-n", type=int, default=20)
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

    g2p = TajweedG2P()
    reciter_last: Dict[str, tuple] = {}
    g2p_cache: Dict[Tuple[int, int], G2PResult] = {}

    stats: Dict[str, int] = {
        "written": 0, "skipped_existing": 0, "unmatched": 0,
        "ambiguous": 0, "basmalah_skipped": 0, "empty_tokens": 0,
        "g2p_cache_hit": 0, "g2p_cache_miss": 0,
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

        reason = process_one(g2p, split, wav_path, txt_path,
                             reciter_last, args.assume_mushaf_order, stats,
                             g2p_cache)
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
    n_g2p = stats["g2p_cache_hit"] + stats["g2p_cache_miss"]
    if n_g2p:
        hit_rate = stats["g2p_cache_hit"] / n_g2p
        print(f"  g2p cache hit rate: {hit_rate:.1%} "
             f"({stats['g2p_cache_miss']} unique ayat phonemized, "
             f"{stats['g2p_cache_hit']} reused)")
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
        print(f"\nLight run looks good -- inspect a couple of the written "
             f".json files by hand, then re-run with --full (same command, "
             f"swap --light for --full) when you're satisfied. Already-"
             f"written clips will be skipped automatically.")


if __name__ == "__main__":
    main()
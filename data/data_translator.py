from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_TEXT_CACHE_PATH = _ASSETS_DIR / "quran_text_hafs.json"


# ===========================================================================
# Arabic text normalization
# ===========================================================================
# Marks to strip: all harakat/tanween/shadda/sukun, the dagger alif, and
# Quranic annotation marks (waqf signs, sajdah, small high meem of iqlab,
# silent-letter zeros).
#
# The dagger alif (U+0670) is stripped rather than mapped to a full alif,
# even though it usually represents a long-a vowel that imlaei spelling
# writes out as a full alif letter (e.g. "ٱلْعَٰلَمِينَ" -> "العالمين").
# It is NOT mapped because Arabic orthography has a small but very
# high-frequency set of EXCEPTION words where the dagger alif is conventionally
# left unwritten even in plain spelling: ٱلرَّحْمَٰنِ (ar-Rahman) -> "الرحمن",
# ذَٰلِكَ (dhalika) -> "ذلك", هَٰذَا (hadha) -> "هذا", and a few more. These are
# some of the most common words in the Quran, so guessing wrong on them would
# cost more exact-matches than a blanket "expand it" rule would gain. There is
# no simple character-level rule that gets both cases right; encoding the
# actual exception list would need a maintained wordlist for marginal benefit.
# So: strip it here, and let the ~1-character-length difference this leaves
# between the two spellings (e.g. "العلمين" vs "العالمين") be picked up by
# ReferenceIndex.lookup_fuzzy's edit-distance fallback instead, which does not
# care which specific rule caused the difference.
_STRIP = re.compile(
    "[\u064B-\u065F\u0670\u06D6-\u06ED\u06E0-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]"
)

# Collapse letter variants that differ between Uthmani rasm and imlaei
# spelling but represent "the same sound" for matching purposes: the four
# alif-with-hamza/madda forms, hamzat al-wasl, and standalone hamza (U+0621,
# whose realization as a full letter vs. a bare mark varies across
# transcription styles) all collapse to bare alif; alif maksura and
# yaa-with-hamza collapse to yaa; waw-with-hamza to waw.
_ALEF = str.maketrans({
    "\u0621": "\u0627",
    "\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627", "\u0671": "\u0627",
    "\u0649": "\u064A", "\u0626": "\u064A",
    "\u0624": "\u0648",
})


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)

    # Equivalent spelling forms for *comparison only*
    text = text.replace("ٱ", "ا")   # alif wasla -> alif
    text = text.replace("ـ", "")    # tatweel
    text = text.replace("ٰ", "")    # dagger alif

    # Quranic annotation marks and Arabic-Indic verse numbers
    text = re.sub(r"[\u06D6-\u06ED\u0660-\u0669]", "", text)

    return " ".join(text.split())



def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """1.0 = identical, 0.0 = completely different, after normalization."""
    na, nb = normalize_arabic(a), normalize_arabic(b)
    if not na and not nb:
        return 1.0
    return 1.0 - levenshtein(na, nb) / max(len(na), len(nb), 1)


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


@dataclass
class RefEntry:
    surah: int
    ayah: int              # 0 = basmalah preamble before this surah
    text: str               # authoritative Uthmani text -- what actually gets G2P'd
    norm: str


@dataclass
class ReferenceIndex:
    entries: List[RefEntry]
    unique: Dict[str, int] = field(default_factory=dict)           # norm -> idx
    ambiguous: Dict[str, List[int]] = field(default_factory=dict)  # norm -> [idx,...]

    def lookup_exact(self, norm_text: str) -> Tuple[Optional[RefEntry], bool]:
        """Returns (entry_or_None, is_ambiguous)."""
        if norm_text in self.unique:
            return self.entries[self.unique[norm_text]], False
        if norm_text in self.ambiguous:
            return None, True
        return None, False

    def lookup_fuzzy(self, norm_text: str, threshold: float = 0.85,
                     tie_epsilon: float = 1e-9
                     ) -> Tuple[Optional[RefEntry], Optional[List[RefEntry]]]:
        """
        Slow path for rows with minor transcription differences that survive
        normalization (a typo, an extra/missing word). Only reached on the exact
        lookup's misses, and only scans entries of similar length to stay fast.

        Returns (best_entry_or_None, tied_candidates_or_None). tied_candidates
        is populated (with >=2 distinct-ayah entries) only when more than one
        DISTINCT (surah, ayah) ties for the top score -- in which case
        best_entry is None. CRITICAL: silently returning an arbitrary one of a
        tie is exactly the failure mode this module exists to avoid -- a
        repeated ayah (or a basmalah preamble) whose exact-hash match happens
        to miss due to an unrelated spelling variant must not fall through to
        a confident-looking but arbitrary guess here.
        """
        L = len(norm_text)
        scored: List[Tuple[float, RefEntry]] = []
        for e in self.entries:
            if abs(len(e.norm) - L) > max(4, L // 6):
                continue
            score = _ratio(norm_text, e.norm)
            if score >= threshold:
                scored.append((score, e))
        if not scored:
            return None, None
        best_score = max(s for s, _e in scored)
        top = [e for s, e in scored if s >= best_score - tie_epsilon]
        distinct = {(e.surah, e.ayah) for e in top}
        if len(distinct) > 1:
            return None, top
        return top[0], None


def iter_all_ayat(riwayah: str = "hafs") -> Iterator[dict]:
    """
    Yields {surah, ayah, text} for all 6,236 ayat, sourced directly from
    quranic-phonemizer's built-in Hafs Uthmani corpus -- no Tanzil download.
    `text` here is res.text() for each ayah, exactly what TajweedG2P would
    produce phonemes from.

    Ayah counts come from assets/surah_ayah_counts.json (produced by
    scripts/build_vocab.py, DERIVED by probing the library rather than
    hand-typed) if present, else probed live here as a fallback.
    """
    from quranic_phonemizer import Phonemizer
    pm = Phonemizer()

    counts_path = _ASSETS_DIR / "surah_ayah_counts.json"
    if counts_path.exists():
        counts = {int(k): v for k, v in
                 json.loads(counts_path.read_text(encoding="utf-8")).items()}
    else:
        counts = {}
        for s in range(1, 115):
            a = 1
            while True:
                try:
                    pm.phonemize(f"{s}:{a}")
                    a += 1
                except ValueError:
                    break
            counts[s] = a - 1

    for s in range(1, 115):
        for a in range(1, counts[s] + 1):
            if (s == 1 and a == 2):
                txt = pm.phonemize(f"{s}:{a}").text()
                print(f"does support s={s}, a = {a}, text = {txt}")
            yield {"surah": s, "ayah": a, "text": pm.phonemize(f"{s}:{a}").text()}


def build_reference_index(quran_rows: List[dict]) -> ReferenceIndex:
    """
    quran_rows: {surah, ayah, text} dicts, typically from iter_all_ayat()
    (authoritative Hafs Uthmani text, sourced from quranic-phonemizer) or a
    small hand-written fixture for isolated tests.
    """
    entries: List[RefEntry] = []
    by_surah: Dict[int, List[dict]] = defaultdict(list)
    for r in quran_rows:
        by_surah[r["surah"]].append(r)

    print(quran_rows[0])
    basmalah_text = next(
        (r["text"] for r in quran_rows if r["surah"] == 1 and r["ayah"] == 1),
        # Fallback only matters for partial/test fixtures that don't include
        # al-Fatihah; the real 6,236-ayah corpus always has 1:1.
        "\u0628\u0650\u0633\u0652\u0645\u0650 \u0627\u0644\u0644\u0651\u064E\u0647\u0650 "
        "\u0627\u0644\u0631\u0651\u064E\u062D\u0652\u0645\u064E\u0670\u0646\u0650 "
        "\u0627\u0644\u0631\u0651\u064E\u062D\u0650\u064A\u0645\u0650",
    )

    for surah in sorted(by_surah):
        ayat = sorted(by_surah[surah], key=lambda r: r["ayah"])
        #if surah not in (1, 9):
        #    entries.append(RefEntry(surah, 0, basmalah_text,
        #                            normalize_arabic(basmalah_text)))
        for r in ayat:
            entries.append(RefEntry(surah, r["ayah"], r["text"],
                                    normalize_arabic(r["text"])))

    counts: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        counts[e.norm].append(i)
    unique = {norm: idxs[0] for norm, idxs in counts.items() if len(idxs) == 1}
    ambiguous = {norm: idxs for norm, idxs in counts.items() if len(idxs) > 1}

    return ReferenceIndex(entries=entries, unique=unique, ambiguous=ambiguous)


def build_reference_index_from_phonemizer(riwayah: str = "hafs") -> ReferenceIndex:
    """Explicit, uncached entry point: build the index straight from the
    library every time. Prefer match_text() for normal use -- this is here
    for callers that specifically want a fresh build, bypassing the disk
    cache (e.g. after regenerating assets/surah_ayah_counts.json)."""
    return build_reference_index(list(iter_all_ayat(riwayah)))


@dataclass
class MatchResult:
    surah: Optional[int]
    ayah: Optional[int]
    confidence: float
    status: str             # "exact" | "fuzzy" | "ambiguous" | "ambiguous_ordered" | "unmatched"
    text: Optional[str] = None       # authoritative text to feed the G2P, if resolved


def resolve_row(hf_text: str, index: ReferenceIndex,
                fuzzy_threshold: float = 0.85) -> MatchResult:
    """Index-explicit implementation. Prefer match_text() unless you're
    testing against a specific, non-default ReferenceIndex."""
    norm = normalize_arabic(hf_text)
    print(f"resolve_row: hf_text={hf_text} text = {norm}")
    entry, ambiguous = index.lookup_exact(norm)
    if entry is not None:
        return MatchResult(entry.surah, entry.ayah, 1.0, "exact", entry.text)
    if ambiguous:
        return MatchResult(None, None, 0.0, "ambiguous")
    fz, tied = index.lookup_fuzzy(norm, fuzzy_threshold)
    if tied is not None:
        return MatchResult(None, None, 0.0, "ambiguous")
    if fz is not None:
        return MatchResult(fz.surah, fz.ayah, _ratio(norm, fz.norm), "fuzzy", fz.text)
    return MatchResult(None, None, 0.0, "unmatched")


def resolve_with_order_hint(hf_text: str, index: ReferenceIndex,
                            reciter_last: Dict[str, Tuple[int, int]], reciter: str,
                            fuzzy_threshold: float = 0.85) -> MatchResult:
    """Index-explicit implementation. Prefer match_text_ordered() unless
    you're testing against a specific, non-default ReferenceIndex."""
    norm = normalize_arabic(hf_text)
    entry, ambiguous = index.lookup_exact(norm)
    if entry is not None:
        reciter_last[reciter] = (entry.surah, entry.ayah)
        return MatchResult(entry.surah, entry.ayah, 1.0, "exact", entry.text)

    candidates: List[RefEntry]
    if ambiguous:
        candidates = [index.entries[i] for i in index.ambiguous[norm]]
    else:
        fz, tied = index.lookup_fuzzy(norm, fuzzy_threshold)
        if tied is not None:
            candidates = tied
        elif fz is not None:
            reciter_last[reciter] = (fz.surah, fz.ayah)
            return MatchResult(fz.surah, fz.ayah, _ratio(norm, fz.norm), "fuzzy", fz.text)
        else:
            return MatchResult(None, None, 0.0, "unmatched")

    prev = reciter_last.get(reciter)
    if prev is None:
        return MatchResult(None, None, 0.0, "ambiguous")   # no anchor -- refuse to guess
    best = min(candidates, key=lambda e: _mushaf_distance((e.surah, e.ayah), prev))
    reciter_last[reciter] = (best.surah, best.ayah)
    return MatchResult(best.surah, best.ayah, 0.5, "ambiguous_ordered", best.text)


def _mushaf_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Rough linear distance in mushaf order, for order-hint tie-breaking only."""
    return abs(a[0] - b[0]) * 300 + abs(a[1] - b[1])


# ===========================================================================
# Public API: lazy, disk-cached singleton + match_text / match_text_ordered
# ===========================================================================
_INDEX: Optional[ReferenceIndex] = None


def _load_or_fetch_quran_rows(riwayah: str = "hafs") -> List[dict]:
    if riwayah == "hafs" and _TEXT_CACHE_PATH.exists():
        return json.loads(_TEXT_CACHE_PATH.read_text(encoding="utf-8"))
    rows = list(iter_all_ayat(riwayah))
    if riwayah == "hafs":
        _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        _TEXT_CACHE_PATH.write_text(json.dumps(rows, ensure_ascii=False),
                                    encoding="utf-8")
    return rows


def _get_index(riwayah: str = "hafs") -> ReferenceIndex:
    """
    Builds (or loads from cache) the reference index on first call, then
    reuses it for the lifetime of the process. This is what makes match_text
    "just work" without the caller ever building an index themselves.
    """
    global _INDEX
    if _INDEX is None:
        _INDEX = build_reference_index(_load_or_fetch_quran_rows(riwayah))
    return _INDEX


def match_text(raw_text: str, fuzzy_threshold: float = 0.85) -> MatchResult:
    """
    Recover (surah, ayah) for a row's `text` column from tarteel-ai/everyayah
    (or any similarly-imlaei-transcribed source). The primary export of this
    module -- no setup call needed, the reference index is built or loaded
    from cache automatically on first use.

    Returns a MatchResult with `status` one of:
      "exact"      -- normalized text matched a unique ayah exactly
      "fuzzy"      -- matched after edit-distance tolerance for minor noise
      "ambiguous"  -- matched more than one ayah (a repeated ayah or a
                      basmalah preamble); surah/ayah are None, not guessed
      "unmatched"  -- matched nothing; worth a manual look
    """
    return resolve_row(raw_text, _get_index(), fuzzy_threshold)


def match_text_ordered(raw_text: str, reciter_last: Dict[str, Tuple[int, int]],
                       reciter: str, fuzzy_threshold: float = 0.85) -> MatchResult:
    """
    Same as match_text, but for the "ambiguous" case (repeated ayat, basmalah
    preambles) picks the candidate nearest in mushaf order to this reciter's
    last resolved ayah, instead of refusing. `reciter_last` is a dict you own
    and pass in across calls -- updated in place as (surah, ayah) per reciter.

    Only use this if you've confirmed your data preserves mushaf order per
    reciter; see this module's docstring. Confidence is capped at 0.5 for a
    result reached this way (status "ambiguous_ordered") -- it is a guess
    anchored on an ordering assumption, not a content match.
    """
    return resolve_with_order_hint(raw_text, _get_index(), reciter_last, reciter,
                                   fuzzy_threshold)

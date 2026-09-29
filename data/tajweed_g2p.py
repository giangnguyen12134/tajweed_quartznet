"""
Tajweed G2P -- adapter over `quranic-phonemizer` (Hafs 'an 'Asim).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from quranic_phonemizer import Phonemizer

from .tokens import TOKEN2ID

# NOMINAL_HARAKAT: how many harakat a correctly-recited occurrence of each
# duration-graded rule should occupy. The library gives us WHICH rule fired
# and WHICH tokens realise it; it does not itself encode a harakat count (the
# madd system's 2/4/6 convention is pedagogy layered on top of the phonetic
# analysis, and Hafs-via-Shatibiyyah even permits some flexibility, e.g. 4 or
# 5 for muttasil/munfasil). This table is therefore intentionally separate
# from the library and lives in tajweed/grader.py, the natural owner of
# grading pedagogy; import it from there rather than duplicating it here.
from utils.tajweed.grader import NOMINAL_HARAKAT  # noqa: E402

 
# Nasal-flavoured phonemes in the 2.9 vocabulary: tokens carrying a combining
# tilde (U+0303, marking nasalisation) plus the two standalone nasal symbols.
# Derived from assets/vocab.json, not guessed -- see scripts/build_vocab.py.
NASAL_PHONEMES = {"m\u0303", "n\u0303", "w\u0303", "j\u0303", "\u0272", "\u014b"}
 
# Rules whose tagged cell is the tanween/noon/meem MARK, not the nasal
# consonant whose duration actually needs grading -- see this module's
# docstring, "THE GHUNNAH HANDOFF".
NASAL_HANDOFF_RULES = {
    "idgham_ghunnah_noon", "idgham_ghunnah_tanween",
    "iqlab_noon", "iqlab_tanween",
    "ikhfaa_noon", "ikhfaa_tanween", "ikhfaa_shafawi",
    "idgham_shafawi", "meem_ghunnah", "noon_ghunnah",
}
_HANDOFF_LOOKAHEAD = 4     # bounded search window, in flat tokens
 
# Rules whose tagged cell has empty phoneme_indices because the underlying
# letter is dropped/silent (hamzat al-wasl, the assimilated laam of lam
# shamsiyyah). No frames exist to grade; these stay categorical.
SILENT_STATUSES = {"dropped"}
 
 
@dataclass
class RuleSpan:
    """A tajweed rule occurrence, pointing at a slice of the token list."""
    rule: str
    start: int                   # inclusive index into tokens
    end: int                     # exclusive
    expected_harakat: float = 0.0
    letters: str = ""
    word_idx: int = 0
    meta: dict = field(default_factory=dict)
 
 
@dataclass
class G2PResult:
    tokens: List[str]
    ids: List[int]
    spans: List[RuleSpan]
    words: List[Tuple[int, int]]   # (token_start, token_end) per orthographic word
    text: str = ""                  # authoritative Uthmani text (ayah-number suffix stripped)
 
 
def _strip_ayah_marker(text: str) -> str:
    """
    res.text() in 2.9 includes a trailing Arabic-Indic ayah number and
    whitespace, e.g. "...ٱلرَّحِيمِ  ١ " for 1:1. Strip it: it is not part of
    the recited text, and leaving it in would corrupt text-matching in
    data_translator.py (an Arabic-Indic digit surviving normalization would
    make every ayah's text end in a different, uninformative character).
    """
    text = text.strip()
    while text and ("\u0660" <= text[-1] <= "\u0669"):   # Arabic-Indic digits
        text = text[:-1].rstrip()
    return text
 
 
class TajweedG2P:
    """
    Parameters
    ----------
    riwayah : str
        Kept for interface stability with callers written against earlier
        versions of this adapter. quranic-phonemizer 2.9 supports only Hafs,
        so this must be "hafs" (the default) -- anything else raises
        ValueError immediately, rather than silently being ignored.
    """
 
    def __init__(self, riwayah: str = "hafs"):
        if riwayah != "hafs":
            raise ValueError(
                f"riwayah={riwayah!r} requested, but quranic-phonemizer 2.9 "
                f"supports only Hafs. Upgrade to a version with multi-riwayah "
                f"support if you need this, and update this adapter to match "
                f"its API (verify empirically -- see this module's docstring "
                f"for why that matters).")
        self._pm = Phonemizer()
 
    # -- public API -----------------------------------------------------
    def __call__(self, surah: int, ayah: int) -> G2PResult:
        """A single, standalone ayah."""
        return self._convert(self._pm.phonemize(f"{surah}:{ayah}"))
 
    def range(self, surah1: int, ayah1: int, surah2: int, ayah2: int) -> G2PResult:
        """A continuous passage from (surah1, ayah1) through (surah2, ayah2)."""
        ref = f"{surah1}:{ayah1}-{surah2}:{ayah2}"
        return self._convert(self._pm.phonemize(ref))
 
    def ref(self, ref: str) -> G2PResult:
        """Escape hatch for any reference string the library accepts directly."""
        return self._convert(self._pm.phonemize(ref))
 
    # -- conversion -------------------------------------------------------
    def _convert(self, res) -> G2PResult:
        per_word_phonemes: List[List[str]] = list(res.phonemes_list())
        tokens: List[str] = [p for word in per_word_phonemes for p in word]
        ids = [TOKEN2ID[t] for t in tokens if t in TOKEN2ID]
 
        words: List[Tuple[int, int]] = []
        word_offset: List[int] = []          # flat-token offset of each word's start
        off = 0
        for word in per_word_phonemes:
            word_offset.append(off)
            words.append((off, off + len(word)))
            off += len(word)
 
        cpm = res.character_phoneme_mappings()
        spans: List[RuleSpan] = []
        for wi, cword in enumerate(cpm.words):
            base = word_offset[wi] if wi < len(word_offset) else off
            for cell in cword.cells:
                if not cell.tag:
                    continue
                rule = cell.tag
                idxs = list(cell.phoneme_indices)
 
                if not idxs:
                    if cell.status in SILENT_STATUSES or not cell.phonemes:
                        continue    # silent/dropped: no frames to grade
                    continue        # defensive: no indices, nothing to span
 
                start, end = base + idxs[0], base + idxs[-1] + 1
                if list(range(idxs[0], idxs[-1] + 1)) != idxs:
                    raise ValueError(
                        f"non-contiguous phoneme_indices {idxs} for rule "
                        f"{rule!r} at word {cword.location}; grader assumes "
                        f"contiguity")
 
                if rule in NASAL_HANDOFF_RULES:
                    end = self._extend_to_nasal(tokens, end)
 
                letters = "".join(tokens[start:end])
                spans.append(RuleSpan(
                    rule=rule, start=start, end=end,
                    expected_harakat=NOMINAL_HARAKAT.get(rule, 0.0),
                    letters=letters, word_idx=wi))
 
        return G2PResult(tokens=tokens, ids=ids, spans=spans, words=words,
                         text=_strip_ayah_marker(res.text()))
 
    @staticmethod
    def _extend_to_nasal(tokens: List[str], end: int) -> int:
        """See this module's docstring, 'THE GHUNNAH HANDOFF'."""
        limit = min(len(tokens), end + _HANDOFF_LOOKAHEAD)
        for i in range(end, limit):
            if tokens[i] in NASAL_PHONEMES:
                return i + 1
        return end   # no nasal phoneme found in the window; leave as-is
 
    # -- introspection ------------------------------------------------------
    def rule_tags_seen(self, surah: int, ayah: int) -> List[str]:
        """Convenience: every rule tag firing in one ayah, for spot-checking."""
        return sorted({sp.rule for sp in self(surah, ayah).spans})
 
 
def phonemize_ayah(surah: int, ayah: int, **kw) -> G2PResult:
    return TajweedG2P(**kw)(surah, ayah)
 
 
if __name__ == "__main__":
    g = TajweedG2P()
    for s, a in [(1, 1), (112, 1), (112, 4), (2, 19)]:
        r = g(s, a)
        print(f"\n{s}:{a}  {r.text}")
        print("  tokens:", " ".join(r.tokens))
        print("  rules :", ", ".join(f"{x.rule}({x.letters})" for x in r.spans))
"""
CTC token vocabulary -- derived from quranic-phonemizer, not hand-typed.

`assets/vocab.json` is built by `scripts/build_vocab.py`, which runs
quranic-phonemizer's Hafs analysis over every one of the 6,236 ayat and
collects every phoneme token that actually occurs. That is a real advantage
over a hand-maintained ASCII inventory: it cannot omit a phoneme the library
produces, and it cannot include one it doesn't. Regenerate it if you switch
riwayah, script, or upgrade quranic-phonemizer to a version with a changed
phoneme set:

    python scripts/build_vocab.py

Tokens fall into a few families (see the phonemizer's own Alphabet for the
authoritative IPA mapping):
  * plain consonants: b, t, th(theta), j(ezh), H(hHa), kh(x), d, dh(edh), r,
    z, s, sh(esh), S(s-pharyngealized), D, T, Z, ain, gh, f, q, k, l, m, n,
    h, w, y, hamza
  * GEMINATED consonants: the token doubled as one string (e.g. "ll", "rr").
    Gemination is baked into the token identity rather than marked with a
    separate symbol, which is exactly right for CTC: a geminated and a plain
    consonant are acoustically different-duration events and deserve
    different labels.
  * short vowels: a, i, u  |  long vowels: a:, i:, u:  |  emphatic long: a-:
  * nasalised glides (ghunnah carried on w/y in some contexts)
  * nasal hums for ikhfa/iqlab
  * qalqalah release marker: Q (attached to the released consonant)

This is a genuine improvement over modelling tajweed length distinctions with
separate ":2"/":4"/":6" suffix tokens the way an earlier hand-rolled version
of this project did: here the ACOUSTIC IDENTITY of a token (short vs long vs
geminated) is what the library assigns from real tajweed analysis, and the
GRADED duration target for how many harakat a given rule's tokens should
occupy is a separate concern, handled per rule_id in tajweed/grader.py. Do
not conflate the two: this file is about what the token IS, not how long a
correct recitation should hold it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

_ASSET = Path(__file__).resolve().parents[1] / "assets" / "vocab.json"

BLANK = "<b>"      # CTC blank, index 0
UNK = "<unk>"


def _load() -> List[str]:
    if _ASSET.exists():
        data = json.loads(_ASSET.read_text(encoding="utf-8"))
        return data["vocab"]
    # Minimal fallback so imports don't hard-fail if the asset hasn't been
    # generated yet; regenerate with `python scripts/build_vocab.py` before
    # training or running inference -- this fallback vocab is NOT the real one.
    import sys
    print(f"WARNING: {_ASSET} not found; using a placeholder vocabulary. "
          f"Run `python scripts/build_vocab.py` before training or inference.",
          file=sys.stderr)
    return [BLANK, UNK, "SIL", "WQF"]


VOCAB: List[str] = _load()
TOKEN2ID: Dict[str, int] = {t: i for i, t in enumerate(VOCAB)}
ID2TOKEN: Dict[int, str] = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE: int = len(VOCAB)


def encode(tokens) -> List[int]:
    return [TOKEN2ID.get(t, TOKEN2ID[UNK]) for t in tokens]


def decode(ids) -> List[str]:
    return [ID2TOKEN[i] for i in ids]


if __name__ == "__main__":
    print(f"vocab size = {VOCAB_SIZE}  (source: {_ASSET})")
    for i, t in enumerate(VOCAB):
        print(f"{i:3d}  {t}")

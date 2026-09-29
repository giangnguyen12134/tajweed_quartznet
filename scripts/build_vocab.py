#!/usr/bin/env python3
"""
Build-time asset generator. Run once, offline (the library ships the Hafs
Uthmani corpus internally -- no network, no Tanzil download needed).

REWRITTEN FOR quranic-phonemizer 2.9. An earlier version of this script was
written against a DIFFERENT installed version whose API turned out to differ
substantially -- verified empirically this time, not assumed:

  * `Phonemizer()` takes NO `riwayah=` argument in 2.9 -- it supports only
    Hafs, so there is nothing to select. The earlier version's
    `Phonemizer(riwayah=riwayah)` raises TypeError under 2.9.
  * The method is `phonemize(ref)`, not `analyse(ref)`.
  * `PhonemizeResult.text()` and `.phonemes_list()` are METHODS (call them).
    `res.text()` also includes a trailing Arabic-Indic ayah-number marker
    and whitespace (e.g. "...ٱلرَّحِيمِ  ١ " for 1:1) that must be stripped.
  * `res.phonemes_list()` returns a list PER WORD (list of lists), not the
    flat sequence `res.phonemes()` gave in the version this replaced --
    flatten it here for vocabulary collection.
  * There is no `res.rule_occurrences` and no `pm.tajweed_rules` catalogue in
    2.9. Tajweed rule tags live in `res.character_phoneme_mappings()`, which
    returns one CharWord per word, each holding `cells` (one per Arabic
    character/mark); a `Cell.tag` names the rule firing there. See
    quran_asr/data/tajweed_g2p.py's module docstring for the full story of
    how this was tracked down, and why `character_phoneme_mappings()` is
    used here rather than `tajweed_mappings()` (which tags at the character
    level too, but does not tie the tag to which phoneme(s) it produced).

Because 2.9 has no rule-description API, rule_catalogue.json's `name` here
is mechanically derived from the tag string (e.g. "madd_wajib_muttasil" ->
"Madd Wajib Muttasil"), not sourced from the library -- fine for a log
message, get a human (qualified in tajweed) to review before using these
names anywhere user-facing. `arabic_name` and `summary` are left blank for
the same reason -- 2.9 does not provide them and this script does not guess.

Produces:
  assets/vocab.json              every phoneme token that occurs anywhere in
                                  the Quran under quranic-phonemizer's Hafs
                                  analysis, frozen for a stable CTC head size
  assets/rule_catalogue.json     every tajweed rule TAG actually observed
                                  (id + a mechanically-derived display name)
  assets/surah_ayah_counts.json  ayah count per surah, DERIVED by probing the
                                  library (phonemize(f"{s}:{a}") until it
                                  fails) rather than hand-typed, so it cannot
                                  go stale or contain a transcription typo
  assets/quran_text_hafs.json    the authoritative (surah, ayah, text) for
                                  all 6,236 ayat, ayah-number marker already
                                  stripped -- reused by data_translator.py's
                                  match_text() so it never has to re-run
                                  phonemize() across the whole Quran itself
"""
from __future__ import annotations

import json
import time

from quranic_phonemizer import Phonemizer


def strip_ayah_marker(text: str) -> str:
    """
    res.text() in 2.9 includes a trailing Arabic-Indic ayah number and
    whitespace, e.g. "...ٱلرَّحِيمِ  ١ " for 1:1. Strip it: it is not part
    of the recited text, and leaving it in corrupts text-matching in
    data_translator.py (an Arabic-Indic digit surviving normalization would
    make every ayah's text end in a different, uninformative character).
    Kept in sync with tajweed_g2p.py's identical helper -- duplicated rather
    than imported so this script has no dependency on the rest of the
    package and can run standalone before anything else is verified working.
    """
    text = text.strip()
    while text and ("\u0660" <= text[-1] <= "\u0669"):   # Arabic-Indic digits
        text = text[:-1].rstrip()
    return text


def main():
    pm = Phonemizer()
    t0 = time.time()

    # --- ayah counts, derived by probing -----------------------------------
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
    total = sum(counts.values())
    print(f"surahs probed: 114, total ayat: {total} (expect 6236)")
    assert total == 6236, f"unexpected ayah total {total}"

    # --- vocabulary + rule tags + full text, one pass over every ayah ------
    vocab = set()
    rule_tags = set()
    quran_rows = []
    for s in range(1, 115):
        for a in range(1, counts[s] + 1):
            res = pm.phonemize(f"{s}:{a}")

            for word in res.phonemes_list():        # list of lists -> flatten
                vocab.update(word)

            cpm = res.character_phoneme_mappings()
            for w in cpm.words:
                for cell in w.cells:
                    if cell.tag:
                        rule_tags.add(cell.tag)

            quran_rows.append({"surah": s, "ayah": a,
                               "text": strip_ayah_marker(res.text())})

    print(f"vocab size: {len(vocab)}  |  rule tags seen: {len(rule_tags)}  |  "
         f"{time.time()-t0:.1f}s")

    vocab_list = ["<b>", "<unk>", "SIL", "WQF"] + sorted(vocab)
    with open("assets/vocab.json", "w", encoding="utf-8") as fh:
        json.dump({"vocab": vocab_list, "blank": 0}, fh, ensure_ascii=False, indent=1)

    catalogue = [{"id": tag, "name": tag.replace("_", " ").title(),
                 "arabic_name": "", "summary": ""}
                for tag in sorted(rule_tags)]
    with open("assets/rule_catalogue.json", "w", encoding="utf-8") as fh:
        json.dump(catalogue, fh, ensure_ascii=False, indent=1)

    with open("assets/surah_ayah_counts.json", "w", encoding="utf-8") as fh:
        json.dump(counts, fh, indent=1)

    with open("assets/quran_text_hafs.json", "w", encoding="utf-8") as fh:
        json.dump(quran_rows, fh, ensure_ascii=False)

    print("wrote assets/vocab.json, assets/rule_catalogue.json, "
         "assets/surah_ayah_counts.json, assets/quran_text_hafs.json")
    print()
    print("NOTE: quran_asr/tajweed/grader.py's NOMINAL_HARAKAT/TOLERANCE/"
         "SEVERITY tables are keyed to specific rule tag strings by hand -- "
         "if rule_tags above differs from what grader.py expects (e.g. after "
         "a quranic-phonemizer upgrade changes tag spellings), those tables "
         "need a matching update or affected rules will silently fall back "
         "to ungraded/categorical treatment.")


if __name__ == "__main__":
    main()

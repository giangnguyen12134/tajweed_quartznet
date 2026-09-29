"""
Seed alignment for bootstrapping (training-time only; never runs on the Edge device).

Stage `align` in build_dataset.py needs frame boundaries before we have a model
of our own. We get them from a pretrained multilingual/Arabic wav2vec2 CTC model
by mapping its character posterior onto our token set, then running the same
Viterbi used at inference. After one training round, re-run this with
`--seed-model exp/ctc/best.pt` to re-align with our own (much better) model and
repeat 2-3 times.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from data.tokens import TOKEN2ID, VOCAB
from src.utils.ctc_force_alignment import forced_align as _viterbi
from src.utils.log_mel import log_mel as _log_mel



def log_mel(path: str) -> np.ndarray:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return _log_mel(wav, sr)


class SeedModel:
    """Wraps either a HF wav2vec2 CTC model or one of our own checkpoints."""

    def __init__(self, name_or_path: str, device: str = "cuda"):
        import torch
        self.torch, self.device = torch, device
        self.own = name_or_path.endswith(".pt")
        if self.own:
            from ..model.tajweed_quartznet import build_model
            from ..data.tokens import VOCAB_SIZE
            self.model = build_model(80, VOCAB_SIZE, with_verifier=False)
            self.model.load_state_dict(
                torch.load(name_or_path, map_location="cpu")["model"], strict=False)
            self.model.to(device).eval()
            self.map = None
        else:
            from transformers import AutoModelForCTC, AutoProcessor
            self.proc = AutoProcessor.from_pretrained(name_or_path)
            self.model = AutoModelForCTC.from_pretrained(name_or_path).to(device).eval()
            self.map = _build_char_map(self.proc.tokenizer.get_vocab())

    def log_probs(self, feats_or_path) -> np.ndarray:
        """
        feats_or_path: for our own .pt models, a (n_mels, T) log-mel array.
        For the HF wav2vec2 seed model, any of: a file path (str), a raw
        float32 waveform array (assumed 16 kHz), or an (array, sampling_rate)
        tuple -- the last is what you get straight out of a HF `datasets`
        Audio column (row["audio"]["array"], row["audio"]["sampling_rate"]),
        with no need to round-trip through a wav file on disk.
        """
        torch = self.torch
        with torch.no_grad():
            if self.own:
                x = torch.from_numpy(feats_or_path)[None].to(self.device)
                lp, _e, _l = self.model(x)
                return lp[0].cpu().numpy()
            if isinstance(feats_or_path, str):
                import soundfile as sf
                wav, sr = sf.read(feats_or_path, dtype="float32")
            elif isinstance(feats_or_path, tuple):
                wav, sr = feats_or_path
            else:
                wav, sr = feats_or_path, 16000
            wav = np.asarray(wav, dtype=np.float32)
            iv = self.proc(wav, sampling_rate=sr, return_tensors="pt").input_values
            lg = self.model(iv.to(self.device)).logits[0]
            lp = torch.log_softmax(lg, dim=-1).cpu().numpy()
            return _project(lp, self.map)


def _build_char_map(hf_vocab: dict) -> List[List[int]]:
    """
    Map the seed model's Arabic character vocabulary onto our token ids --
    ONE ARABIC LETTER TO MANY TARGET TOKENS, not the one-to-one map an
    earlier version of this file used.

    WHY ONE-TO-MANY IS NECESSARY, NOT JUST NICER
    -----------------------------------------------
    A one-to-one map (each letter -> exactly one token) leaves most of our
    73-token vocabulary with literally zero acoustic evidence: emphatic
    (pharyngealized) and geminated variants of a letter -- rˤ, rˤrˤ, lˤlˤ,
    bb, ll, and so on -- have no letter distinct from their plain form;
    nasal-hum phonemes (m̃, ñ, ŋ, w̃, j̃) have no letter either. With no
    acoustic evidence anywhere in the posterior for a token, CTC forced
    alignment has nothing to place it against and collapses it to the
    cheapest possible allocation -- one frame, wherever it happens to be
    "least improbable". So instead: each Arabic letter's posterior mass is
    shared across EVERY token that letter could plausibly realise -- its
    plain form, its emphatic form, and its geminated form. This does not
    recover the fine distinctions themselves (a crude bootstrap cannot know
    whether a given raa is emphatic without more context than a
    character-level projection carries) -- it recovers TIMING, which is
    what forced alignment needs. Round 2 onward, aligning with your own
    trained model instead of this seed, the fine distinctions become real;
    this map only has to get boundaries roughly right for round 1.

    ON SHORT VOWELS -- VERIFY YOUR TOKENIZER'S VOCABULARY, DO NOT ASSUME IT
    ----------------------------------------------------------------------
    An earlier version of this file assumed short vowels (a/i/u) had no
    letter at all to draw evidence from, reasoning that harakat are
    diacritics typically absent from an ASR tokenizer's character set. That
    assumption was WRONG for jonatasgrosman/wav2vec2-large-xlsr-53-arabic
    specifically -- confirmed by direct inspection of its tokenizer
    vocabulary, not assumed from documentation or general expectations
    about Arabic ASR models. Its 51-token vocabulary includes fatha/damma/
    kasra and fathatan/dammatan/kasratan as separate tokens (indices
    43-50), so short vowels DO have real acoustic evidence available, and
    the map below uses it. Getting this specific wrong assumption right
    took three rounds of "the alignment is still degenerate, why" before
    actually asking the model for its vocabulary rather than guessing at
    it -- if you swap in a different seed model, verify its vocabulary the
    same way (`AutoProcessor.from_pretrained(...).tokenizer.get_vocab()`)
    before assuming this map's keys still apply; a silently-wrong key here
    produces no error, just degenerate alignment with no obvious cause.

    A LETTER SERVING TWO ROLES GETS BOTH, TOGETHER
    ---------------------------------------------------
    و (waw) is both the long-u vowel carrier (as in a word ending -بِهُ) and a
    consonant (the glide /w/). ي (yaa) is both the long-i vowel carrier and a
    consonant (the glide /j/) -- crucially NOT the same phoneme as ج (jim,
    /ʒ/), which an earlier version of this map conflated. Since the grapheme
    alone does not disambiguate the role, both roles' target tokens get
    evidence from the same source letter; better to support both plausible
    readings than to silently pick one and starve the other of any signal.

    Token spellings match assets/vocab.json exactly (see
    scripts/build_vocab.py) -- this is the real, derived quranic-phonemizer
    inventory, not a hand-guessed one, so these strings must match it letter
    for letter or every lookup silently misses.
    """
    # base Arabic letter -> every target token that letter's acoustic
    # evidence should support
    #
    # VERIFIED against jonatasgrosman/wav2vec2-large-xlsr-53-arabic's actual
    # 51-token vocabulary (confirmed by direct inspection, not assumed --
    # an earlier version of this map was built on an unverified assumption
    # that this tokenizer had no diacritic tokens, which was wrong for this
    # specific model and left every short vowel with zero evidence
    # regardless of how complete the rest of the map was). This vocabulary
    # DOES carry short-vowel and gemination diacritics as separate tokens
    # (indices 43-50: fathatan/dammatan/kasratan/fatha/damma/kasra/shadda/
    # sukun), which changes the right design substantially: short vowels
    # get their OWN evidence source now, so ا/و/ي only need to carry their
    # LONG-vowel and (for و/ي) consonant-glide roles, not the short vowel
    # too -- giving both sharper, less conflated evidence.
    ar2toks: dict = {
        # consonants: plain + geminated (+ emphatic where the letter is
        # conditionally rather than inherently emphatic -- raa and laam)
        "ب": ["b", "bb"], "ت": ["t", "tt"], "ث": ["θ", "θθ"],
        "ج": ["ʒ", "ʒʒ"], "ح": ["ħ", "ħħ"], "خ": ["x", "xx"],
        "د": ["d", "dd"], "ذ": ["ð", "ðð"],
        "ر": ["r", "rr", "rˤ", "rˤrˤ"], "ز": ["z", "zz"],
        "س": ["s", "ss"], "ش": ["ʃ", "ʃʃ"],
        "ص": ["sˤ", "sˤsˤ"], "ض": ["dˤ", "dˤdˤ"],
        "ط": ["tˤ", "tˤtˤ"], "ظ": ["ðˤ", "ðˤðˤ"],
        "ع": ["ʕ", "ʕʕ"], "غ": ["ɣ"],
        "ف": ["f", "ff"], "ق": ["q", "qq"], "ك": ["k", "kk"],
        "ل": ["l", "ll", "lˤlˤ"],
        "م": ["m", "m̃", "ŋ"],   # ŋ (iqlab's inserted hum) is articulately
                                  # closer to a nasalised bilabial than to n
        "ن": ["n", "ñ"], "ه": ["h", "hh"],
        "ة": ["h"],               # taa marbuta -- realised as h in citation
                                   # recitation; a reasonable default given
                                   # the ambiguity between that and t/silent
        # waw and yaa: LONG vowel + consonant-glide roles only -- the SHORT
        # vowel is now covered by damma/kasra below, so it is deliberately
        # not duplicated here (sharper evidence, less conflation)
        "و": ["u:", "w", "ww", "w̃"],
        "ي": ["i:", "j", "jj", "j̃"],
        # alef and alef maksura: long-a vowel carriers only
        "ا": ["a:", "aˤ:"], "ى": ["a:", "aˤ:"],
        # hamza and its carrier composites all point at the glottal stop;
        # alef-madda additionally supports the long-a vowel, since it
        # phonetically realises hamza immediately followed by a long a
        # (madd badal)
        "ء": ["ʔ"], "أ": ["ʔ"], "إ": ["ʔ"], "ؤ": ["ʔ"], "ئ": ["ʔ"],
        "آ": ["ʔ", "a:", "aˤ:"],
        # SHORT vowels and tanween -- this tokenizer's diacritic tokens are
        # exactly the fix for the "short vowels have no letter" problem.
        # Tanween realises as vowel+n acoustically; for this crude bootstrap
        # mapping it to the vowel quality alone is a safe, defensible
        # simplification (the following consonant's own evidence, plus the
        # target sequence's own nasal tokens, carry the rest).
        "َ": ["a", "aˤ"], "ُ": ["u"], "ِ": ["i"],
        "ً": ["a", "aˤ"], "ٌ": ["u"], "ٍ": ["i"],
        # shadda (gemination) and sukun (absence of a vowel) are modifiers,
        # not sounds in their own right, and are deliberately left with NO
        # destination -- inferring gemination from shadda's co-occurrence
        # timing with the preceding consonant's own frame is a reasonable
        # next improvement, but it adds real complexity and is out of scope
        # here; each geminated token already gets support from its base
        # consonant's own letter above, which is the safer, simpler choice.
        # tatweel (ـ), hyphen (-), and the word-delimiter (|) carry no
        # phonemic content of their own and are treated as blank below.
    }
    dest_of: dict = {ch: [TOKEN2ID[t] for t in toks if t in TOKEN2ID]
                     for ch, toks in ar2toks.items()}
    BLANK_CHARS = {"<pad>", "<s>", "</s>", "|", "-", "ـ", "ّ", "ْ"}

    m: List[List[int]] = [[] for _ in range(max(hf_vocab.values()) + 1)]
    for ch, i in hf_vocab.items():
        if ch in BLANK_CHARS:
            m[i] = [0]
        elif ch in dest_of:
            m[i] = dest_of[ch]
    return m


def _project(lp: np.ndarray, m: List[List[int]]) -> np.ndarray:
    """
    m[src] is now a LIST of destination token ids (possibly empty) -- every
    destination in that list receives the full log-probability mass of the
    source character independently, via logaddexp accumulation across every
    source letter that could plausibly have produced it. A destination with
    no contributing source stays at the -30 floor (~0 probability), which is
    now the rare case rather than the majority one.
    """
    out = np.full((lp.shape[0], len(VOCAB)), -30.0, dtype=np.float32)
    for src, dsts in enumerate(m):
        for dst in dsts:
            out[:, dst] = np.logaddexp(out[:, dst], lp[:, src])
    return out - np.log(np.exp(out).sum(1, keepdims=True))


def load_seed_model(name: str, device: str = "cuda") -> SeedModel:
    return SeedModel(name, device)


def forced_align(model: SeedModel, feats: np.ndarray,
                 tokens: Sequence[str]) -> List[Tuple[str, int, int]]:
    lp = model.log_probs(feats)
    ids = [TOKEN2ID[t] for t in tokens if t in TOKEN2ID]
    return [(VOCAB[t], s, e) for (t, s, e) in _viterbi(lp, ids)]

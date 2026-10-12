"""
Unit + benchmark tests for Tajweed-QuartzNet on the Quran-Lab ASR benchmark.


Setup
-----
    This dataset required the token to checkout. Go to https://huggingface.co/datasets/Quran-Lab/quranic-asr-benchmark
    to request the access right for your token
    
    export HF_TOKEN=hf_xxx
Run
---
    pytest app/test_model_benchmark.py -v -o log_cli=true -o log_cli_level=INFO

Environment variables
---------------------
    TAJWEED_CKPT          checkpoint (required for the benchmark tests)
    TAJWEED_MODEL_FILE    path to the file defining TajweedQuartzNet
                          (default: auto-discovered under the project root)
    TAJWEED_DEVICE        cpu | cuda (default: cuda if available)
    TAJWEED_FEATURE_FN    optional "pkg.module:function" to replace src.utils.log_mel.log_mel
    TAJWEED_BLANK_ID      CTC blank id (default: looked up in data.tokens.TOKEN2ID, else 0)
    BENCH_DATASET         default Quran-Lab/quranic-asr-benchmark
    BENCH_CONFIGS         default everyayah_heldout,qul_alnufais,tlog_holdout
    BENCH_MAX_CLIPS       clips per config, evenly spaced (default 10, 0 = all 200)
    GOP_THRESHOLD         a token is flagged if its GOP-LPR < this (default -1.0)
    BENCH_MAX_RTF         fail a clip if real-time factor exceeds this (default off)
    BENCH_MAX_PER         fail a clip if PER exceeds this (default off)
    BENCH_OUT_DIR         where logs/results go (default <root>/bench_results)

Reference tokens
----------------
Each clip's Arabic `text` is resolved to (surah, ayah) with data.data_translator
(match_text, then match_text_ordered, then the surah/ayah in the clip id) and
converted to tokens with data.tajweed_g2p.TajweedG2P. The G2P RuleSpans are used
to report GOP per tajweed rule.

Outputs (in BENCH_OUT_DIR): bench_<ts>.jsonl (one row per clip), bench_<ts>.log,
summary_<ts>.json.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("vmx_quran.bench")
log.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.tokens import TOKEN2ID, ID2TOKEN, VOCAB, VOCAB_SIZE
from src.utils.log_mel import N_MELS, SR, log_mel
from data.data_translator import match_text, match_text_ordered
from data.tajweed_g2p import G2PResult, TajweedG2P

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATASET_ID = os.environ.get("BENCH_DATASET", "Quran-Lab/quranic-asr-benchmark")
CONFIGS = [c.strip() for c in os.environ.get(
    "BENCH_CONFIGS", "everyayah_heldout,qul_alnufais,tlog_holdout").split(",") if c.strip()]
MAX_CLIPS = int(os.environ.get("BENCH_MAX_CLIPS", "10"))
CKPT = os.environ.get("TAJWEED_CKPT", "exp/full/best.pt")
MODEL_FILE = os.environ.get("TAJWEED_MODEL_FILE", "src/model/tajweed_quartznet.py")
DEVICE = os.environ.get("TAJWEED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = int(SR)
GOP_THRESHOLD = float(os.environ.get("GOP_THRESHOLD", "-1.0"))
MAX_RTF = float(os.environ.get("BENCH_MAX_RTF", "0"))
MAX_PER = float(os.environ.get("BENCH_MAX_PER", "0"))
OUT_DIR = Path(os.environ.get("BENCH_OUT_DIR", str(ROOT / "bench_results")))

_SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__",
              "site-packages", "tests", "bench_results"}


# --------------------------------------------------------------------------
# Locating / loading the model and token utilities
# --------------------------------------------------------------------------
def _find_model_file() -> Path:
    if MODEL_FILE:
        return Path(MODEL_FILE)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            p = Path(dirpath) / fn
            try:
                if "class TajweedQuartzNet" in p.read_text(encoding="utf-8", errors="ignore"):
                    return p
            except OSError:
                continue
    raise FileNotFoundError(
        "Could not find a file defining TajweedQuartzNet under %s; set TAJWEED_MODEL_FILE." % ROOT)


@lru_cache(maxsize=1)
def model_module():
    path = _find_model_file()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("tajweed_model_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # required for @dataclass inside the module
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Tokens (data.tokens)
# --------------------------------------------------------------------------
BLANK = TOKEN2ID["<b>"]
assert BLANK == 0, "CTCLoss's blank index and this script's BLANK constant must agree"

def blank_id() -> int:
    return BLANK

def token_name(i: int) -> str:
    try:
        return str(ID2TOKEN[i])
    except (KeyError, IndexError, TypeError):
        return str(i)


# --------------------------------------------------------------------------
# Reference text -> (surah, ayah) -> phonemes (data_translator + TajweedG2P)
# --------------------------------------------------------------------------
def _as_refs(obj) -> List[Tuple[int, int]]:
    """Interpret what match_text / match_text_ordered return as [(surah, ayah), ...]."""
    if obj is None:
        return []
    if isinstance(obj, str):
        return [(int(s), int(a)) for s, a in re.findall(r"(\d{1,3})\s*[:_]\s*(\d{1,3})", obj)]
    if isinstance(obj, dict):
        s = obj.get("surah", obj.get("sura"))
        a = obj.get("ayah", obj.get("aya", obj.get("verse")))
        if s is not None and a is not None:
            return [(int(s), int(a))]
        for key in ("matches", "refs", "results", "ayahs"):
            if key in obj:
                return _as_refs(obj[key])
        return []
    if hasattr(obj, "surah"):
        a = getattr(obj, "ayah", getattr(obj, "verse", None))
        return [(int(obj.surah), int(a))] if a is not None else []
    if isinstance(obj, (tuple, list)):
        if len(obj) >= 2 and all(isinstance(x, (int, np.integer)) for x in obj[:2]):
            return [(int(obj[0]), int(obj[1]))]            # (surah, ayah[, score...])
        refs: List[Tuple[int, int]] = []
        for x in obj:
            refs.extend(_as_refs(x))
        return refs
    return []


def concat_g2p(parts: List[G2PResult]) -> G2PResult:
    """Join per-ayah G2P results (spans/words are re-based onto the joined token list)."""
    if len(parts) == 1:
        return parts[0]
    tokens: List[str] = []
    ids: List[int] = []
    spans, words, texts = [], [], []
    for p in parts:
        off, woff = len(tokens), len(words)
        tokens += p.tokens
        ids += p.ids
        spans += [replace(sp, start=sp.start + off, end=sp.end + off,
                          word_idx=sp.word_idx + woff) for sp in p.spans]
        words += [(a + off, b + off) for a, b in p.words]
        texts.append(p.text)
    return G2PResult(tokens=tokens, ids=ids, spans=spans, words=words, text=" ".join(texts))


def span_id_ranges(res: G2PResult) -> List[Optional[Tuple[int, int]]]:
    """
    RuleSpans index into `res.tokens`, but `res.ids` silently drops tokens missing from
    TOKEN2ID. Map each span to a [start, end) range over `res.ids` (None if all dropped).
    """
    pos: Dict[int, int] = {}
    k = 0
    for i, t in enumerate(res.tokens):
        if t in TOKEN2ID:
            pos[i] = k
            k += 1
    out: List[Optional[Tuple[int, int]]] = []
    for sp in res.spans:
        kept = [pos[i] for i in range(sp.start, sp.end) if i in pos]
        out.append((kept[0], kept[-1] + 1) if kept else None)
    return out


class RefResolver:
    """
    Clip -> reference G2PResult.

    Order: match_text(text) -> first (surah, ayah); else match_text_ordered(text) -> ordered
    list of ayahs (multi-ayah / repeated recitations); else the surah_ayah in the clip id.
    """

    def __init__(self):
        self.g2p = TajweedG2P()
        self.last_error: Optional[str] = None
        self._cache: Dict[Tuple[str, str], Tuple[Optional[G2PResult], str, list]] = {}

    def _try(self, fn, text: str, first_only: bool) -> List[Tuple[int, int]]:
        try:
            refs = _as_refs(fn(text))
        except Exception as e:  # noqa: BLE001 - must never break the run
            self.last_error = f"{fn.__name__}: {type(e).__name__}: {e}"
            return []
        if not refs:
            self.last_error = f"{fn.__name__}: returned nothing usable"
        return refs[:1] if first_only else refs

    def refs_for(self, config: str, clip_id: str, text: str) -> Tuple[List[Tuple[int, int]], str]:
        if text:
            refs = self._try(match_text, text, True)
            if refs:
                return refs, "match_text"
            refs = self._try(match_text_ordered, text, False)
            if refs:
                return refs, "match_text_ordered"
        m = re.search(r"(?<!\d)(\d{1,3})[_:](\d{1,3})(?!\d)", clip_id.split("/")[-1])
        if m:
            return [(int(m.group(1)), int(m.group(2)))], "clip_id"
        return [], "none"

    def resolve(self, config: str, clip_id: str, text: str):
        """-> (G2PResult | None, how, refs)"""
        key = (config, clip_id)
        if key in self._cache:
            return self._cache[key]
        refs, via = self.refs_for(config, clip_id, text)
        res: Optional[G2PResult] = None
        if refs:
            try:
                res = concat_g2p([self.g2p(s, a) for s, a in refs])
            except Exception as e:  # noqa: BLE001
                self.last_error = f"TajweedG2P{refs}: {type(e).__name__}: {e}"
        out = (res, via, refs)
        self._cache[key] = out
        return out


# --------------------------------------------------------------------------
# Features (src.utils.log_mel)
# --------------------------------------------------------------------------
def make_feature_fn() -> Callable[[torch.Tensor], torch.Tensor]:
    """wav (N,) float32 @SR -> (N_MELS, T), using the project's log_mel."""
    spec = os.environ.get("TAJWEED_FEATURE_FN")
    raw = log_mel
    if spec:
        mod, fn = spec.split(":")
        raw = getattr(importlib.import_module(mod), fn)

    def fn(wav: torch.Tensor) -> torch.Tensor:
        attempts = (lambda: raw(wav), lambda: raw(wav.numpy()), lambda: raw(wav, SR))
        out, err = None, None
        for attempt in attempts:
            try:
                out = attempt()
                break
            except (TypeError, AttributeError, ValueError) as e:
                err = e
        if out is None:
            raise err
        out = torch.as_tensor(out, dtype=torch.float32)
        while out.dim() > 2:                        # (1, n_mels, T) -> (n_mels, T)
            out = out.squeeze(0)
        if out.shape[0] != N_MELS and out.shape[1] == N_MELS:
            out = out.transpose(0, 1)               # (T, n_mels) -> (n_mels, T)
        return out.contiguous()

    return fn


# --------------------------------------------------------------------------
# CTC decode, forced alignment, GOP, edit distance
# --------------------------------------------------------------------------
def greedy_ctc_decode(log_probs: torch.Tensor, blank: int) -> List[int]:
    """log_probs (T, V) -> collapsed token ids."""
    out, prev = [], None
    for i in log_probs.argmax(-1).tolist():
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


_NEG = -1e30


def ctc_forced_align(log_probs: np.ndarray, targets: Sequence[int],
                     blank: int) -> Optional[Tuple[np.ndarray, float]]:
    """
    Viterbi forced alignment of `targets` to CTC posteriors (T, V).

    Returns (state_path (T,), score) or None if infeasible. State s is blank when
    s is even and token index (s-1)//2 when s is odd.
    """
    T = log_probs.shape[0]
    L = len(targets)
    if L == 0 or T == 0:
        return None
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = np.asarray(targets, dtype=np.int64)
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    emit = log_probs[:, ext]                       # (T, S)
    dp = np.full((T, S), _NEG)
    bp = np.zeros((T, S), dtype=np.int8)           # how many states we came from: 0/1/2
    dp[0, 0] = emit[0, 0]
    dp[0, 1] = emit[0, 1]
    idx = np.arange(S)
    for t in range(1, T):
        prev = dp[t - 1]
        c1 = np.concatenate(([_NEG], prev[:-1]))
        c2 = np.concatenate(([_NEG, _NEG], prev[:-2]))
        c2 = np.where(skip_ok, c2, _NEG)
        cand = np.stack([prev, c1, c2])
        best = cand.argmax(axis=0)
        dp[t] = cand[best, idx] + emit[t]
        bp[t] = best

    end = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    score = float(dp[T - 1, end])
    if score < _NEG / 2:
        return None
    path = np.empty(T, dtype=np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path, score


def compute_gop(log_probs: np.ndarray, targets: Sequence[int], blank: int,
                threshold: float = GOP_THRESHOLD) -> dict:
    """
    CTC-based Goodness of Pronunciation.

    1. Force-align `targets` to the posteriors (Viterbi over the CTC lattice).
    2. For every target token, take the frames the alignment assigned to it and compute:
         lpp    mean log P(token)                                  (raw posterior)
         lpp_nb mean log [P(token) / (1 - P(blank))]               (blank-normalised)
         lpr    mean [log P(token) - max_q log P(q)], q over ALL labels incl. blank
                (0 when the token wins its frames; very negative when it loses,
                 including when the aligner had to park it on a blank frame)
    The per-token `lpr` is the GOP score; tokens with lpr < threshold are flagged.
    """
    res = {"ok": False, "reason": "", "n_tokens": len(targets)}
    aligned = ctc_forced_align(log_probs, targets, blank)
    if aligned is None:
        res["reason"] = "alignment_infeasible (empty target or T < required frames)"
        return res
    path, score = aligned

    lpp, lpp_nb, lpr, nfr = [], [], [], []
    for i, tok in enumerate(targets):
        frames = np.nonzero(path == 2 * i + 1)[0]
        lp_t = log_probs[frames, tok]
        best = log_probs[frames].max(axis=1)
        p_blank = np.exp(log_probs[frames, blank]).clip(max=1 - 1e-6)
        lpp.append(float(lp_t.mean()))
        lpp_nb.append(float((lp_t - np.log1p(-p_blank)).mean()))
        lpr.append(float((lp_t - best).mean()))
        nfr.append(int(len(frames)))

    lpr_a = np.asarray(lpr)
    flagged = np.nonzero(lpr_a < threshold)[0].tolist()
    worst = np.argsort(lpr_a)[:5].tolist()

    # full-lattice CTC NLL (sum over all alignments), per target token
    nll = float(F.ctc_loss(
        torch.from_numpy(log_probs).float().unsqueeze(1),
        torch.tensor([list(targets)], dtype=torch.long),
        torch.tensor([log_probs.shape[0]]), torch.tensor([len(targets)]),
        blank=blank, reduction="sum", zero_infinity=True))

    res.update(
        ok=True, reason="", targets=list(targets),
        lpp=lpp, lpp_nb=lpp_nb, lpr=lpr, frames=nfr,
        mean=float(lpr_a.mean()), min=float(lpr_a.min()),
        lpp_mean=float(np.mean(lpp)), lpp_nb_mean=float(np.mean(lpp_nb)),
        flagged=flagged, frac_flagged=len(flagged) / len(targets),
        worst=[(int(i), int(targets[i]), float(lpr_a[i])) for i in worst],
        viterbi_score_per_token=score / len(targets),
        ctc_nll_per_token=nll / len(targets),
    )
    return res


def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------
# Dataset loading (at collection time, so each clip becomes its own test case)
# --------------------------------------------------------------------------
@dataclass
class Clip:
    config: str
    id: str
    text: str
    wav: np.ndarray
    duration_s: float


def _decode_audio(audio: dict) -> np.ndarray:
    import soundfile as sf
    if audio.get("bytes"):
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    else:
        data, sr = sf.read(audio["path"], dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio
        data = torchaudio.functional.resample(
            torch.from_numpy(data), sr, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(data, dtype=np.float32)


@lru_cache(maxsize=1)
def load_clips() -> Tuple[Tuple[Clip, ...], str]:
    """Returns (clips, skip_reason). Never raises: collection must not crash."""
    try:
        from datasets import Audio, load_dataset
    except Exception as e:  # noqa: BLE001
        return (), f"`datasets` not importable: {e}"
    clips: List[Clip] = []
    errors: List[str] = []
    for cfg in CONFIGS:
        try:
            ds = load_dataset(DATASET_ID, cfg, split="test")
            ds = ds.cast_column("audio", Audio(decode=False))
            n = len(ds)
            if MAX_CLIPS > 0 and n > MAX_CLIPS:
                sel = np.unique(np.linspace(0, n - 1, MAX_CLIPS).round().astype(int))
                ds = ds.select(sel.tolist())
            for i, row in enumerate(ds):
                wav = _decode_audio(row["audio"])
                clips.append(Clip(cfg, str(row.get("id", f"{cfg}/{i}")), row.get("text", "") or "",
                                  wav, len(wav) / SAMPLE_RATE))
        except Exception as e:  # noqa: BLE001 - gated repo, no network, missing token...
            errors.append(f"{cfg}: {type(e).__name__}: {str(e)[:200]}")
    if not clips:
        return (), ("could not load %s (%s). The dataset is gated: accept its terms on the "
                    "Hub and set HF_TOKEN." % (DATASET_ID, "; ".join(errors) or "no rows"))
    return tuple(clips), ""


_CLIPS, _SKIP_REASON = load_clips()
CLIP_PARAMS = [pytest.param(c, id=f"{c.config}:{c.id}") for c in _CLIPS] or [
    pytest.param(None, id="no-data", marks=pytest.mark.skip(reason=_SKIP_REASON or "no clips"))]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mod():
    return model_module()


@pytest.fixture()
def fresh_model(mod):
    """Randomly initialised model on CPU for structural tests."""
    torch.manual_seed(0)
    return mod.build_model().eval()


def _load_checkpoint(model: nn.Module, path: str) -> None:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # checkpoints holding pickled configs need weights_only=False
        ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for key in ("model", "model_state_dict", "state_dict", "ema"):
            if key in ck and isinstance(ck[key], dict):
                ck = ck[key]
                break
    sd = {}
    for k, v in ck.items():
        for pref in ("module.", "_orig_mod."):
            if k.startswith(pref):
                k = k[len(pref):]
        sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("verifier.")]   # verifier is optional
    if missing or unexpected:
        pytest.fail(f"checkpoint mismatch: missing={missing[:5]} unexpected={list(unexpected)[:5]}")


@pytest.fixture(scope="session")
def trained_model(mod):
    if not CKPT:
        pytest.skip("set TAJWEED_CKPT to run the benchmark tests")
    model = mod.build_model()
    _load_checkpoint(model, CKPT)
    model.to(DEVICE).eval()
    # warm-up so first-call overhead (cuDNN autotune, lazy init) is not billed to clip #1
    with torch.inference_mode():
        for _ in range(3):
            x = torch.randn(1, N_MELS, 500, device=DEVICE)
            model(x, torch.tensor([500], device=DEVICE))
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    return model


@pytest.fixture(scope="session")
def feature_fn():
    return make_feature_fn()


@pytest.fixture(scope="session")
def ref_resolver():
    return RefResolver()


class Collector:
    def __init__(self, fh):
        self.rows: List[dict] = []
        self.fh = fh

    def add(self, row: dict) -> None:
        self.rows.append(row)
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fh.flush()


@pytest.fixture(scope="session")
def results(request):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fh = open(OUT_DIR / f"bench_{ts}.jsonl", "w", encoding="utf-8")
    handler = logging.FileHandler(OUT_DIR / f"bench_{ts}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    col = Collector(fh)
    yield col
    fh.close()
    log.removeHandler(handler)
    handler.close()
    if not col.rows:
        return

    def agg(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    def rule_table(rows):
        acc: Dict[str, List[dict]] = {}
        for r in rows:
            for rs in r.get("gop_ref_rules") or []:
                acc.setdefault(rs["rule"], []).append(rs)
        return {rule: {"n": len(v), "gop_mean": float(np.mean([x["gop_mean"] for x in v])),
                       "flagged_frac": float(np.mean([x["flagged"] for x in v]))}
                for rule, v in sorted(acc.items())}

    summary = {}
    for cfg in sorted({r["config"] for r in col.rows}):
        rows = [r for r in col.rows if r["config"] == cfg]
        tot = [r["t_total_ms"] for r in rows]
        summary[cfg] = {
            "n": len(rows),
            "audio_s": sum(r["duration_s"] for r in rows),
            "total_ms_mean": float(np.mean(tot)),
            "total_ms_p50": float(np.percentile(tot, 50)),
            "total_ms_p95": float(np.percentile(tot, 95)),
            "infer_ms_mean": agg(rows, "t_infer_ms"),
            "rtf_mean": agg(rows, "rtf"),
            "per_mean": agg(rows, "per"),
            "gop_ref_mean": agg(rows, "gop_ref_mean"),
            "gop_self_mean": agg(rows, "gop_self_mean"),
            "flagged_ref_frac": agg(rows, "gop_ref_frac_flagged"),
            "ref_resolved": sum(1 for r in rows if r["ref_available"]),
            "rule_gop": rule_table(rows),
        }
    (OUT_DIR / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2))

    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        tr.write_sep("=", f"Tajweed benchmark summary (device={DEVICE}) -> {OUT_DIR}")
        for cfg, s in summary.items():
            fmt = lambda x, p=3: "n/a" if x is None else f"{x:.{p}f}"  # noqa: E731
            tr.write_line(
                f"{cfg:20s} n={s['n']:3d}  total ms mean/p50/p95="
                f"{s['total_ms_mean']:.0f}/{s['total_ms_p50']:.0f}/{s['total_ms_p95']:.0f}  "
                f"RTF={fmt(s['rtf_mean'], 4)}  PER={fmt(s['per_mean'])}  "
                f"GOP(ref)={fmt(s['gop_ref_mean'])}  GOP(self)={fmt(s['gop_self_mean'])}")


# --------------------------------------------------------------------------
# 1) Model structure unit tests
# --------------------------------------------------------------------------
class TestModelStructure:
    def test_forward_shapes_and_normalisation(self, fresh_model, mod):
        V = mod.ModelSpec.default().vocab
        x = torch.randn(2, N_MELS, 800)
        with torch.no_grad():
            lp, enc, out_lens = fresh_model(x, torch.tensor([800, 641]))
        assert lp.shape == (2, 400, V)
        assert enc.shape == (2, 768, 400)
        assert out_lens.tolist() == [400, 321]
        assert torch.isfinite(lp).all()
        assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 400), atol=1e-4)

    @pytest.mark.parametrize("T", [101, 400, 777, 1600])
    def test_output_length_formula_matches_conv_arithmetic(self, fresh_model, T):
        with torch.no_grad():
            lp, _, out_lens = fresh_model(torch.randn(1, N_MELS, T), torch.tensor([T]))
        assert lp.shape[1] == int(out_lens[0]) == (T + 1) // 2

    def test_stride_and_receptive_field(self, fresh_model):
        assert fresh_model.total_stride == 2                      # 50 fps encoder
        rf_s = fresh_model.receptive_field_frames * 0.010
        assert 5.0 <= rf_s <= 7.0, f"RF {rf_s:.2f}s, design target is ~6 s"

    def test_parameter_count_matches_docstring(self, mod, fresh_model):
        V = mod.ModelSpec.default().vocab
        total = fresh_model.num_parameters()
        encoder_ctc = total - sum(p.numel() for p in fresh_model.verifier.parameters())
        verifier = total - encoder_ctc
        # docstring: 3,764,441 (encoder+CTC @ V=73) + 255,380 (verifier)
        assert verifier == 255_380
        assert encoder_ctc == 3_764_441 + 769 * (V - 73)

    def test_uses_groupnorm_not_batchnorm(self, fresh_model):
        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in fresh_model.modules())
        gns = [m for m in fresh_model.modules() if isinstance(m, nn.GroupNorm)]
        assert gns and all(g.num_channels % g.num_groups == 0 for g in gns)

    def test_eval_is_deterministic(self, fresh_model):
        x = torch.randn(1, N_MELS, 500)
        with torch.no_grad():
            a, _, _ = fresh_model(x)
            b, _, _ = fresh_model(x)
        assert torch.equal(a, b)

    def test_batch_independence(self, fresh_model):
        """An utterance's output must not depend on what else is in the batch."""
        torch.manual_seed(1)
        a, b = torch.randn(1, N_MELS, 600), torch.randn(1, N_MELS, 600) * 3 + 1
        with torch.no_grad():
            alone, _, _ = fresh_model(a)
            both, _, _ = fresh_model(torch.cat([a, b]))
        assert torch.allclose(alone[0], both[0], atol=1e-4)

    def test_ctc_gradients_flow(self, mod):
        torch.manual_seed(0)
        model = mod.build_model().train()
        V = mod.ModelSpec.default().vocab
        x = torch.randn(2, N_MELS, 300)
        lp, _, out_lens = model(x, torch.tensor([300, 300]))
        targets = torch.randint(1, V, (2, 8))
        loss = F.ctc_loss(lp.transpose(0, 1), targets, out_lens,
                          torch.tensor([8, 8]), blank=blank_id(), zero_infinity=True)
        loss.backward()
        assert torch.isfinite(loss)
        for name, p in model.named_parameters():
            if name.startswith("verifier."):
                continue
            assert p.grad is not None and torch.isfinite(p.grad).all(), name

    def test_verifier_shapes(self, fresh_model):
        feats = torch.randn(3, 20, 768)
        extra = torch.randn(3, 6)
        with torch.no_grad():
            out = fresh_model.verifier(feats, extra, torch.tensor([20, 10, 5]))
        assert out.shape == (3, 8) and torch.isfinite(out).all()


class TestPaddingRobustness:
    """
    Informational (xfail, non-strict): documents whether zero-padding changes an
    utterance's own outputs. They flip to XPASS once the behaviour is fixed.
    """

    @pytest.mark.xfail(strict=False, reason="GroupNorm stats include the utterance's own padded frames")
    def test_encoder_invariant_to_trailing_padding(self, fresh_model):
        torch.manual_seed(2)
        x = torch.randn(1, N_MELS, 800)
        padded = torch.cat([x, torch.zeros(1, N_MELS, 400)], dim=-1)
        with torch.no_grad():
            a, _, _ = fresh_model(x)
            b, _, _ = fresh_model(padded)
        # compare the first 3 s of frames (far from the padded edge)
        assert torch.allclose(a[0, :150], b[0, :150], atol=1e-3)

    @pytest.mark.xfail(strict=False, reason="BiGRU backward direction reads padded frames (no packing)")
    def test_verifier_invariant_to_padding(self, fresh_model):
        torch.manual_seed(3)
        feats = torch.randn(1, 10, 768)
        extra = torch.randn(1, 6)
        padded = torch.cat([feats, torch.randn(1, 10, 768)], dim=1)
        with torch.no_grad():
            a = fresh_model.verifier(feats, extra, torch.tensor([10]))
            b = fresh_model.verifier(padded, extra, torch.tensor([10]))
        assert torch.allclose(a, b, atol=1e-4)


class TestFeatures:
    def test_feature_shape_and_finiteness(self):
        fn = make_feature_fn()
        wav = torch.randn(SAMPLE_RATE * 3) * 0.1
        f = fn(wav)
        assert f.shape[0] == N_MELS and torch.isfinite(f).all()
        assert abs(f.shape[1] - 300) <= 2          # 10 ms hop (encoder stride 2 -> 50 fps)

    def test_n_mels_matches_model(self, mod):
        assert N_MELS == mod.ModelSpec.default().n_mels


# --------------------------------------------------------------------------
# Project API wiring: tokens, G2P, text matching
# --------------------------------------------------------------------------
_G2P_SAMPLES = [(1, 1), (112, 1), (112, 4), (2, 19)]


class TestTokensAndG2P:
    def test_vocab_is_consistent_with_model(self, mod):
        assert VOCAB_SIZE == len(VOCAB) == len(TOKEN2ID) == mod.ModelSpec.default().vocab
        for tok, i in TOKEN2ID.items():
            assert token_name(i) == tok, f"TOKEN2ID/ID2TOKEN disagree on {tok!r}"

    def test_blank_id_is_valid_and_not_a_phoneme(self):
        b = blank_id()
        assert 0 <= b < VOCAB_SIZE, f"blank id {b} outside vocab"
        g = TajweedG2P()
        for s, a in _G2P_SAMPLES:
            assert b not in g(s, a).ids, (
                f"blank id {b} ({token_name(b)!r}) occurs in the reference of {s}:{a}; "
                "set TAJWEED_BLANK_ID to the real CTC blank")

    @pytest.mark.parametrize("surah,ayah", _G2P_SAMPLES)
    def test_g2p_output_is_usable(self, surah, ayah):
        r = TajweedG2P()(surah, ayah)
        assert r.tokens and r.ids and r.text
        # RuleSpans index into `tokens`; ids silently drops tokens missing from TOKEN2ID.
        assert len(r.ids) == len(r.tokens), "tokens missing from TOKEN2ID were dropped"
        assert all(0 <= i < VOCAB_SIZE for i in r.ids)
        assert all(0 <= sp.start < sp.end <= len(r.tokens) for sp in r.spans)
        assert all(sp.end <= len(r.tokens) for sp in r.spans)

    def test_concat_rebases_spans(self):
        g = TajweedG2P()
        a, b = g(112, 1), g(112, 2)
        c = concat_g2p([a, b])
        assert c.ids == a.ids + b.ids and len(c.tokens) == len(a.tokens) + len(b.tokens)
        assert [(s.rule, s.letters) for s in c.spans] == \
            [(s.rule, s.letters) for s in a.spans + b.spans]
        assert all(c.tokens[s.start:s.end] for s in c.spans)

    def test_match_text_round_trips_ayah_text(self):
        """Validates how this script interprets match_text()'s return value."""
        resolver = RefResolver()
        r = TajweedG2P()(112, 1)
        refs, via = resolver.refs_for("unit", "no-id", r.text)
        assert via in ("match_text", "match_text_ordered"), resolver.last_error
        assert refs[:1] == [(112, 1)], f"resolved {refs} via {via}"


# --------------------------------------------------------------------------
# 2) GOP algorithm unit tests (synthetic posteriors)
# --------------------------------------------------------------------------
def _peaky(T: int, V: int, emissions: Dict[int, int], blank: int = 0, p: float = 0.9) -> np.ndarray:
    """Posteriors that are mostly blank, with a confident token at the given frames."""
    P = np.full((T, V), (1 - p) / (V - 1))
    P[:, blank] = p
    for t, tok in emissions.items():
        P[t] = (1 - p) / (V - 1)
        P[t, tok] = p
    return np.log(P)


class TestGOP:
    def test_perfect_match_scores_zero_lpr(self):
        lp = _peaky(20, 6, {3: 1, 8: 2, 14: 3})
        r = compute_gop(lp, [1, 2, 3], blank=0)
        assert r["ok"] and r["frames"] == [1, 1, 1]
        assert all(abs(x) < 1e-9 for x in r["lpr"])
        assert r["flagged"] == []
        assert all(-0.2 < x <= 0 for x in r["lpp_nb"])

    def test_substituted_token_is_flagged(self):
        lp = _peaky(20, 6, {3: 1, 8: 2, 14: 3})
        r = compute_gop(lp, [1, 4, 3], blank=0)         # model heard 2, reference says 4
        assert r["ok"]
        assert r["flagged"] == [1]
        assert r["lpr"][1] < -3.0
        assert r["worst"][0][0] == 1

    def test_gop_scores_are_never_positive(self):
        rng = np.random.default_rng(0)
        lp = np.log(rng.dirichlet(np.ones(7), size=60))
        r = compute_gop(lp, [1, 2, 2, 5, 3], blank=0)
        assert r["ok"]
        assert max(r["lpr"]) <= 1e-9 and max(r["lpp"]) <= 1e-9 and max(r["lpp_nb"]) <= 1e-9

    def test_repeated_token_needs_a_separator_frame(self):
        lp = _peaky(6, 5, {1: 2, 4: 2})
        assert compute_gop(lp, [2, 2], blank=0)["ok"]
        assert not compute_gop(lp[:1], [2, 2], blank=0)["ok"]       # T=1 cannot emit 2 tokens
        assert not compute_gop(lp[:2], [2, 2], blank=0)["ok"]       # needs T >= 3 for "a _ a"

    def test_empty_target_is_rejected(self):
        assert not compute_gop(_peaky(5, 4, {}), [], blank=0)["ok"]

    def test_viterbi_path_is_monotonic_and_covers_all_tokens(self):
        lp = _peaky(30, 8, {2: 1, 9: 5, 17: 7, 25: 3})
        path, _ = ctc_forced_align(lp, [1, 5, 7, 3], blank=0)
        assert (np.diff(path) >= 0).all()
        assert {s for s in path if s % 2 == 1} == {1, 3, 5, 7}

    def test_greedy_decode_collapses_repeats_and_blanks(self):
        lp = torch.from_numpy(_peaky(8, 5, {1: 2, 2: 2, 4: 3, 6: 3}))
        assert greedy_ctc_decode(lp, blank=0) == [2, 3, 3]

    def test_edit_distance(self):
        assert edit_distance([1, 2, 3], [1, 2, 3]) == 0
        assert edit_distance([1, 2, 3], [1, 3]) == 1
        assert edit_distance([], [1, 2]) == 2


# --------------------------------------------------------------------------
# 3) Per-clip benchmark test (inference timing + GOP)
# --------------------------------------------------------------------------
def _sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def _gop_fields(prefix: str, g: Optional[dict]) -> dict:
    if g is None:
        return {f"{prefix}_ok": None, f"{prefix}_mean": None, f"{prefix}_min": None,
                f"{prefix}_frac_flagged": None}
    if not g["ok"]:
        return {f"{prefix}_ok": False, f"{prefix}_reason": g["reason"], f"{prefix}_mean": None,
                f"{prefix}_min": None, f"{prefix}_frac_flagged": None}
    return {
        f"{prefix}_ok": True, f"{prefix}_mean": g["mean"], f"{prefix}_min": g["min"],
        f"{prefix}_lpp_mean": g["lpp_mean"], f"{prefix}_lpp_nb_mean": g["lpp_nb_mean"],
        f"{prefix}_frac_flagged": g["frac_flagged"], f"{prefix}_n_flagged": len(g["flagged"]),
        f"{prefix}_ctc_nll_per_token": g["ctc_nll_per_token"],
        f"{prefix}_worst": g["worst"],
        f"{prefix}_lpr_per_token": [round(x, 3) for x in g["lpr"]],
    }


def rule_gop(res: G2PResult, lpr: Sequence[float], threshold: float = GOP_THRESHOLD) -> List[dict]:
    """Aggregate per-token GOP-LPR over each tajweed RuleSpan of the reference."""
    out: List[dict] = []
    for sp, rng in zip(res.spans, span_id_ranges(res)):
        if rng is None or rng[1] > len(lpr):
            continue
        seg = list(lpr[rng[0]:rng[1]])
        if not seg:
            continue
        out.append({
            "rule": sp.rule, "letters": sp.letters, "word_idx": sp.word_idx,
            "expected_harakat": sp.expected_harakat, "tok_start": rng[0], "tok_end": rng[1],
            "gop_mean": float(np.mean(seg)), "gop_min": float(np.min(seg)),
            "flagged": bool(np.min(seg) < threshold),
        })
    return out


def align_ops(ref: Sequence[int], hyp: Sequence[int]) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Levenshtein alignment -> [(op, ref_idx, hyp_idx)], op in ok / sub / del / ins."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                          D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops: List[Tuple[str, Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", i - 1, None))         # expected, but not recited
            i -= 1
        else:
            ops.append(("ins", None, j - 1))         # recited, but not expected
            j -= 1
    return ops[::-1]


def format_comparison(text: str, ref_tokens: Optional[Sequence[str]], ref_ids: Optional[Sequence[int]],
                      hyp_ids: Sequence[int], lpr: Optional[Sequence[float]],
                      rules: Sequence[dict], cols_per_block: int = 14) -> str:
    """Expected vs inferred text/phonemes, an aligned token view with per-token GOP, flagged rules."""
    L = ["  expected text    : " + (text or "(none)")]
    if ref_tokens:
        L.append("  expected phonemes: " + " ".join(ref_tokens))
    L.append("  inferred phonemes: " + " ".join(token_name(i) for i in hyp_ids))
    L.append("  (the model outputs phonemes only; the phoneme->Arabic is not implemented yet)")
    if not ref_ids:
        return "\n".join(L)

    ops = align_ops(ref_ids, hyp_ids)
    n = {k: sum(1 for o in ops if o[0] == k) for k in ("ok", "sub", "del", "ins")}
    L.append(f"  alignment: {n['ok']} match, {n['sub']} substituted, {n['del']} missing "
             f"(expected but not recited), {n['ins']} inserted")
    cols = []
    for op, i, j in ops:
        r = token_name(ref_ids[i]) if i is not None else "-"
        h = token_name(hyp_ids[j]) if j is not None else "-"
        g = f"{lpr[i]:.1f}" if (i is not None and lpr is not None) else ""
        mark = {"ok": "", "sub": "X", "del": "D", "ins": "I"}[op]
        w = max(len(r), len(h), len(g), 1)
        cols.append((r.ljust(w), h.ljust(w), g.ljust(w), mark.ljust(w)))
    for b in range(0, len(cols), cols_per_block):
        blk = cols[b:b + cols_per_block]
        L.append("  EXP : " + " ".join(c[0] for c in blk))
        L.append("  GOT : " + " ".join(c[1] for c in blk))
        L.append("  GOP : " + " ".join(c[2] for c in blk))
        L.append("  diff: " + " ".join(c[3] for c in blk).rstrip())
    L.append("        (X substituted, D missing, I inserted; GOP < "
             f"{GOP_THRESHOLD} means the expected phoneme lost to another label)")
    bad = sorted((r for r in rules if r["flagged"]), key=lambda r: r["gop_min"])[:8]
    if bad:
        L.append("  flagged tajweed rules: " + ", ".join(
            f"{r['rule']}({r['letters']})={r['gop_min']:.1f}" for r in bad))
    return "\n".join(L)


def run_clip(model, feature_fn, clip: Clip, blank: int, resolver: RefResolver) -> dict:
    wav = torch.from_numpy(clip.wav)

    # --- features ---
    _sync(); t0 = time.perf_counter()
    feats = feature_fn(wav).to(DEVICE)
    _sync(); t1 = time.perf_counter()

    # --- model forward (+ copy posteriors back to host) ---
    with torch.inference_mode():
        lp, _enc, out_lens = model(feats.unsqueeze(0), torch.tensor([feats.shape[-1]], device=DEVICE))
        lp0 = lp[0, : int(out_lens[0])].float().cpu()
    _sync(); t2 = time.perf_counter()

    # --- greedy CTC decode ---
    hyp = greedy_ctc_decode(lp0, blank)
    t3 = time.perf_counter()

    # --- GOP ---
    lp_np = lp0.numpy().astype(np.float64)
    gop_self = compute_gop(lp_np, hyp, blank) if hyp else None
    ref, ref_via, ref_refs = resolver.resolve(clip.config, clip.id, clip.text)
    log.info("text = %s, (surah,ayah) = %s via API=%s", clip.text, str(ref_refs), ref_via)
    #log.info("text = %s", clip.text)
    ref_ids = list(ref.ids) if ref is not None and ref.ids else None
    gop_ref = compute_gop(lp_np, ref_ids, blank) if ref_ids else None
    rules = rule_gop(ref, gop_ref["lpr"]) if gop_ref and gop_ref["ok"] else []
    t4 = time.perf_counter()

    per = (edit_distance(hyp, ref_ids) / len(ref_ids)) if ref_ids else None
    t_feat, t_fwd = (t1 - t0) * 1e3, (t2 - t1) * 1e3
    t_dec, t_gop, t_total = (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t4 - t0) * 1e3
    t_infer = t_feat + t_fwd + t_dec           # what a user waits for (GOP is the extra grading step)

    row = {
        "config": clip.config, "id": clip.id, "device": DEVICE,
        "duration_s": round(clip.duration_s, 3),
        "feat_frames": int(feats.shape[-1]), "n_frames": int(lp0.shape[0]),
        "t_feat_ms": round(t_feat, 2), "t_forward_ms": round(t_fwd, 2),
        "t_decode_ms": round(t_dec, 3), "t_gop_ms": round(t_gop, 2),
        "t_infer_ms": round(t_infer, 2), "t_total_ms": round(t_total, 2),
        "rtf": round(t_infer / 1e3 / clip.duration_s, 5),
        "finite": bool(np.isfinite(lp_np).all()),
        "prob_sum_err": float(np.abs(np.exp(lp_np).sum(-1) - 1).max()),
        "hyp_len": len(hyp), "hyp_ids": hyp,
        "hyp_tokens": " ".join(token_name(i) for i in hyp),
        "ref_text": clip.text, "ref_available": ref_ids is not None,
        "ref_via": ref_via, "ref_ayahs": [list(r) for r in ref_refs],
        "ref_tokens": " ".join(ref.tokens) if ref is not None else None,
        "ref_ids": ref_ids, "ref_len": len(ref_ids) if ref_ids else None, "per": per,
        "blank_frac": float((lp0.argmax(-1) == blank).float().mean()),
        "gop_ref_rules": rules,
    }
    row.update(_gop_fields("gop_self", gop_self))
    row.update(_gop_fields("gop_ref", gop_ref))
    if ref_ids is None:
        row["ref_error"] = resolver.last_error
    return row


@pytest.mark.parametrize("clip", CLIP_PARAMS)
def test_benchmark_clip(clip, trained_model, feature_fn, ref_resolver, results, record_property):
    r = run_clip(trained_model, feature_fn, clip, blank_id(), ref_resolver)
    results.add(r)

    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    log.info(
        "[%s] dur=%.2fs | feat %.1f + fwd %.1f + dec %.2f = infer %.1f ms (RTF %.4f) | gop %.1f ms "
        "| hyp=%d tok PER=%s | GOP ref=%s self=%s flagged(ref)=%s",
        f"{clip.config}:{clip.id}", clip.duration_s, r["t_feat_ms"], r["t_forward_ms"],
        r["t_decode_ms"], r["t_infer_ms"], r["rtf"], r["t_gop_ms"], r["hyp_len"], fmt(r["per"]),
        fmt(r.get("gop_ref_mean")), fmt(r.get("gop_self_mean")), fmt(r.get("gop_ref_frac_flagged")))
    if os.environ.get("BENCH_VERBOSE", "1") != "0":
        ref_tokens = r["ref_tokens"].split(" ") if r.get("ref_tokens") else None
        log.info("[%s] expected vs inferred\n%s", f"{clip.config}:{clip.id}",
                 format_comparison(clip.text, ref_tokens, r.get("ref_ids"), r["hyp_ids"],
                                   r.get("gop_ref_lpr_per_token"), r.get("gop_ref_rules") or []))
    for k in ("t_feat_ms", "t_forward_ms", "t_decode_ms", "t_gop_ms", "t_infer_ms", "rtf",
              "per", "gop_ref_mean", "gop_self_mean"):
        if r.get(k) is not None:
            record_property(k, r[k])

    # -- sanity: output is a valid posterior sequence of the right length ---------
    assert r["finite"], "non-finite log-probs"
    assert r["prob_sum_err"] < 1e-3, "posteriors are not normalised"
    assert abs(r["n_frames"] - math.ceil(r["feat_frames"] / 2)) <= 1
    assert r["hyp_len"] > 0, (
        "model emitted only blanks on real recitation; this usually means the feature "
        "pipeline differs from training (set TAJWEED_FEATURE_FN)")

    # -- GOP sanity ---------------------------------------------------------------
    assert r["gop_self_ok"], f"self-GOP alignment failed: {r.get('gop_self_reason')}"
    assert r["gop_self_mean"] <= 1e-9
    assert r["ref_available"], (
        f"no reference phonemes for {r['ref_text'][:60]!r} (via {r['ref_via']}): {r.get('ref_error')}")
    assert r["gop_ref_ok"], f"reference GOP alignment failed: {r.get('gop_ref_reason')}"
    assert math.isfinite(r["gop_ref_mean"]) and r["gop_ref_mean"] <= 1e-9

    # -- optional budgets ---------------------------------------------------------
    if MAX_RTF:
        assert r["rtf"] <= MAX_RTF, f"RTF {r['rtf']:.4f} > {MAX_RTF}"
    if MAX_PER and r["per"] is not None:
        assert r["per"] <= MAX_PER, f"PER {r['per']:.3f} > {MAX_PER}"

#!/usr/bin/env python3
"""
Train Tajweed-QuartzNet on build_dataset.py or build_dataset_without_verifier.py's output.
Two phases:
--------------------------------------------------------------------------
  --phase ctc        (default) Trains the acoustic encoder + CTC head from
                      audio + target token sequence alone. Does not touch
                      `align`/`spans` -- nn.CTCLoss computes its own
                      alignment internally.
  --phase verifier    Trains TajweedVerifier -- the head that actually
                      grades a tajweed rule occurrence as correct/too_short/
                      too_long/etc -- on top of a FROZEN, already-trained
                      CTC encoder (--init points at that checkpoint). Needs
                      `align` and `spans` from build_dataset.py's JSON.

Two modes:
--------------------------------------------------------------------------
  --light   A small, fixed subset (--light-n clips, default 20) for enough
            epochs/steps that a correctly-wired pipeline should nearly
            memorise it. A SANITY CHECK, not a real model -- see the
            "epochs vs steps" note below.
  --full    The full train split, evaluating on validation, with real
            checkpointing.

On epochs and optimizer steps for --light
--------------------------------------------
--light-n is small, so the number of PER-EPOCH gradient steps is tiny (at
light-n=20 and batch-size=16, roughly ONE step per epoch). What matters for
convergence is total STEPS, not epoch count -- a --light run that needs 400
epochs to converge at batch-size=16 is not broken, it needed ~400 steps,
which --full will reach within a small fraction of one real epoch once the
dataset is thousands of clips instead of 20. Don't carry a --light epoch
count over to --full; start --full at the script's own default and watch
the loss/PER curve.

Usage
-----
  # 1. CTC phase
  python src/training/train.py --data-dir data/everyayah_extracted --light --lr 5e-4  --epochs 500
  python src/training/train.py --data-dir data/everyayah_extracted --full --epochs 120

  # 2. Verifier phase (optional)
  python train.py --phase verifier --init exp/full/best.pt \
      --data-dir data/everyayah_extracted --light
  python train.py --phase verifier --init exp/full/best.pt \
      --data-dir data/everyayah_extracted --full

"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tokens import TOKEN2ID, VOCAB, VOCAB_SIZE
from src.model.tajweed_quartznet import build_model
from src.utils.log_mel import N_MELS, SR, log_mel
from src.utils.grader import CLASSES, FRAME_MS

BLANK = TOKEN2ID["<b>"]
assert BLANK == 0, "CTCLoss's blank index and this script's BLANK constant must agree"


# ===========================================================================
# Shared: dataset listing, checkpointing, resume
# ===========================================================================
def list_clip_jsons(data_dir: Path, split: str, limit: Optional[int] = None) -> List[Path]:
    """
    Sorted for determinism, so --light always picks the same clips on
    re-runs (and so a --light run and a later --full run, which necessarily
    includes those same clips, are directly comparable).
    """
    split_dir = data_dir / split
    if not split_dir.is_dir():
        sys.exit(f"missing {split_dir} -- run build_dataset.py first")
    paths = sorted(split_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def save_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer,
                    sched, epoch: int, phase: str, **metrics) -> None:
    """
    Saves everything needed for a TRUE resume, not just weights: optimizer
    state (Adam's momentum/variance buffers -- resuming without these causes
    a discontinuity right after resume, since the optimizer effectively
    forgets its own history) and the scheduler's position (so the LR
    schedule continues from where it was, rather than restarting at
    warmup).
    """
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "phase": phase,
        "vocab_size": VOCAB_SIZE,
        **metrics,
    }, path)


def find_resume_checkpoint(args, out_dir: Path) -> Optional[str]:
    """
    Explicit --resume always wins. Otherwise, auto-detect an interrupted run:
    if out_dir/last.pt already exists, resume from it rather than silently
    overwriting a run in progress -- matching this project's convention
    elsewhere (build_dataset.py, extract_everyayah.py are both auto-resumable
    without needing an explicit flag). --no-resume opts out, for the case
    where you genuinely want to start over in the same --out directory.
    """
    if args.resume:
        return args.resume
    if args.no_resume:
        return None
    auto = out_dir / "last.pt"
    if auto.exists():
        print(f"found an existing checkpoint at {auto} -- auto-resuming "
             f"(pass --no-resume to start fresh here instead)")
        return str(auto)
    return None


def load_for_resume(path: str, model: nn.Module, opt: torch.optim.Optimizer,
                    sched, expected_phase: str) -> int:
    """Returns the epoch to resume FROM (i.e. the next epoch to run)."""
    ck = torch.load(path, map_location="cpu")
    ckpt_vocab = ck.get("vocab_size")
    if ckpt_vocab is not None and ckpt_vocab != VOCAB_SIZE:
        sys.exit(f"cannot resume from {path}: it was trained with "
                 f"vocab_size={ckpt_vocab}, but the current vocabulary has "
                 f"VOCAB_SIZE={VOCAB_SIZE}. Token ids mean different things "
                 f"under each vocabulary -- resuming would silently corrupt "
                 f"training.")
    ckpt_phase = ck.get("phase")
    if ckpt_phase is not None and ckpt_phase != expected_phase:
        sys.exit(f"cannot resume from {path}: it is a {ckpt_phase!r}-phase "
                 f"checkpoint, but this is a {expected_phase!r}-phase run. "
                 f"For the verifier phase, point --init at a ctc checkpoint "
                 f"instead of --resume.")
    model.load_state_dict(ck["model"], strict=False)
    if ck.get("opt") is not None:
        try:
            opt.load_state_dict(ck["opt"])
        except (ValueError, KeyError) as e:
            print(f"  WARNING: could not restore optimizer state ({e}) -- "
                 f"continuing with freshly-initialised optimizer state. "
                 f"Training will still work, just without Adam's momentum "
                 f"carried over from before the interruption.", file=sys.stderr)
    if sched is not None and ck.get("sched") is not None:
        try:
            sched.load_state_dict(ck["sched"])
        except (ValueError, KeyError) as e:
            print(f"  WARNING: could not restore scheduler state ({e}) -- "
                 f"the LR schedule will restart from this epoch's step "
                 f"count instead of continuing seamlessly.", file=sys.stderr)
    epoch = ck.get("epoch", 0) + 1
    print(f"resumed from {path} at epoch {epoch}")
    return epoch


def make_cosine_lambda(total_steps: int, warm: int):
    """Linear warmup then cosine decay, indexed by OPTIMIZER step."""
    def lr_at(step):
        if step < warm:
            return step / warm
        p = min((step - warm) / max(total_steps - warm, 1), 1.0)   # clamp
        return 0.5 * (1 + math.cos(math.pi * p))
    return lr_at


def resolve_micro_batching(batch_size: int, micro_batch_size: int) -> Tuple[int, int]:
    """Returns (micro_bs, accum_steps). batch_size is the EFFECTIVE batch."""
    micro_bs = max(1, min(micro_batch_size, batch_size))
    accum_steps = math.ceil(batch_size / micro_bs)
    print(f"effective batch {batch_size} = {micro_bs} x {accum_steps} accumulation steps")
    return micro_bs, accum_steps


# ===========================================================================
# CTC phase: data
# ===========================================================================
class SpecAugment:
    """
    Frequency and time masking.

    Time masks are capped SHORT on purpose (see the project's data-pipeline
    docs): the standard 100-frame recipe is a full second, long enough to
    swallow an entire 4-harakat madd, which would teach the model to guess
    madd length from context rather than measure it -- exactly the skill a
    tajweed grader must not have, since a student's wrong madd is also
    contextually predictable. 15 frames (300ms) is the ceiling here.
    """

    def __init__(self, n_freq: int = 2, freq_width: int = 20,
                 n_time: int = 2, time_width: int = 15, p: float = 0.8):
        self.nf, self.fw, self.nt, self.tw, self.p = n_freq, freq_width, n_time, time_width, p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return x
        x = x.copy()
        F_, T = x.shape
        for _ in range(self.nf):
            w = random.randint(0, self.fw)
            if w and F_ > w:
                f0 = random.randint(0, F_ - w)
                x[f0:f0 + w, :] = 0.0
        for _ in range(self.nt):
            w = random.randint(0, self.tw)
            if w and T > w:
                t0 = random.randint(0, T - w)
                x[:, t0:t0 + w] = 0.0
        return x


class ClipDataset(Dataset):
    """
    One item per build_dataset.py JSON file. __getitem__ does the actual
    audio I/O and feature extraction (not __init__/listing), and returns
    None for a clip that turns out unreadable or too long -- collate_ctc
    below filters those out of the batch, the standard PyTorch pattern for
    a dataset with occasional bad items, so one corrupt file does not crash
    an entire epoch.
    """

    def __init__(self, json_paths: List[Path], train: bool = True,
                 max_sec: float = 20.0, speed_perturb: bool = True):
        self.json_paths = json_paths
        self.train = train
        self.max_sec = max_sec
        self.aug = SpecAugment() if train else None
        self.speed_perturb = speed_perturb and train

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, i):
        import soundfile as sf

        row = json.loads(self.json_paths[i].read_text(encoding="utf-8"))
        if row.get("label", "correct") != "correct":
            return None            # negatives belong to the verifier phase, not CTC
        tokens = row.get("tokens") or []
        ids = [TOKEN2ID[t] for t in tokens if t in TOKEN2ID]
        if not ids:
            return None
        if (len(ids) != len(tokens)):
            print(f"WARNING: some tokens were ignored, len(ids)={len(ids)}, len(tokens)={len(tokens)}")

        audio_path = row.get("audio")
        if not audio_path or not Path(audio_path).exists():
            return None
        try:
            wav, sr = sf.read(audio_path, dtype="float32")
        except Exception:                                  # noqa: BLE001
            return None
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if len(wav) / max(sr, 1) > self.max_sec:
            return None
        if sr != 16000:
            print("WARNING: sample rate is not 16Khz")

        if self.speed_perturb:
            # 0.9/1.0/1.1 resampling. Safe under CTC (unlike time-masking a
            # madd): it scales every duration in the utterance uniformly, so
            # it does not distort tajweed-relevant relative timing.
            factor = random.choice([0.9, 1.0, 1.1])
            if factor != 1.0:
                idx = np.arange(0, len(wav), factor)
                wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)

        feats = log_mel(wav, sr)
        if feats.shape[1] < 2:
            return None
        if self.aug is not None:
            feats = self.aug(feats)

        t_out = feats.shape[1] // 2          # matches the model's stride-2 prologue
        if t_out < len(ids):
            return None

        return (torch.from_numpy(feats), torch.tensor(ids, dtype=torch.long),
                row.get("uid", self.json_paths[i].stem))


def collate_ctc(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    feats, tgts, uids = zip(*batch)
    fl = torch.tensor([f.shape[1] for f in feats], dtype=torch.long)
    tl = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    F_, T = feats[0].shape[0], int(fl.max())
    x = torch.zeros(len(batch), F_, T)
    for i, f in enumerate(feats):
        x[i, :, :f.shape[1]] = f
    y = torch.cat(tgts)
    return x, fl, y, tl, uids


# ===========================================================================
# CTC phase: metrics, train/eval
# ===========================================================================
def edit_distance(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def per(hyps: List[List[int]], refs: List[List[int]]) -> float:
    num = sum(edit_distance(h, r) for h, r in zip(hyps, refs))
    den = sum(len(r) for r in refs)
    return num / max(den, 1)


def greedy_decode_batch(log_probs: torch.Tensor, lens: torch.Tensor) -> List[List[int]]:
    best = log_probs.argmax(-1).cpu().numpy()
    out = []
    for i, L in enumerate(lens.cpu().tolist()):
        seq, prev = [], -1
        for k in best[i, :L]:
            k = int(k)
            if k != prev and k != BLANK:
                seq.append(k)
            prev = k
        out.append(seq)
    return out


@torch.no_grad()
def evaluate(model, dl, ctc, device) -> Tuple[float, float]:
    model.eval()
    tot, seen, hyps, refs = 0.0, 0, [], []
    for batch in dl:
        if batch is None:
            continue
        x, fl, y, tl, _uids = batch
        x, fl, y, tl = x.to(device), fl.to(device), y.to(device), tl.to(device)
        logp, _enc, ol = model(x, fl)
        loss = ctc(logp.transpose(0, 1), y, ol, tl)
        tot += loss.item() * x.size(0)
        seen += x.size(0)

        hyp_batch = greedy_decode_batch(logp, ol)
        off = 0
        for L in tl.cpu().tolist():
            refs.append(y[off:off + L].cpu().tolist())
            off += L
        hyps.extend(hyp_batch)
    return tot / max(seen, 1), per(hyps, refs)


def train_one_epoch(model, dl, ctc, opt, sched, scaler, device, clip: float,
                    accum_steps: int = 1) -> float:
    """
    Gradient accumulation: `dl` yields micro-batches; gradients are summed
    over `accum_steps` of them and the optimizer (and LR scheduler) step once
    per window, so the effective batch size is micro_bs * accum_steps.
    """
    model.train()
    tot, seen, pending = 0.0, 0, 0
    n_batches = len(dl)
    opt.zero_grad(set_to_none=True)

    for i, batch in enumerate(dl):
        # size of the accumulation window this micro-batch belongs to
        # (the last window of an epoch may be smaller than accum_steps)
        window = min(accum_steps, n_batches - (i // accum_steps) * accum_steps)

        if batch is not None:
            x, fl, y, tl, _uids = batch
            x, fl, y, tl = x.to(device), fl.to(device), y.to(device), tl.to(device)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logp, _enc, ol = model(x, fl)
                loss = ctc(logp.transpose(0, 1), y, ol, tl)
            scaler.scale(loss / window).backward()   # average across the window
            tot += loss.item() * x.size(0)
            seen += x.size(0)
            pending += 1
            del logp, _enc, loss                      # free activations early

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            if pending:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                # GradScaler skips opt.step() on inf/NaN grads and lowers its
                # scale; only advance the LR schedule when a real step happened.
                if scaler.get_scale() >= scale_before:
                    sched.step()
            opt.zero_grad(set_to_none=True)
            pending = 0

    return tot / max(seen, 1)


# ===========================================================================
# Verifier phase: data
# ===========================================================================
class SpanDataset(Dataset):
    """
    One item per tajweed rule occurrence (span), not per clip -- a clip with
    5 tajweed rules produces 5 training items, each re-reading (and later
    re-encoding) that same clip's audio. Not the most compute-efficient
    design, but the simplest correct one, and verifier training is a small
    fraction of CTC training's cost regardless.

    Verdict labels: a span dict from build_dataset.py currently has no
    explicit "verdict" field -- every clip produced so far is labelled
    "correct" at the whole-clip level (see this module's docstring). Each
    span's verdict defaults to "correct" unless the span dict itself
    carries an explicit "verdict" key, which is how negative-synthesis data
    (once quran_asr/data/augment.py's corrupt_span is wired into
    build_dataset.py) is expected to be marked. `n_negative` on this
    dataset tells you, after construction, whether any non-"correct" spans
    were actually found -- check it before trusting a training run.
    """

    def __init__(self, json_paths: List[Path], max_sec: float = 20.0):
        self.items: List[Tuple[Path, int]] = []
        self.max_sec = max_sec
        self.n_total = 0
        self.n_negative = 0
        for p in json_paths:
            try:
                row = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            align = row.get("align")
            spans = row.get("spans")
            if not align or not spans:
                continue
            for i, sp in enumerate(spans):
                verdict = sp.get("verdict", "correct")
                if verdict not in CLASSES:
                    continue
                self.items.append((p, i))
                self.n_total += 1
                if verdict != "correct":
                    self.n_negative += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import soundfile as sf

        p, span_i = self.items[idx]
        row = json.loads(p.read_text(encoding="utf-8"))
        sp = row["spans"][span_i]
        align = row["align"]
        verdict = sp.get("verdict", "correct")
        cls = CLASSES.index(verdict)

        audio_path = row.get("audio")
        if not audio_path or not Path(audio_path).exists():
            return None
        try:
            wav, sr = sf.read(audio_path, dtype="float32")
        except Exception:                                  # noqa: BLE001
            return None
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if len(wav) / max(sr, 1) > self.max_sec:
            return None

        feats = log_mel(wav, sr)
        if feats.shape[1] < 2:
            return None

        s, e = sp["start"], min(sp["end"], len(align))
        if s >= len(align) or e <= s:
            f0, f1 = 0, 1
        else:
            f0, f1 = align[s][1], align[e - 1][2]

        harakah_ms = row.get("harakah_ms", 180.0)
        measured = sp.get("measured_harakat")
        if measured is None:
            measured = (f1 - f0) * FRAME_MS / max(harakah_ms, 1e-6)
        expected = sp.get("expected_harakat") or 1.0
        extra = np.array([
            measured, expected, measured / max(expected, 1e-6),
            (f1 - f0) * FRAME_MS / 1000.0, harakah_ms / 1000.0,
            float(hash(sp["rule"]) % 97) / 97.0,
        ], dtype=np.float32)

        return (torch.from_numpy(feats), f0, f1, torch.from_numpy(extra),
                torch.tensor(cls, dtype=torch.long))


def collate_span(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    feats, f0s, f1s, extras, ys = zip(*batch)
    F_, T = feats[0].shape[0], max(f.shape[1] for f in feats)
    x = torch.zeros(len(batch), F_, T)
    for i, f in enumerate(feats):
        x[i, :, :f.shape[1]] = f
    return (x, torch.tensor(f0s), torch.tensor(f1s),
            torch.stack(extras), torch.stack(ys))


def gather_spans(enc: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor,
                 max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice each item's rule span out of the encoder output and right-pad.
    f0/f1 are already in ENCODER-FRAME units (they came straight from
    build_dataset.py's `align`, itself produced by forced_align against a
    50fps encoder output), so no stride conversion is needed here."""
    B, C, T = enc.shape
    lens = (f1 - f0).clamp(min=1, max=max_len)
    out = enc.new_zeros(B, int(lens.max()), C)
    for i in range(B):
        a = int(max(0, min(int(f0[i]), T - 1)))
        b = int(max(a + 1, min(int(f1[i]), T, a + max_len)))
        out[i, :b - a] = enc[i, :, a:b].transpose(0, 1)
        lens[i] = b - a
    return out, lens


# ===========================================================================
# Verifier phase: metrics, train/eval
# ===========================================================================
def train_verifier_epoch(model, dl, crit, opt, sched, device, clip: float,
                         accum_steps: int = 1) -> float:
    model.verifier.train()
    model.blocks.eval()          # frozen encoder stays in eval mode: no
    model.ctc_head.eval()        # dropout noise in features the verifier
                                  # trains against, since the encoder's own
                                  # weights are not being updated here
    tot, seen, pending = 0.0, 0, 0
    n_batches = len(dl)
    opt.zero_grad(set_to_none=True)

    for i, batch in enumerate(dl):
        window = min(accum_steps, n_batches - (i // accum_steps) * accum_steps)
        if batch is not None:
            x, f0, f1, extra, y = batch
            x, f0, f1 = x.to(device), f0.to(device), f1.to(device)
            extra, y = extra.to(device), y.to(device)
            with torch.no_grad():
                enc = model.encode(x)
            spans, lens = gather_spans(enc, f0, f1)
            del enc
            logits = model.verifier(spans, extra, lens)
            loss = crit(logits, y)
            (loss / window).backward()
            tot += loss.item() * x.size(0)
            seen += x.size(0)
            pending += 1

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            if pending:
                nn.utils.clip_grad_norm_(model.verifier.parameters(), clip)
                opt.step()
                sched.step()
            opt.zero_grad(set_to_none=True)
            pending = 0

    return tot / max(seen, 1)


@torch.no_grad()
def evaluate_verifier(model, dl, crit, device) -> Tuple[float, float, float]:
    """Returns (loss, accuracy, macro_f1). Macro-F1, not accuracy, is the
    number to trust under class imbalance -- with today's all-"correct"
    data, a trivial always-"correct" classifier hits high accuracy while
    being useless, and macro-F1 over the classes that actually have support
    makes that visible rather than hiding it."""
    model.eval()
    tot, seen = 0.0, 0
    all_pred: List[int] = []
    all_true: List[int] = []
    for batch in dl:
        if batch is None:
            continue
        x, f0, f1, extra, y = batch
        x, f0, f1 = x.to(device), f0.to(device), f1.to(device)
        extra, y = extra.to(device), y.to(device)
        enc = model.encode(x)
        spans, lens = gather_spans(enc, f0, f1)
        logits = model.verifier(spans, extra, lens)
        loss = crit(logits, y)
        tot += loss.item() * x.size(0)
        seen += x.size(0)
        all_pred.extend(logits.argmax(-1).cpu().tolist())
        all_true.extend(y.cpu().tolist())

    acc = sum(p == t for p, t in zip(all_pred, all_true)) / max(len(all_true), 1)
    f1s = []
    for c in range(len(CLASSES)):
        tp = sum(1 for p, t in zip(all_pred, all_true) if p == c and t == c)
        fp = sum(1 for p, t in zip(all_pred, all_true) if p == c and t != c)
        fn = sum(1 for p, t in zip(all_pred, all_true) if p != c and t == c)
        if tp + fn == 0:
            continue          # class has no support in this eval set, skip it
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    macro_f1 = sum(f1s) / max(len(f1s), 1) if f1s else 0.0
    return tot / max(seen, 1), acc, macro_f1


# ===========================================================================
# CTC phase: run
# ===========================================================================
def run_ctc(args) -> None:
    device = torch.device(args.device)

    if args.light:
        epochs = args.epochs or 60
        batch_size = args.batch_size or min(8, args.light_n)
        out_dir = Path(args.out or "exp/light")
        train_paths = list_clip_jsons(Path(args.data_dir), args.train_split, limit=args.light_n)
        val_paths = train_paths
        print(f"[light] {len(train_paths)} clips (train == eval set on purpose)")
        if len(train_paths) < args.light_n:
            print(f"  WARNING: found only {len(train_paths)} clips in "
                 f"{args.data_dir}/{args.train_split}, wanted {args.light_n}.")
    else:
        epochs = args.epochs or 120
        batch_size = args.batch_size or 32
        out_dir = Path(args.out or "exp/full")
        train_paths = list_clip_jsons(Path(args.data_dir), args.train_split)
        val_paths = list_clip_jsons(Path(args.data_dir), args.val_split)
        print(f"[full] {len(train_paths)} train clips, {len(val_paths)} val clips")
        if not train_paths:
            sys.exit(f"no .json files in {args.data_dir}/{args.train_split} -- "
                     f"run build_dataset.py --full first")

    out_dir.mkdir(parents=True, exist_ok=True)

    micro_bs, accum_steps = resolve_micro_batching(batch_size, args.micro_batch_size)

    ds_train = ClipDataset(train_paths, train=not args.light, max_sec=args.max_sec)
    ds_val = ClipDataset(val_paths, train=False, max_sec=args.max_sec)
    dl_train = DataLoader(ds_train, batch_size=micro_bs, shuffle=True,
                          num_workers=args.workers, collate_fn=collate_ctc,
                          drop_last=len(ds_train) > micro_bs)
    dl_val = DataLoader(ds_val, batch_size=micro_bs, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_ctc)

    model = build_model(N_MELS, VOCAB_SIZE, with_verifier=False).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, "
         f"vocab={VOCAB_SIZE}")

    ctc = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))

    # The schedule counts OPTIMIZER steps, not micro-batches.
    steps_per_epoch = max(1, math.ceil(len(dl_train) / accum_steps))
    total_steps = epochs * steps_per_epoch
    warm = max(1, int(0.05 * total_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_cosine_lambda(total_steps, warm))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    resume_path = find_resume_checkpoint(args, out_dir)
    if resume_path:
        start_epoch = load_for_resume(resume_path, model, opt, sched, "ctc")

    best_per = float("inf")
    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, ctc, opt, sched, scaler,
                                  device, args.clip, accum_steps)

        if epoch % args.eval_every == 0 or epoch == epochs:
            val_loss, val_per = evaluate(model, dl_val, ctc, device)
            print(f"ep {epoch:4d}/{epochs} | train {tr_loss:.4f} | "
                 f"val {val_loss:.4f} | PER {val_per*100:5.1f}% | "
                 f"lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s")

            save_checkpoint(out_dir / "last.pt", model, opt, sched, epoch,
                            "ctc", per=val_per)
            if val_per < best_per:
                best_per = val_per
                save_checkpoint(out_dir / "best.pt", model, opt, sched, epoch,
                                "ctc", per=val_per)
        else:
            print(f"ep {epoch:4d}/{epochs} | train {tr_loss:.4f}")

    print(f"\n{'='*60}")
    if args.light:
        print(f"LIGHT RUN COMPLETE -- final PER on the {len(train_paths)}-clip "
             f"training set: {best_per*100:.1f}%")
        if best_per < 0.05:
            print("PER < 5%: the pipeline is wired correctly. Safe to move to --full.")
        elif best_per < 0.30:
            print("PER dropped but did not converge near 0. Check total "
                 "OPTIMIZER STEPS, not just epoch count -- with a small "
                 "--light-n and a batch-size close to it, each epoch may be "
                 "only 1-3 real steps. Try more epochs, or a larger --lr, "
                 "before suspecting a real bug.")
        else:
            print("PER did not meaningfully drop -- check, in order: (1) are "
                 "`tokens` in the JSON files non-empty and using the SAME "
                 "vocabulary as quran_asr.data.tokens right now (re-run "
                 "build_dataset.py if you've regenerated assets/vocab.json "
                 "since), (2) does audio at `audio` in each JSON actually "
                 "exist and play correctly, (3) is CTCLoss ever returning a "
                 "finite, non-exploding value in the printed train loss above.")
    else:
        print(f"FULL RUN COMPLETE -- best val PER: {best_per*100:.1f}%  "
             f"(checkpoint: {out_dir/'best.pt'})")
    print(f"{'='*60}")


# ===========================================================================
# Verifier phase: run
# ===========================================================================
def run_verifier(args) -> None:
    if not args.init:
        sys.exit("--phase verifier needs --init pointing at a trained CTC "
                 "checkpoint (e.g. exp/full/best.pt) -- the verifier reads "
                 "features from a frozen, already-trained encoder, it is "
                 "never trained from scratch on its own.")

    device = torch.device(args.device)

    if args.light:
        epochs = args.epochs or 60
        batch_size = args.batch_size or min(8, args.light_n)
        out_dir = Path(args.out or "exp/verifier_light")
        train_paths = list_clip_jsons(Path(args.data_dir), args.train_split, limit=args.light_n)
        val_paths = train_paths
        print(f"[light] {len(train_paths)} clips (train == eval set on purpose)")
    else:
        epochs = args.epochs or 30
        batch_size = args.batch_size or 32
        out_dir = Path(args.out or "exp/verifier_full")
        train_paths = list_clip_jsons(Path(args.data_dir), args.train_split)
        val_paths = list_clip_jsons(Path(args.data_dir), args.val_split)
        print(f"[full] {len(train_paths)} train clips, {len(val_paths)} val clips")
        if not train_paths:
            sys.exit(f"no .json files in {args.data_dir}/{args.train_split} -- "
                     f"run build_dataset.py --full first")

    out_dir.mkdir(parents=True, exist_ok=True)

    ds_train = SpanDataset(train_paths, max_sec=args.max_sec)
    ds_val = SpanDataset(val_paths, max_sec=args.max_sec)
    print(f"spans: {ds_train.n_total} train ({ds_train.n_negative} non-correct), "
         f"{ds_val.n_total} val ({ds_val.n_negative} non-correct)")

    if ds_train.n_negative == 0:
        print(f"\n{'!'*70}")
        print("WARNING: every span in this training set is labelled 'correct'.")
        print("This is expected right now -- build_dataset.py has not yet been")
        print("run through negative synthesis (quran_asr/data/augment.py's")
        print("corrupt_span). A verifier trained on ONLY-correct data has")
        print("literally never seen an error and WILL converge to always")
        print("predicting 'correct' -- that is what this data necessarily")
        print("produces, not a bug in this script. The resulting checkpoint")
        print("will report high accuracy and be useless as a grader. Add")
        print("negative examples before trusting anything this run produces.")
        print(f"{'!'*70}\n")
        if not args.allow_no_negatives:
            sys.exit("refusing to train a verifier with zero negative examples "
                     "(pass --allow-no-negatives to proceed anyway, e.g. purely "
                     "to sanity-check that the training loop itself runs)")

    if not ds_train.items:
        sys.exit("no usable spans found -- check that build_dataset.py's JSON "
                 "files actually have non-empty `align`/`spans` fields")

    micro_bs, accum_steps = resolve_micro_batching(batch_size, args.micro_batch_size)

    dl_train = DataLoader(ds_train, batch_size=micro_bs, shuffle=True,
                          num_workers=args.workers, collate_fn=collate_span,
                          drop_last=len(ds_train) > micro_bs)
    dl_val = DataLoader(ds_val, batch_size=micro_bs, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_span)

    model = build_model(N_MELS, VOCAB_SIZE, with_verifier=True).to(device)
    ck = torch.load(args.init, map_location="cpu")
    ckpt_vocab = ck.get("vocab_size")
    if ckpt_vocab is not None and ckpt_vocab != VOCAB_SIZE:
        sys.exit(f"--init checkpoint was trained with vocab_size={ckpt_vocab}, "
                 f"current VOCAB_SIZE={VOCAB_SIZE} -- these must match.")
    missing = model.load_state_dict(ck["model"], strict=False)
    print(f"loaded encoder+CTC weights from {args.init} "
         f"(missing: {missing.missing_keys or 'none'})")

    for p in model.blocks.parameters():
        p.requires_grad = False
    for p in model.ctc_head.parameters():
        p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"encoder + CTC head frozen; training {n_trainable:,} verifier parameters")

    # inverse-sqrt-frequency class weights: with real negative data present,
    # this stops the loss converging to "always predict the majority class"
    # (with today's all-correct data it is moot -- only one class has any
    # support at all -- but the machinery is correct for when negatives exist)
    counts = np.ones(len(CLASSES))
    for _p, i in ds_train.items:
        row = json.loads(_p.read_text(encoding="utf-8"))
        verdict = row["spans"][i].get("verdict", "correct")
        counts[CLASSES.index(verdict)] += 1
    weights = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)

    # The schedule counts OPTIMIZER steps, not micro-batches.
    steps_per_epoch = max(1, math.ceil(len(dl_train) / accum_steps))
    total_steps = epochs * steps_per_epoch
    warm = max(1, int(0.05 * total_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_cosine_lambda(total_steps, warm))

    start_epoch = 1
    resume_path = find_resume_checkpoint(args, out_dir)
    if resume_path:
        start_epoch = load_for_resume(resume_path, model, opt, sched, "verifier")

    best_f1 = -1.0
    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        tr_loss = train_verifier_epoch(model, dl_train, crit, opt, sched,
                                       device, args.clip, accum_steps)

        if epoch % args.eval_every == 0 or epoch == epochs:
            val_loss, val_acc, val_f1 = evaluate_verifier(model, dl_val, crit, device)
            print(f"ep {epoch:4d}/{epochs} | train {tr_loss:.4f} | "
                 f"val {val_loss:.4f} | acc {val_acc*100:5.1f}% | "
                 f"macro-F1 {val_f1*100:5.1f}% | "
                 f"lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s")

            save_checkpoint(out_dir / "last.pt", model, opt, sched, epoch,
                            "verifier", f1=val_f1, acc=val_acc)
            if val_f1 > best_f1:
                best_f1 = val_f1
                save_checkpoint(out_dir / "best.pt", model, opt, sched, epoch,
                                "verifier", f1=val_f1, acc=val_acc)
        else:
            print(f"ep {epoch:4d}/{epochs} | train {tr_loss:.4f}")

    print(f"\n{'='*60}")
    print(f"VERIFIER {'LIGHT' if args.light else 'FULL'} RUN COMPLETE -- "
         f"best macro-F1: {best_f1*100:.1f}%  (checkpoint: {out_dir/'best.pt'})")
    if ds_train.n_negative == 0:
        print("Remember: this run had zero negative examples, so this F1 "
             "number reflects a trivial classifier, not a real grader.")
    print(f"{'='*60}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["ctc", "verifier"], default="ctc")
    ap.add_argument("--data-dir", default="data/everyayah_extracted")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--light", action="store_true",
                      help="quick pipeline-verification run on a tiny subset")
    ap.add_argument("--light-n", type=int, default=20,
                    help="clips used for BOTH train and eval in --light mode "
                         "-- the goal is memorisation, not generalisation")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--epochs", type=int, default=None,
                    help="default: 60/120 for ctc --light/--full, 60/30 for "
                         "verifier --light/--full")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="EFFECTIVE batch size (reached via gradient "
                         "accumulation). default: min(8, light-n) for "
                         "--light, 32 for --full")
    ap.add_argument("--micro-batch-size", type=int, default=4,
                    help="clips per forward/backward pass on the GPU; lower "
                         "it if you hit out-of-memory errors")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--max-sec", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--out", default=None,
                    help="default: exp/light, exp/full, exp/verifier_light, "
                         "or exp/verifier_full depending on --phase/--light")
    ap.add_argument("--init", default=None,
                    help="[verifier phase, required] trained CTC checkpoint "
                         "to load the encoder from")
    ap.add_argument("--allow-no-negatives", action="store_true",
                    help="[verifier phase] proceed even though every span is "
                         "labelled 'correct' -- only useful to sanity-check "
                         "that the training loop itself runs, see the loud "
                         "warning this prints for why the result won't be a "
                         "useful grader")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from. If omitted, auto-resumes "
                         "from <out>/last.pt when it exists -- see --no-resume")
    ap.add_argument("--no-resume", action="store_true",
                    help="start fresh even if <out>/last.pt exists")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="run validation every N epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    (run_ctc if args.phase == "ctc" else run_verifier)(args)


if __name__ == "__main__":
    main()
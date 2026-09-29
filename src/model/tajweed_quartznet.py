"""
Tajweed-QuartzNet — the acoustic model.

Two heads on one encoder:
  * CTC head      -> tajweed phoneme tokens (vocab-size-way, from
                     quran_asr.data.tokens -- currently 73), gives text +
                     alignment
  * Verifier head -> per-rule quality classification, gives the grade

TOTAL: 3,764,441 (encoder+CTC) + 255,380 (verifier) = 4,019,821 parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tokens import VOCAB_SIZE

# ---------------------------------------------------------------------------
# Architecture spec — single source of truth, shared with the param counter
# ---------------------------------------------------------------------------
@dataclass
class BlockSpec:
    name: str
    in_ch: int
    out_ch: int
    kernel: int
    repeat: int
    stride: int = 1
    dilation: int = 1
    residual: bool = True
    separable: bool = True
    dropout: float = 0.1


@dataclass
class ModelSpec:
    n_mels: int = 80
    vocab: int = VOCAB_SIZE
    blocks: Tuple[BlockSpec, ...] = ()
    verifier_dim: int = 128
    n_rule_classes: int = 8

    @staticmethod
    def default(n_mels: int = 80, vocab: int = VOCAB_SIZE) -> "ModelSpec":
        return ModelSpec(
            n_mels=n_mels,
            vocab=vocab,
            blocks=(
                # prologue: stride 2 -> 20 ms per encoder frame (50 fps).
                # 50 fps is deliberate: a harakah is ~150-250 ms, so a 2-harakat
                # ghunnah spans ~15-25 frames. That is ample resolution to grade
                # madd length, while halving the compute of a 100 fps encoder.
                BlockSpec("C1", 80, 256, 13, 1, stride=2, residual=False),
                BlockSpec("B1", 256, 256, 17, 3),
                BlockSpec("B2", 256, 384, 21, 3),
                BlockSpec("B3", 384, 512, 25, 3),
                BlockSpec("B4", 512, 512, 29, 3),
                # epilogue. Kernels are sized so the total receptive field lands
                # at ~6 s: long enough to cover a 6-harakat madd (~1.5 s) plus a
                # word of context either side for idgham/ikhfa, without the 14 s
                # RF that QuartzNet-15x5's 33..87 kernels would give us.
                BlockSpec("C2", 512, 640, 33, 1, residual=False),
                BlockSpec("C3", 640, 768, 1, 1, residual=False, separable=False),
            ),
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
# GroupNorm, not BatchNorm1d: nn.BatchNorm1d normalizes each channel over
# ALL of (batch, time) jointly. collate_ctc zero-pads variable-length
# utterances to the batch's longest one, and BatchNorm has no notion of
# which frames are real audio and which are padding -- so at batch_size > 1
# with utterances of different lengths, its statistics are a mix of real
# signal and artificial zero-padding, in proportion to how much padding that
# particular batch happened to contain. This is not a small effect: it
# degrades the RUNNING statistics used at inference (model.eval()) even when
# the training loss still improves (CTCLoss correctly ignores padding via
# the lengths tensor, so loss can look fine while eval-time decoding is
# badly corrupted). Confirmed in practice on this exact model: a --light
# training run at batch_size=1 (no padding possible, since a single
# utterance needs no padding to match anyone else) converged cleanly to a
# few percent PER; the same run at batch_size=8 had loss dropping normally
# while decoded output stayed near-random.
#
# GroupNorm computes statistics per-sample (over channels-within-a-group and
# time, but never across the batch dimension), so one utterance's padding
# can never contaminate another's normalization -- the standard fix for
# exactly this class of problem in sequence models trained with padded
# batches. It also has no running statistics at all, so train() and eval()
# behave identically; the BatchNorm train/eval mismatch this was masking is
# gone as a side effect, not just the padding contamination.
#
# num_groups=32 divides every channel count in this architecture (256, 384,
# 512, 640, 768) evenly -- verified against ModelSpec.default() directly,
# not assumed.
NORM_GROUPS = 32


def make_norm(channels: int) -> nn.Module:
    groups = NORM_GROUPS if channels % NORM_GROUPS == 0 else 1
    return nn.GroupNorm(groups, channels)


class SeparableConv1d(nn.Module):
    """Depthwise (K) followed by pointwise (1x1). ~K/1 + 1/C_out of a dense conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 dilation: int = 1, separable: bool = True):
        super().__init__()
        pad = (kernel // 2) * dilation
        if separable and kernel > 1:
            self.depthwise = nn.Conv1d(in_ch, in_ch, kernel, stride=stride,
                                       padding=pad, dilation=dilation,
                                       groups=in_ch, bias=False)
            self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        else:
            self.depthwise = None
            self.pointwise = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                                       padding=pad, dilation=dilation, bias=False)
        self.bn = make_norm(out_ch)

    def forward(self, x):
        if self.depthwise is not None:
            x = self.depthwise(x)
        return self.bn(self.pointwise(x))


class JasperBlock(nn.Module):
    """`repeat` separable convs, residual added before the final activation."""

    def __init__(self, spec: BlockSpec):
        super().__init__()
        layers: List[nn.Module] = []
        c_in = spec.in_ch
        for r in range(spec.repeat):
            layers.append(SeparableConv1d(c_in, spec.out_ch, spec.kernel,
                                          stride=spec.stride if r == 0 else 1,
                                          dilation=spec.dilation,
                                          separable=spec.separable))
            c_in = spec.out_ch
        self.convs = nn.ModuleList(layers)
        self.drop = nn.Dropout(spec.dropout)
        self.residual = None
        if spec.residual:
            # 1x1 projection so the skip path matches channels; cheap, and it is
            # what lets us stack 12 separable convs without vanishing gradients.
            self.residual = nn.Sequential(
                nn.Conv1d(spec.in_ch, spec.out_ch, 1, bias=False),
                make_norm(spec.out_ch),
            )

    def forward(self, x):
        out = x
        for i, conv in enumerate(self.convs):
            out = conv(out)
            if i < len(self.convs) - 1:
                out = self.drop(F.relu(out))
        if self.residual is not None:
            out = out + self.residual(x)
        return self.drop(F.relu(out))


# ---------------------------------------------------------------------------
# Verifier head
# ---------------------------------------------------------------------------
class TajweedVerifier(nn.Module):
    """
    Grades one tajweed rule occurrence.

    Input is the encoder features pooled over the rule's aligned frames (from
    CTC forced alignment), concatenated with explicit duration features that the
    convnet cannot see directly (measured harakat, expected harakat, ratio).

    We keep a *separate* head rather than reading verdicts off the CTC posterior
    because CTC is trained only on correct recitation, so its posterior is not
    calibrated for errors. The verifier is trained on deliberate negatives.

    Classes: 0 correct | 1 too_short | 2 too_long | 3 missing | 4 substituted
             5 no_ghunnah | 6 weak_articulation | 7 inserted
    """

    def __init__(self, enc_dim: int = 768, hidden: int = 128, n_classes: int = 8,
                 n_extra: int = 6, bottleneck: int = 128, gru_hidden: int = 96):
        super().__init__()
        # Bottleneck first: a BiGRU straight off 768 channels costs 0.69M params
        # on its own, which would be 18% of the whole model for a head that only
        # ever sees ~20 frames. Projecting to 128 first costs 0.23M in total.
        self.proj = nn.Linear(enc_dim, bottleneck, bias=False)
        self.gru = nn.GRU(bottleneck, gru_hidden, num_layers=1,
                          batch_first=True, bidirectional=True)
        d = 2 * gru_hidden + n_extra
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, span_feats: torch.Tensor, extra: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # span_feats: (B, T_span, enc_dim)   extra: (B, n_extra)
        h, _ = self.gru(self.proj(span_feats))
        if lengths is not None:
            mask = (torch.arange(h.size(1), device=h.device)[None, :]
                    < lengths[:, None]).unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        return self.fc(self.norm(torch.cat([pooled, extra], dim=-1)))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class TajweedQuartzNet(nn.Module):
    def __init__(self, spec: Optional[ModelSpec] = None, with_verifier: bool = True):
        super().__init__()
        self.spec = spec or ModelSpec.default()
        self.blocks = nn.ModuleList([JasperBlock(b) for b in self.spec.blocks])
        enc_dim = self.spec.blocks[-1].out_ch
        self.enc_dim = enc_dim
        # CTC projection. Bias kept here (it is 64 numbers) because the blank
        # prior matters and BN upstream has no bias of its own on this path.
        self.ctc_head = nn.Conv1d(enc_dim, self.spec.vocab, 1, bias=True)
        self.verifier = TajweedVerifier(enc_dim, self.spec.verifier_dim,
                                        self.spec.n_rule_classes) if with_verifier else None

    # -- forward ------------------------------------------------------------
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, n_mels, T) -> (B, enc_dim, T')"""
        x = feats
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, feats: torch.Tensor,
                feat_lens: Optional[torch.Tensor] = None):
        enc = self.encode(feats)
        logits = self.ctc_head(enc)                    # (B, V, T')
        log_probs = F.log_softmax(logits, dim=1).transpose(1, 2)   # (B, T', V)
        out_lens = None
        if feat_lens is not None:
            out_lens = torch.div(feat_lens + 1, self.total_stride,
                                 rounding_mode="floor").clamp(min=1)
        return log_probs, enc, out_lens

    # -- properties ---------------------------------------------------------
    @property
    def total_stride(self) -> int:
        s = 1
        for b in self.spec.blocks:
            s *= b.stride
        return s

    @property
    def receptive_field_frames(self) -> int:
        """Encoder receptive field, in input mel frames."""
        rf, stride = 1, 1
        for b in self.spec.blocks:
            for r in range(b.repeat):
                rf += (b.kernel - 1) * b.dilation * stride
                stride *= b.stride if r == 0 else 1
        return rf

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)


def build_model(n_mels: int = 80, vocab: int = VOCAB_SIZE,
                with_verifier: bool = True) -> TajweedQuartzNet:
    return TajweedQuartzNet(ModelSpec.default(n_mels, vocab), with_verifier)


if __name__ == "__main__":
    m = build_model()
    print(m)
    print("params:", m.num_parameters())
    print("stride:", m.total_stride, "RF frames:", m.receptive_field_frames)
    x = torch.randn(2, 80, 800)
    lp, enc, ol = m(x, torch.tensor([800, 640]))
    print("log_probs", tuple(lp.shape), "enc", tuple(enc.shape), "lens", ol.tolist())

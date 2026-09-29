import torch
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tokens import VOCAB, VOCAB_SIZE
from src.model.tajweed_quartznet import build_model
import src.training.train as T

paths = T.list_clip_jsons(Path("data/everyayah_extracted"), "train", limit=20)
ds = T.ClipDataset(paths, train=False)   # no augmentation, for inspection
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, collate_fn=T.collate_ctc)

ck = torch.load("exp/light/best.pt", map_location="cpu")
model = build_model(80, VOCAB_SIZE, with_verifier=False)
model.load_state_dict(ck["model"])
model.eval()

x, fl, y, tl, uids = next(iter(dl))
with torch.no_grad():
    logp, enc, ol = model(x, fl)
hyps = T.greedy_decode_batch(logp, ol)

off = 0
for i, L in enumerate(tl.tolist()):
    ref_ids = y[off:off+L].tolist(); off += L
    print(uids[i])
    print("  ref:", [VOCAB[t] for t in ref_ids])
    print("  hyp:", [VOCAB[t] for t in hyps[i]] if hyps[i] else "(EMPTY)")


print(f"Input shape (x):        {x.shape}")        # Expected: (B, C, T) or (B, T, C)
print(f"Logp shape (logp):     {logp.shape}")     # Expected: (B, T, V) or (T, B, V)
print(f"Output lengths (ol):   {ol.tolist()}")

# Print raw predictions for sample 0 (before CTC collapse/blank removal)
if logp.size(0) == x.size(0):  # (B, T, V)
    raw_ids = logp.argmax(dim=-1)[0].tolist()
else:                          # (T, B, V)
    raw_ids = logp.transpose(0, 1).argmax(dim=-1)[0].tolist()

print("\nFirst 40 raw frame predictions (Sample 0):")
print([VOCAB[idx] if idx < len(VOCAB) else f"UNK_{idx}" for idx in raw_ids[:40]])
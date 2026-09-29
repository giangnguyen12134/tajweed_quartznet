from __future__ import annotations
import os 
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional, Set, Tuple
 
import numpy as np
import soundfile as sf
from datasets import Audio, DatasetDict, load_from_disk
from tqdm import tqdm
 

TARGET_DIR = r"f:\VMX\AiProjects\Quran_ASR\data\cache"

os.environ["HF_HOME"] = TARGET_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(TARGET_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(TARGET_DIR, "datasets")

os.environ["TMP"] = os.path.join(TARGET_DIR, "tmp")
os.environ["TEMP"] = os.path.join(TARGET_DIR, "tmp")
os.environ["TMPDIR"] = os.path.join(TARGET_DIR, "tmp")

LOCAL_DIR = "./data/everyayah"
EXPORT_DIR = r"f:\VMX\AiProjects\Quran_ASR\data\everyayah_extracted"


def sanitize(s: str) -> str:
    """Make a string safe to use as (part of) a filename."""
    s = re.sub(r"\s+", "_", (s or "").strip())
    s = re.sub(r"[^A-Za-z0-9_.\-]", "", s)
    return s or "unknown"
 
 
def decode_with_torchcodec(raw: dict) -> Tuple[np.ndarray, int]:
    """
    raw: the {"bytes": ..., "path": ...} dict `datasets` gives you for an
    `Audio(decode=False)` column -- exactly one of the two keys is populated,
    depending on how the dataset stored its audio (embedded vs external file).
 
    Returns (samples, sample_rate): samples as a (num_samples, num_channels)
    float32 array, ready to hand to soundfile.write.
    """
    from torchcodec.decoders import AudioDecoder
 
    if raw.get("bytes") is not None:
        source = raw["bytes"]              # AudioDecoder accepts raw bytes directly
    elif raw.get("path") is not None:
        source = raw["path"]
    else:
        raise ValueError("audio row has neither 'bytes' nor 'path' populated")
 
    decoder = AudioDecoder(source)
    samples = decoder.get_all_samples()     # AudioSamples: .data (C, N) tensor, .sample_rate
    data = samples.data.numpy()              # (num_channels, num_samples), float32 in [-1, 1]
    return data.T, samples.sample_rate        # -> (num_samples, num_channels)
 
 
def extract_split(ds, split_name: str, out_dir: Path, limit: Optional[int],
                  reciters: Optional[Set[str]], start: int, end: Optional[int],
                  decode_fn=decode_with_torchcodec) -> Tuple[int, int]:
    """Returns (n_written, n_failed)."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "metadata.tsv"
 
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))
 
    n = len(ds)
    end = n if end is None else min(end, n)
    stop = end if limit is None else min(end, start + limit)
    indices = range(start, stop)
 
    write_header = not manifest_path.exists()
    n_written = n_failed = 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf, delimiter="\t")
        if write_header:
            writer.writerow(["id", "text", "reciter", "duration_s",
                             "sampling_rate", "wav_path"])
 
        for i in tqdm(indices, desc=split_name, unit="clip"):
            row = ds[i]
            reciter = row.get("reciter") or "unknown"
            if reciters and reciter not in reciters:
                continue
 
            uid = f"{sanitize(reciter)}-{i:07d}"
            wav_path = split_dir / f"{uid}.wav"
            txt_path = split_dir / f"{uid}.txt"
 
            if wav_path.exists() and txt_path.exists():
                continue   # resumable: a re-run skips clips already extracted
 
            try:
                samples, sr = decode_fn(row["audio"])
            except Exception as e:                          # noqa: BLE001
                print(f"  [{uid}] audio decode failed: {e}", file=sys.stderr)
                n_failed += 1
                continue
 
            sf.write(str(wav_path), samples, sr, subtype="PCM_16")
            text = row.get("text") or ""
            txt_path.write_text(text, encoding="utf-8")
 
            duration_s = samples.shape[0] / sr
            writer.writerow([uid, text, reciter, f"{duration_s:.3f}",
                             sr, str(wav_path)])
            n_written += 1
 
    return n_written, n_failed
 
 
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", default=LOCAL_DIR,
                    help="path previously passed to save_to_disk() / load_from_disk()")
    ap.add_argument("--out-dir", default=EXPORT_DIR)
    ap.add_argument("--splits", nargs="*", default=None,
                    help="which splits to extract (default: all present)")
    ap.add_argument("--reciters", nargs="*", default=None,
                    help="only extract these reciters (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of clips per split (useful for a "
                         "quick smoke test before a full run)")
    ap.add_argument("--start", type=int, default=0,
                    help="start index within each split -- combine with "
                         "--end to shard a split across parallel processes")
    ap.add_argument("--end", type=int, default=None,
                    help="end index within each split (exclusive)")
    args = ap.parse_args()
 
    print(f"loading dataset from {args.local_dir} ...")
    dataset = load_from_disk(args.local_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reciters = set(args.reciters) if args.reciters else None
 
    total_written = total_failed = 0
    if isinstance(dataset, DatasetDict):
        splits = args.splits or list(dataset.keys())
        for split_name in splits:
            if split_name not in dataset:
                print(f"WARNING: split {split_name!r} not found in dataset "
                     f"(have {list(dataset.keys())}), skipping", file=sys.stderr)
                continue
            w, f = extract_split(dataset[split_name], split_name, out_dir,
                                 args.limit, reciters, args.start, args.end)
            total_written += w
            total_failed += f
    else:
        total_written, total_failed = extract_split(
            dataset, "data", out_dir, args.limit, reciters, args.start, args.end)
 
    print(f"\ndone: {total_written} clips written, {total_failed} failed.")
    print(f"output under {out_dir}/<split>/<id>.wav + .txt, "
         f"with a metadata.tsv manifest per split.")
 
 
if __name__ == "__main__":
    main()
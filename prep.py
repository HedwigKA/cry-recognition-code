"""Build the window-level dataset from Donate-a-Cry and ESC-50.

Usage: python prep.py --dac <donateacry-corpus dir> --esc <ESC-50 dir> --out data.npz
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from features import load_audio, logmel, rms_dbfs, windows, SILENCE_DBFS

REASONS = ["hungry", "belly_pain", "burping", "discomfort", "tired"]


def clips(dac_dir: str, esc_dir: str) -> pd.DataFrame:
    rows = []
    base = os.path.join(dac_dir, "donateacry_corpus_cleaned_and_updated_data")
    for reason in REASONS:
        for p in sorted(glob.glob(os.path.join(base, reason, "*.wav"))):
            name = os.path.basename(p)
            rows.append(dict(path=p, source="dac", cry=1, reason=reason,
                             esc_category="", group="dac_" + name[:36]))  # app-instance ID
    meta = pd.read_csv(os.path.join(esc_dir, "meta", "esc50.csv"))
    for r in meta.itertuples():
        cry = int(r.category == "crying_baby")
        rows.append(dict(path=os.path.join(esc_dir, "audio", r.filename), source="esc",
                         cry=cry, reason="", esc_category=r.category,
                         group=f"esc_{r.src_file}"))  # same Freesound source stays together
    df = pd.DataFrame(rows)
    df["clip_id"] = np.arange(len(df))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dac", required=True)
    ap.add_argument("--esc", required=True)
    ap.add_argument("--out", default="data.npz")
    a = ap.parse_args()

    df = clips(a.dac, a.esc)
    feats, clip_ids, loud = [], [], []
    for r in df.itertuples():
        w = windows(load_audio(r.path))
        feats.append(logmel(w).astype(np.float16))
        clip_ids.append(np.full(len(w), r.clip_id))
        loud.append(rms_dbfs(w))
    X = np.concatenate(feats)
    clip_id = np.concatenate(clip_ids)
    db = np.concatenate(loud)
    np.savez_compressed(a.out, X=X, clip_id=clip_id, dbfs=db)
    df.to_csv(a.out.replace(".npz", "_clips.csv"), index=False)
    silent = db < SILENCE_DBFS
    print(f"clips={len(df)} windows={len(X)} shape={X.shape[1:]} "
          f"silent windows={silent.sum()} ({silent.mean():.1%})")
    print(df.groupby(["source", "cry"]).size())


if __name__ == "__main__":
    main()

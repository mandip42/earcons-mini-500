#!/usr/bin/env python3
"""
Recompute a minimal set of spectral features for WAV files and merge into metadata CSV.
This is optional; the generator already writes features.
"""
import argparse, os, csv, math, wave, struct
from pathlib import Path
import numpy as np
import pandas as pd

SR = 48000

def read_wav_mono(path: Path):
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        sr = wf.getframerate()
        n = wf.getnframes()
        data = wf.readframes(n)
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
    return sr, x

def spectral_features(x, sr=SR):
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x*w)) + 1e-12
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    sc = float(np.sum(freqs * X) / np.sum(X))
    bw = float(np.sqrt(np.sum(((freqs - sc)**2) * X) / np.sum(X)))
    zc = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    return sc, bw, zc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="audio")
    ap.add_argument("--meta_in", type=str, default="metadata/metadata.csv")
    ap.add_argument("--meta_out", type=str, default="metadata/metadata.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.meta_in)
    rows = []
    for i, r in df.iterrows():
        f = Path(args.indir) / r["file"]
        if not f.exists():
            continue
        sr, x = read_wav_mono(f)
        sc, bw, zc = spectral_features(x, sr)
        r["spec_centroid_hz"] = round(sc, 3)
        r["bandwidth_hz"] = round(bw, 3)
        r["zcr"] = round(zc, 6)
        rows.append(r)

    out = pd.DataFrame(rows, columns=df.columns)
    out.to_csv(args.meta_out, index=False)
    print(f"Updated features for {len(rows)} files -> {args.meta_out}")

if __name__ == "__main__":
    main()

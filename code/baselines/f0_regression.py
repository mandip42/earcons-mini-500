#!/usr/bin/env python3
"""
f0 regression baseline using librosa.yin on single-tone items.
Requires: librosa, numpy, pandas, scikit-learn.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, librosa
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def estimate_f0(y, sr):
    f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=str, required=True)
    ap.add_argument("--meta", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    df = df[(df["chord"]=="single")]  # restrict to single tones

    X_train, y_train, X_test, y_test = [], [], [], []
    for _, r in df.iterrows():
        y, sr = librosa.load(Path(args.audio_dir)/r["file"], sr=48000, mono=True)
        f0_est = estimate_f0(y, sr)
        if np.isnan(f0_est): 
            continue
        if r["split"] == "test":
            X_test.append([f0_est]); y_test.append(r["f0_hz"])
        else:
            X_train.append([f0_est]); y_train.append(r["f0_hz"])

    reg = LinearRegression().fit(np.array(X_train), np.array(y_train))
    pred = reg.predict(np.array(X_test))
    mae = mean_absolute_error(y_test, pred)
    print(f"Test MAE (Hz): {mae:.2f} (n={len(y_test)})")

    Path("baselines").mkdir(exist_ok=True, parents=True)
    with open("baselines/results_f0_regression.json","w") as f:
        json.dump({"test_mae_hz": float(mae), "n": int(len(y_test))}, f, indent=2)

if __name__ == "__main__":
    main()

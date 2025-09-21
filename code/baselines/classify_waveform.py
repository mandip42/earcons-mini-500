#!/usr/bin/env python3
"""
Tiny waveform-family classifier baseline.
Requires: librosa, scikit-learn, numpy, pandas.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_item(path):
    y, sr = librosa.load(path, sr=48000, mono=True)
    # log-mel spectrogram mean+var pooling
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmin=50, fmax=8000)
    F = np.log1p(S)
    return np.hstack([F.mean(axis=1), F.std(axis=1)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=str, required=True)
    ap.add_argument("--meta", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    df = df.copy()
    # collapse fm variants to "fm"
    df["waveform_family"] = df["waveform"].apply(lambda w: "fm" if w.startswith("fm_") else w)

    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    for _, r in df.iterrows():
        x = load_item(Path(args.audio_dir)/r["file"])
        y = r["waveform_family"]
        if r["split"] == "train": X_train.append(x); y_train.append(y)
        elif r["split"] == "val": X_val.append(x); y_val.append(y)
        else: X_test.append(x); y_test.append(y)
    X_train, X_val, X_test = map(lambda L: np.vstack(L), [X_train, X_val, X_test])

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    Path("baselines").mkdir(exist_ok=True, parents=True)
    with open("baselines/results_classify_waveform.json","w") as f:
        json.dump({"test_accuracy": float(acc)}, f, indent=2)

if __name__ == "__main__":
    main()



import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

def pick_examples(df: pd.DataFrame):
    # Collapse fm_* to "fm"
    df = df.copy()
    df["waveform_family"] = df["waveform"].apply(lambda w: "fm" if str(w).startswith("fm_") else w)

    # Target selection: 6 panels (waveform × am_rate)
    targets = [
        ("sine", 0), ("sine", 8),
        ("square", 0), ("square", 30),
        ("triangle", 0), ("fm", 8),
    ]

    picks = []
    for wf, am in targets:
        sub = df[(df["waveform_family"] == wf) & (df["am_rate_hz"] == am)]
        if len(sub) == 0:
            # fallback: just any with the waveform family
            sub = df[(df["waveform_family"] == wf)]
        if len(sub) == 0:
            continue
        # prefer dry to keep spectrograms clean
        sub = sub.sort_values(by=["reverb", "am_depth"]).reset_index(drop=True)
        picks.append(sub.iloc[0])
    return pd.DataFrame(picks)

def make_panel(ax, wav_path, title, sr_target=48000):
    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmin=50, fmax=8000)
    S_log = np.log1p(S)  # simple log
    img = librosa.display.specshow(S_log, sr=sr, x_axis=None, y_axis=None, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=str, default="audio")
    ap.add_argument("--meta", type=str, default="metadata/metadata.csv")
    ap.add_argument("--out", type=str, default="docs/figures/spectrogram_grid.png")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    meta = pd.read_csv(args.meta)

    picks = pick_examples(meta)
    if len(picks) == 0:
        raise SystemExit("No examples found. Did you generate audio and metadata first?")

    n = len(picks)
    rows, cols = 2, 3  # 6 panels
    fig_w, fig_h = 11, 6  # inches
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    for ax, (_, r) in zip(axes, picks.iterrows()):
        wav = audio_dir / r["file"]
        title = f'{r["waveform"]} | AM {r["am_rate_hz"]} Hz | {r["reverb"]}'
        make_panel(ax, wav, title)

    # If fewer than 6, hide extra axes
    for k in range(len(picks), rows*cols):
        axes[k].axis("off")

    fig.suptitle("BeepBank-500: Log-mel Spectrograms (examples)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()

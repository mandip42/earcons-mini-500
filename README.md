# BeepBank-500 (earcons-mini-500)

**BeepBank-500** is a compact, fully synthetic earcon/alert mini‑dataset (≈300–500 clips) for UI sound research.
It contains short tones and triads generated from a controlled parameter grid (waveform family, f0, duration,
envelope, amplitude modulation, and simple Schroeder-style reverbs). The dataset ships with a metadata schema,
lightweight baselines, and a data note template for arXiv. Audio is intended for release under CC0-1.0 (public domain).
Code is MIT-licensed.

> Why this dataset? Reproducible, rights‑clean, and tiny enough to bundle in a 2–3 day sprint as a citable asset.
> Typical tasks: earcon classification, timbre analysis, f0 regression, onset detection, psychoacoustic proxies.

## Quick start (2–3 day plan)

### 0) Create environment
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Generate audio + metadata (≈400 clips by default)
```bash
python code/generate_earcons.py --outdir audio --meta metadata/metadata.csv --seed 13 --target_n 400
```
Optional: change `--target_n` to 500 for a bigger set.

### 2) (Optional) Recompute/augment features later
```bash
python code/compute_features.py --indir audio --meta_in metadata/metadata.csv --meta_out metadata/metadata.csv
```

### 3) Run tiny baselines
```bash
# Waveform family classifier (sine/square/triangle/fm_*)
python code/baselines/classify_waveform.py --audio_dir audio --meta metadata/metadata.csv

# f0 regression MAE (Hz) on single‑tone items only
python code/baselines/f0_regression.py --audio_dir audio --meta metadata/metadata.csv
```

### 4) Archive and publish
- Create a **Zenodo** record (v1.0.0) with `/audio`, `/metadata`, `/code`, `README.md`, `LICENSE`, `CITATION.cff`.
- Mirror code on **GitHub**; connect Zenodo badge; tag a release.
- Submit the **arXiv** data note using `docs/data_note_arxiv.tex` (subject: eess.AS). Link the Zenodo DOI.

## Folder layout
```
earcons-mini-500/
├─ audio/                    # WAV files (mono, 48kHz, 16-bit PCM)
├─ metadata/
│  ├─ metadata.csv           # one row per file with parameters and features
│  └─ LICENSES.md            # audio license statement (CC0-1.0)
├─ code/
│  ├─ generate_earcons.py    # main generator (no external DSP deps)
│  ├─ compute_features.py    # recompute features if needed
│  └─ baselines/
│     ├─ classify_waveform.py
│     └─ f0_regression.py
├─ docs/
│  ├─ data_note_arxiv.tex    # 2–3k word data note skeleton
│  └─ figures/
├─ requirements.txt
├─ CITATION.cff
├─ LICENSE                   # MIT for code
├─ CHANGELOG.md
└─ README.md
```

## Dataset design notes
- **Waveforms**: sine, square, triangle, FM (2:1 & 3:2), optional chords (major/minor triads).
- **Durations**: 100/250/500 ms; **Envelopes**: fast, medium, percussive.
- **AM**: none, 8 Hz (0.3), 30 Hz (0.5).
- **Reverbs**: simple Schroeder-style small (~0.3 s) and medium (~0.6 s).
- **Normalization**: RMS target with peak cap at −1 dBFS. (If `pyloudnorm` is installed, LUFS can be computed; normalization remains RMS‑based by default.)
- **Metadata**: includes generation params plus features (peak/rms dBFS, centroid, bandwidth, zcr, proxies).
- **Splits**: deterministic hash-based (train/val/test).

## Licensing
- **Audio**: CC0‑1.0 (public domain dedication). Keep attribution unnecessary.
- **Code**: MIT License (see `LICENSE`).
- If you later add CC‑BY assets, list them in `metadata/LICENSES.md` with full attribution and URLs.

## How to cite
`(https://doi.org/10.5281/zenodo.17172016)`. Also cite the arXiv data note.

---

**Maintainer**: Mandip Goswami  
**Scope**: niche psychoacoustic/UI earcon research  
**Keywords**: earcon, alarm, psychoacoustics, timbre, AM, ADSR, reverb

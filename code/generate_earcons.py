#!/usr/bin/env python3
"""
Generate a compact set of synthetic earcons and write metadata.csv.
No external DSP deps beyond numpy/pandas. WAV writing uses stdlib `wave`.
Optional: if `pyloudnorm` is installed, LUFS will be computed (not used for normalization).

Example:
    python code/generate_earcons.py --outdir audio --meta metadata/metadata.csv --seed 13 --target_n 400
"""
import os, csv, math, argparse, hashlib, random
import numpy as np
import pandas as pd
from pathlib import Path
import wave, struct

try:
    import pyloudnorm as pyln  # optional
    HAVE_LOUDNORM = True
except Exception:
    HAVE_LOUDNORM = False

SR = 48000

def _hash_split(name: str) -> str:
    """Deterministic train/val/test split based on file name hash."""
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 100
    if h < 10:
        return "test"
    elif h < 20:
        return "val"
    return "train"

def write_wav_int16(path: Path, data: np.ndarray, sr: int = SR):
    """Write mono int16 PCM WAV using stdlib wave."""
    # peak cap at -1 dBFS
    peak = np.max(np.abs(data) + 1e-12)
    if peak > 10**(-1/20):
        data = data * (10**(-1/20) / peak)
    # scale
    data_i16 = np.clip(data, -1.0, 1.0)
    data_i16 = (data_i16 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(data_i16.tobytes())

def dbfs_rms(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)

def dbfs_peak(x: np.ndarray) -> float:
    peak = np.max(np.abs(x) + 1e-12)
    return 20 * np.log10(peak + 1e-12)

def normalize_rms(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    target = 10**(target_dbfs/20)
    if rms < 1e-9:
        return x
    return x * (target / rms)

def osc_sine(f0, t):
    return np.sin(2*np.pi*f0*t)

def osc_square(f0, t):
    # simple (non-bandlimited) square via sign(sin)
    return np.sign(np.sin(2*np.pi*f0*t))

def osc_triangle(f0, t):
    # triangle via arcsin of sine (bandlimited-ish)
    return (2/np.pi) * np.arcsin(np.sin(2*np.pi*f0*t))

def osc_fm(fc, fm, index, t):
    return np.sin(2*np.pi*fc*t + index*np.sin(2*np.pi*fm*t))

def apply_am(x, rate_hz, depth, t):
    if rate_hz <= 0 or depth <= 0:
        return x
    mod = 1.0 + depth * np.sin(2*np.pi*rate_hz*t)
    return x * mod

def adsr_envelope(t, dur_s, env_type: str):
    # Three presets
    if env_type == "adsr_fast":
        A, D, S, R, S_level = 0.005, 0.03, 0.3, 0.05, 0.6
    elif env_type == "adsr_med":
        A, D, S, R, S_level = 0.02, 0.10, 0.5, 0.1, 0.7
    else:  # percussive
        A, D, S, R, S_level = 0.001, 0.05, 0.2, 0.05, 0.4
    atk = np.clip(t/A, 0, 1)
    dec = np.clip((1 - (t - A)/D), 0, 1)
    sus = np.ones_like(t) * S_level
    rel = np.clip((dur_s - t)/R, 0, 1)
    env = np.zeros_like(t)
    env = np.where(t < A, atk, env)
    env = np.where((t >= A) & (t < A + D), S_level + (1 - S_level)*dec, env)
    env = np.where((t >= A + D) & (t < dur_s - R), sus, env)
    env = np.where(t >= dur_s - R, sus * rel, env)
    return env

def schroeder_reverb(x, sr=SR, kind="small"):
    """Very small Schroeder-like reverb (light CPU)."""
    # simple fixed comb + allpass chain parameters
    if kind == "small":
        comb_delays = [0.0297, 0.0371, 0.0411]
        comb_gains  = [0.805, 0.827, 0.783]
        ap_delays   = [0.005, 0.0017]
        ap_gains    = [0.7, 0.7]
    else:
        comb_delays = [0.045, 0.055, 0.065]
        comb_gains  = [0.82, 0.84, 0.80]
        ap_delays   = [0.012, 0.0043]
        ap_gains    = [0.7, 0.7]
    y = np.zeros_like(x)
    for d, g in zip(comb_delays, comb_gains):
        n = int(sr*d)
        buf = np.zeros_like(x)
        for i in range(n, len(x)):
            buf[i] = x[i] + g * buf[i-n]
        y += buf
    # all-pass
    for d, g in zip(ap_delays, ap_gains):
        n = int(sr*d)
        buf = np.zeros_like(y)
        for i in range(n, len(y)):
            buf[i] = -g*y[i] + y[i-n] + g*buf[i-n]
        y = buf
    # mix wet/dry
    out = 0.3*y + x
    return out

def spectral_features(x, sr=SR):
    # Hann window full signal
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x*w))
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    X += 1e-12
    sc = float(np.sum(freqs * X) / np.sum(X))
    bw = float(np.sqrt(np.sum(((freqs - sc)**2) * X) / np.sum(X)))
    # zero-crossing rate
    zc = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    return sc, bw, zc

def build(params, seed, target_n):
    random.seed(seed)
    np.random.seed(seed)
    # Parameter space
    waveforms = ["sine", "square", "triangle", "fm_2to1", "fm_3to2"]
    f0s = [350, 500, 750, 1000]
    durs_ms = [100, 250, 500]
    envs = ["adsr_fast", "adsr_med", "percussive"]
    am_rates = [0, 8, 30]
    am_depths = [0.0, 0.3, 0.5]
    chords = ["single", "major", "minor"]
    reverbs = ["dry", "rir_small", "rir_medium"]

    combos = []
    for wf in waveforms:
        for f0 in f0s:
            for dur in durs_ms:
                for env in envs:
                    for amr in am_rates:
                        for amd in am_depths:
                            for ch in chords:
                                for rv in reverbs:
                                    combos.append((wf, f0, dur, env, amr, amd, ch, rv))

    random.shuffle(combos)
    if target_n is not None:
        combos = combos[:target_n]

    rows = []
    for idx, (wf, f0, dur, env, amr, amd, ch, rv) in enumerate(combos):
        dur_s = dur / 1000.0
        n = int(SR * dur_s)
        t = np.arange(n) / SR

        # build tone or chord
        def tone(wform, f, tvec):
            if wform == "sine":
                return osc_sine(f, tvec)
            elif wform == "square":
                return osc_square(f, tvec)
            elif wform == "triangle":
                return osc_triangle(f, tvec)
            elif wform == "fm_2to1":
                return osc_fm(f, 2*f, index=2.0, t=tvec)
            else:
                return osc_fm(f, 1.5*f, index=2.5, t=tvec)

        if ch == "single":
            x = tone(wf, f0, t)
        else:
            if ch == "major":
                freqs = [f0, f0*2**(4/12), f0*2**(7/12)]
            else:
                freqs = [f0, f0*2**(3/12), f0*2**(7/12)]
            x = sum(tone(wf, f, t) for f in freqs) / 3.0

        # apply AM before envelope (either order is fine here)
        x = apply_am(x, amr, amd, t)
        # envelope
        env_curve = adsr_envelope(t, dur_s, env)
        x = x * env_curve
        # normalize RMS and cap peaks
        x = normalize_rms(x, target_dbfs=-20.0)

        # reverb
        if rv == "rir_small":
            x = schroeder_reverb(x, SR, kind="small")
        elif rv == "rir_medium":
            x = schroeder_reverb(x, SR, kind="medium")

        # features
        peak_db = dbfs_peak(x)
        rms_db = dbfs_rms(x)
        sc, bw, zc = spectral_features(x, SR)
        if HAVE_LOUDNORM:
            meter = pyln.Meter(SR)
            lufs = float(meter.integrated_loudness(x.astype(np.float64)))
        else:
            lufs = None

        # filename and split
        fname = f"bb_{idx:04d}_{wf}_f{f0}_d{dur}_{env}_am{amr}x{amd}_{ch}_{rv}.wav"
        split = _hash_split(fname)

        yield fname, x, {
            "file": fname, "split": split, "sr_hz": SR, "bit_depth": 16,
            "duration_ms": dur, "peak_dbfs": round(peak_db, 3),
            "lufs": (round(lufs, 3) if lufs is not None else ""),
            "rms_dbfs": round(rms_db, 3), "waveform": wf, "f0_hz": f0,
            "chord": ch, "am_rate_hz": amr, "am_depth": amd,
            "envelope": env, "reverb": rv, "spec_centroid_hz": round(sc, 3),
            "bandwidth_hz": round(bw, 3), "zcr": round(zc, 6),
            "inharmonicity_proxy": (0 if ch == "single" else 1),
            "roughness_proxy": amd, "attack_ms": 0, "release_ms": 0,
            "seed": seed, "version": "1.0.0"
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="audio")
    ap.add_argument("--meta", type=str, default="metadata/metadata.csv")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--target_n", type=int, default=400, help="number of items to generate")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta); meta_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fname, audio, row in build(vars(args), args.seed, args.target_n):
        write_wav_int16(outdir / fname, audio, SR)
        rows.append(row)

    # write metadata CSV with fixed column order
    cols = ["file","split","sr_hz","bit_depth","duration_ms","peak_dbfs","lufs","rms_dbfs",
            "waveform","f0_hz","chord","am_rate_hz","am_depth","envelope","reverb",
            "spec_centroid_hz","bandwidth_hz","zcr","inharmonicity_proxy","roughness_proxy",
            "attack_ms","release_ms","seed","version"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(meta_path, index=False)
    print(f"Wrote {len(rows)} files to {outdir} and metadata to {meta_path}")

if __name__ == "__main__":
    main()

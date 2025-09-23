---
title: "BeepBank: A Parametric Earcon Toolkit and Mini-Benchmark for Reproducible Audio Experiments"
tags:
  - audio
  - machine learning
  - signal processing
  - earcons
  - dataset
authors:
  - name: Mandip Goswami
    orcid: 0009-0004-1363-4368
    affiliation: 1
affiliations:
  - name: Amazon, USA (work done outside regular employment; views are the author's own)
    index: 1
date: 2025-09-23
bibliography: paper.bib
---

# Summary
**BeepBank** is a small, fully synthetic **earcon/alert** toolkit and mini-benchmark enabling controlled, reproducible audio experiments. It provides: (i) a parametric **synthesis engine** for short non-speech tones with explicit control over waveform family, fundamental frequency, duration, envelopes, amplitude modulation, and room simulation; (ii) a compact **reference corpus** (hundreds of WAV files) with ground-truth metadata; and (iii) minimal **baseline tasks** (waveform-family classification and f₀ regression) for quick verification and fair comparisons. Audio is released under **CC0-1.0** and the software under a permissive **OSI license**.

# Statement of need
Audio ML and HCI work often needs **simple, controllable, rights-clean** non-speech sounds for unit tests, pedagogy, and model sanity checks. Existing open corpora focus on speech, music, or environmental sounds; openly licensed **earcon** resources with transparent parametric control are scarce. BeepBank fills this gap with a **small, well-documented, citable** resource that:
- downloads and runs in minutes,
- supports **factor-swept** experiments (e.g., robustness to AM depth or reverberation),
- enables **leakage-aware** splits by parameter families, and
- provides a **common baseline** for comparable results.

# Software description
**Design goals:** rights-clean; small but complete; deterministic; pedagogically useful; extensible.

**Core functionality**
- **Parametric synthesis:** waveform families (sine/square/triangle/FM), f₀ ranges, durations, ADSR presets, optional AM (rate/depth), and three room conditions (dry/small/medium) via lightweight algorithmic reverberation.
- **Metadata emission:** `metadata.csv` with ID, split, labels/parameters (f₀, duration, envelope, AM, reverb) and level statistics (RMS/peak/crest).
- **Deterministic splits:** rule- or hash-based splitting to minimize information leakage.
- **Baselines:** reproducible waveform-family classification and f₀ regression to verify pipelines.

**Project structure:** documentation, quickstart, generation recipes, QC scripts, and a **dataset card** (intended use, limitations, licensing).

# Quality control
Automated checks ensure:
- **Audio integrity:** sample rate/bit depth, channel count, duration bounds, anti-click fades.
- **Signal safety:** no clipping, bounded crest factor, DC-offset checks.
- **Determinism:** fixed seeds and recipe hashing to regenerate the reference set exactly.
- **De-duplication:** heuristics to avoid near-identical configurations.
A brief listening pass verified envelope correctness, AM audibility, and consistent reverb character.

# Example use cases
1. **Benchmarking:** probe timbre embeddings or f₀ estimators while sweeping one factor at a time.
2. **Education:** lab exercises on envelopes, modulation, and room effects using ground-truth parameters.
3. **Prototype → production:** prototype on BeepBank (rights-clean), then fine-tune on proprietary alarms with unchanged scaffolding.
4. **Reproducible ablations:** publish factor-isolated results that others can replicate exactly.

# State of the field
Open audio datasets emphasize speech, environmental scenes, or music. BeepBank complements them by focusing on **earcons/alerts** with **explicit parametric control** and compact size for rapid iteration [@earcons-foundations; @auditory-icons; @dcase-overview]. It doubles as a teaching corpus and a toy benchmark for sanity-checking model behavior.

# Limitations and intended scope
BeepBank does not replace real-world alarms; domain gaps remain (recording chains, noise, context). It is intentionally **small** and limited to **single-event** tones to maximize accessibility and speed.

# Availability and licensing
- **Repository:** https://github.com/mandip42/earcons-mini-500
- **Version:** v0.1.0 
- **Software license:** MIT (or Apache-2.0/BSD-3-Clause) at repo root
- **Audio/data license:** CC0-1.0
- **Archival DOI:** https://doi.org/10.5281/zenodo.17172015
- **Preprint:** https://doi.org/10.48550/arXiv.2509.17277

# Author contributions
Conceptualization, methodology, software, data curation, writing – original draft, project administration: **Mandip Goswami**.

# Acknowledgements
Thanks to community reviewers for feedback on dataset design and documentation.

# Ethical & competing interests
No human-subject data. Work performed outside regular employment duties; no competing interests.

# References

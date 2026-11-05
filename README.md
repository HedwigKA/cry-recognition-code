# Cry recognition – Smart Neonatal Incubator Sentinel (offline experiments, E3)

Two-stage infant cry recognition following paper Ch. 3.4:
stage 1 = cry / non-cry / silence, stage 2 = parent-reported cry reason.

## Setup

```bash
pip install tensorflow-cpu scikit-learn soundfile scipy pandas matplotlib
git clone --depth 1 https://github.com/gveres/donateacry-corpus.git
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git
```

## Run

```bash
python prep.py --dac donateacry-corpus --esc ESC-50 --out data.npz   # ~1 min
python stage1.py             # cry vs non-cry vs silence, ~30 min on 2 CPU cores
python stage2.py             # cry reason, 3 label schemes x weighted/unweighted x 3 seeds
python plots.py              # figures for Chapter 4
python export_model.py       # train & export final deployable Stage 1 & Stage 2 models to models/
python sentinel_inference.py # run end-to-end Stage 1 + Stage 2 + Incubator Sensor Context Fusion demo
```

> Dokumentasi lengkap berbahasa Indonesia (penjelasan arsitektur, hasil eksperimen, penggabungan sensor inkubator, dan panduan integrasi C/C++ mikrokontroler) tersedia di [`DOKUMENTASI.md`](DOKUMENTASI.md).

## Files

| File | Role |
|---|---|
| `features.py` | Load audio, resample to 8 kHz, 1 s windows (0.5 s hop), 40-band log-mel (25 ms / 10 ms) |
| `prep.py` | Builds `data.npz` (window features) and `data_clips.csv` (labels, groups) |
| `train_lib.py` | 1D CNN from the paper (Conv1D 8 → 16, kernel 3, max-pool, dropout 0.25, softmax), augmentation, weighted loss, int8 export |
| `baseline.py` | Baseline after Fahmi et al. (2020): energy statistics + spectral centroid/bandwidth, k-NN (k = 5) |
| `stage1.py` | Stage 1 CV, int8 evaluation, cross-source check, false positives per ESC-50 class |
| `stage2.py` | Stage 2 CV, baselines, per-class metrics, confusion matrices |
| `plots.py` | Confusion matrices and macro-F1 bar chart |
| `export_model.py` | Trains final Stage 1 & Stage 2 models and exports `.keras`, `.tflite` (int8), `.h` (C array), and `*_norm.json` to `models/` |
| `sentinel_inference.py` | End-to-end two-stage TFLite INT8 inference combined with real-time incubator telemetry (temperature, humidity, wet diaper, feeding timer, awake duration) |
| `DOKUMENTASI.md` | Full project & experiment documentation in Indonesian |

## Design decisions

- **Labels.** ESC-50 `crying_baby` (40 clips) is labelled *cry*, not non-cry. Windows below −50 dBFS are *silence*
  (mostly the digital zero padding in ESC-50 clips).
- **Grouping.** Folds are split by Donate-a-Cry app-instance ID (first 36 characters of the file name) and by
  ESC-50 `src_file`, so clips from the same phone/recording never appear in both training and test.
- **Class weight.** Weighted cross-entropy with w_c = N / (K · n_c), applied as per-sample weights on training windows.
- **Normalisation.** Global mean/std fitted on each training fold only; dB values floored at −100.
- **Augmentation** (training only): random gain, time mask (≤ 15 frames), frequency mask (≤ 5 bands), small noise.
- **Clip decision.** Mean P(class) over non-silent windows. Stage 1 also reports the device rule from the paper
  (any 3 consecutive windows with mean P(cry) ≥ 0.80).
- **Shortcut check.** `dac_only` retrains stage 1 without ESC-50 `crying_baby`; its recall on those clips shows whether
  the model learned "crying" or "Donate-a-Cry recording conditions".

## Deployment note

The log-mel here is computed in NumPy. If the model is deployed through Edge Impulse, train it there with the same
parameters so the on-device MFE block matches; otherwise port `features.logmel` to C. Do not mix the two.

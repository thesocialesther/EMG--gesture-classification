# EMG gesture classification and recorded-signal replay

**Author: Oluwaferanmi Esther Onifade**

A MATLAB workflow that classifies hand gestures from surface electromyography (EMG). It explores recorded muscle signals, extracts windowed features, compares classifiers, and replays predictions over a recorded gesture segment. This work supports an offline prosthetic-control investigation; physical prosthesis actuation is not implemented in this repository.

## Signal processing and classification

The workflow uses the NinaPro DB2 subject 1, exercise 1 recording `S1_E1_A1.mat`, with `emg`, `restimulus`, and `rerepetition` fields. Scripts assume a 2,000 Hz sampling rate.

Features are computed over 200 ms windows (400 samples): root mean square (RMS), mean absolute value (MAV), and waveform length. Training windows must contain a single gesture and a single positive repetition label. Features are log-transformed and standardized using training-only statistics.

The baseline uses diagonal linear discriminant analysis (LDA). Training uses repetitions 1–4 and testing uses repetitions 5–6. Model comparison uses repetitions 1–3 for training and repetition 4 for validation, comparing RMS-only and combined features with diagonal and regularized LDA. Replay applies a trailing 200 ms window every 50 ms.

## Workflow

Run the scripts in `scripts/` in numerical order:

1. `step01`: inspect the recording and gesture signals.
2. `step02`: extract labelled window features.
3. `step03`: train and evaluate the baseline classifier.
4. `step04`: compare feature sets and classifier variants.
5. `step05`: train the selected model.
6. `step06`: replay predictions over recorded EMG.

Later steps depend on artifacts generated earlier. Exported scripts use project-relative paths; `models/` and `results/` are the output locations.

## Requirements and data

Requires MATLAB and Statistics and Machine Learning Toolbox. Obtain the matching recording from [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html) under its access terms and place it in `data/`. Confirm the file fields and sampling rate before running the scripts.

See [DATASETS.md](DATASETS.md) for the recording manifest and setup details. Raw recordings and fitted models are not included in this source export.

## Results and current scope

The repository contains the analysis and training source, but saved classifiers, validation results, and final confusion-matrix figures still need to be added. Numerical accuracy claims should be tied to those saved results and their repetition split.

Replay demonstrates inference on an existing recording. Live acquisition, prosthesis firmware, and physical control evaluation would be separate additions.

# Recorded EMG classification results

These results were extracted on 9 November 2026 from MATLAB artifacts retained on the author's computer at `Documents/MATLAB/EMG/`. They are **saved evaluation values**, not the result of a new independent training run or new hardware experiment.

## Saved evaluation

| Model/artifact | Overall accuracy | Balanced accuracy | Other reported measure |
|---|---:|---:|---|
| RMS-only baseline (`emg_rms_baseline.mat`) | 71.57% | 50.99% | — |
| Selected classifier (`emg_selected_model.mat`) | 79.34% | 70.79% | Movement accuracy 69.35% |

- Baseline file identifies training repetitions **1–4** and test repetitions **5–6**.
- Selected-model evaluation contains an **18 × 18 confusion matrix**.
- Recorded-signal replay artifact contains **206 saved predictions**. Replay is not live EMG acquisition or prosthesis actuation.
- Figures are rounded to two decimal places from values stored in the MATLAB files.
- The saved selected-model result should be independently reproduced using the scripts and the same NinaPro recording before claiming a fully reproducible benchmark.

## Reproduce and verify

1. Obtain the NinaPro DB2 subject 1 recordings subject to the dataset's access terms. Place files in `data/`.
2. Use MATLAB with Statistics and Machine Learning Toolbox.
3. Run `scripts/step01_explore_emg.m` through `scripts/step06_replay_emg.m` in sequence, checking the repetition split and generated files.
4. Compare `overallAccuracy`, `balancedAccuracy`, `movementAccuracy` and `confusionMatrix` in the generated `models/emg_selected_model.mat` against the values above.
5. Keep training, validation and testing repetitions distinct; compare gesture-wise recall and confusion, not accuracy alone.

## Scope and limitations

This is an offline classification and prerecorded-signal replay study. It does **not** establish performance on a wearable real-time device, new participants, different acquisition sessions, embedded inference, or an actuated prosthetic hand. Class imbalance and repetition/session variability may affect generalization.

The original NinaPro recordings are not redistributed here. Local saved MATLAB classifiers are not automatically published because provenance, serialization portability and redistribution permissions should be checked first.

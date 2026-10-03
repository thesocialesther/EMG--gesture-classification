# EMG data setup
The local source recording is S1_E1_A1.mat, associated with the NinaPro DB2 subject-1 exercise files.
Official database instructions: https://ninapro.hevs.ch/instructions/DB2.html
Follow the provider's access and citation instructions. Put S1_E1_A1.mat in data/. The scripts use emg, restimulus, and rerepetition and assume 2000 Hz sampling. Check the source metadata before use.

data/manifest.json records the names, sizes and SHA-256 hashes of the original local exercise files. The scripts currently use exercise 1 only. The first recording is about 151 MB; raw recordings are not committed as ordinary Git files. Obtain them separately rather than treating a placeholder as training data.
